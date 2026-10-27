from __future__ import annotations

import argparse
import csv
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import ROOT as PACKAGE_ROOT, CIF_DIR as DEFAULT_CIF_DIR
from .config import ATOM_INIT as DEFAULT_ATOM_INIT, CHECKPOINTS as DEFAULT_MODELS
from .data import CrystalDataset, collate_graphs
from .model import build_model
from .utils import (
    Normalizer,
    infer_n_conv,
    load_checkpoint,
    move_inputs,
    resolve_device,
)


def read_input_csv(path: Path, limit: int | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "material_id" not in frame.columns:
        raise ValueError("Input CSV must contain a material_id column.")
    frame = frame.copy()
    frame["material_id"] = frame["material_id"].astype(str)
    if frame["material_id"].duplicated().any():
        duplicates = frame.loc[
            frame["material_id"].duplicated(), "material_id"
        ].tolist()
        raise ValueError("Duplicate material_id values: {}".format(duplicates[:5]))
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be at least 1.")
        frame = frame.iloc[:limit].copy()
    if frame.empty:
        raise ValueError("Input CSV contains no rows.")
    return frame


@contextmanager
def prepared_dataset(
    frame: pd.DataFrame,
    cif_dir: Path = DEFAULT_CIF_DIR,
    atom_init: Path = DEFAULT_ATOM_INIT,
) -> Iterator[CrystalDataset]:

    cif_dir = Path(cif_dir).resolve()
    atom_init = Path(atom_init).resolve()
    missing = [
        mid for mid in frame["material_id"] if not (cif_dir / (mid + ".cif")).is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing CIF files for: {}".format(missing[:10]))
    if not atom_init.is_file():
        raise FileNotFoundError("Missing atom_init.json: {}".format(atom_init))

    with tempfile.TemporaryDirectory(prefix="agenet_inference_") as tmp:
        root = Path(tmp)
        with (root / "id_prop.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            for mid in frame["material_id"]:
                writer.writerow([mid, 0.0])
        shutil.copy2(atom_init, root / "atom_init.json")
        for mid in frame["material_id"]:
            source = cif_dir / (mid + ".cif")
            destination = root / source.name
            try:
                destination.symlink_to(source)
            except OSError:
                shutil.copy2(source, destination)
        yield CrystalDataset(str(root))


def load_paper_model(
    prop: str,
    dataset: CrystalDataset,
    device: torch.device,
    checkpoint_path: Path | None = None,
) -> Tuple[torch.nn.Module, Normalizer, Dict]:
    checkpoint_path = Path(checkpoint_path or DEFAULT_MODELS[prop]).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError("Missing checkpoint: {}".format(checkpoint_path))
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint.get("prop") != prop:
        raise ValueError(
            "Checkpoint property mismatch: expected {}, found {}".format(
                prop, checkpoint.get("prop")
            )
        )
    args = checkpoint.get("args", {})
    arch = checkpoint.get("arch", args.get("arch"))
    if not arch:
        raise ValueError("Checkpoint does not record its architecture.")
    fc_edge = int(args.get("fc_edge", 64))
    num_frequencies = int(args.get("num_frequencies", 5))
    n_conv = infer_n_conv(checkpoint["state_dict"])
    model = build_model(
        arch,
        fc_edge=fc_edge,
        num_frequencies=num_frequencies,
    )(dataset, device.type == "cuda", n_conv=n_conv)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    normalizer = Normalizer()
    normalizer.load_state_dict(checkpoint["normalizer"])
    return model, normalizer, checkpoint


def predict_dataframe(
    frame: pd.DataFrame,
    cif_dir: Path = DEFAULT_CIF_DIR,
    atom_init: Path = DEFAULT_ATOM_INIT,
    device: torch.device = torch.device("cpu"),
    batch_size: int = 64,
) -> pd.DataFrame:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    frame = frame.copy()
    frame["material_id"] = frame["material_id"].astype(str)
    with prepared_dataset(frame, cif_dir=cif_dir, atom_init=atom_init) as dataset:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_graphs,
            num_workers=0,
        )
        models = {}
        for prop in ("K", "G"):
            models[prop] = load_paper_model(prop, dataset, device)

        rows = []
        with torch.no_grad():
            for inputs, _, material_ids in loader:
                moved = move_inputs(inputs, device)
                batch_predictions = {}
                for prop in ("K", "G"):
                    model, normalizer, _ = models[prop]
                    normalized = model(*moved)
                    log_values = (
                        normalizer.denorm(normalized.detach().cpu()).view(-1).numpy()
                    )
                    batch_predictions[prop] = log_values
                for index, material_id in enumerate(material_ids):
                    log_k = float(batch_predictions["K"][index])
                    log_g = float(batch_predictions["G"][index])
                    rows.append(
                        {
                            "material_id": str(material_id),
                            "pred_log10_K": log_k,
                            "pred_K_GPa": 10.0**log_k,
                            "pred_log10_G": log_g,
                            "pred_G_GPa": 10.0**log_g,
                        }
                    )

    predictions = pd.DataFrame(rows)
    ordered = frame[["material_id"]].merge(
        predictions,
        on="material_id",
        how="left",
        validate="one_to_one",
    )
    if ordered[["pred_log10_K", "pred_log10_G"]].isna().any().any():
        raise RuntimeError("At least one input material did not receive a prediction.")
    for prop in ("K", "G"):
        true_column = "{}_VRH".format(prop)
        if true_column in frame.columns:
            truth = frame[["material_id", true_column]].copy()
            truth["true_log10_{}".format(prop)] = np.log10(
                truth[true_column].astype(float)
            )
            ordered = ordered.merge(
                truth, on="material_id", how="left", validate="one_to_one"
            )
    return ordered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict K and G with the two paper-locked AGE-Net models."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=PACKAGE_ROOT / "data" / "splits" / "test_B.csv",
        help="CSV containing material_id; K_VRH/G_VRH columns are optional.",
    )
    parser.add_argument("--cif-dir", type=Path, default=DEFAULT_CIF_DIR)
    parser.add_argument("--atom-init", type=Path, default=DEFAULT_ATOM_INIT)
    parser.add_argument(
        "--output", type=Path, default=PACKAGE_ROOT / "outputs" / "predictions.csv"
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--limit", type=int, default=None, help="Optional first-N-row smoke limit."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    frame = read_input_csv(args.input_csv, limit=args.limit)
    predictions = predict_dataframe(
        frame,
        cif_dir=args.cif_dir,
        atom_init=args.atom_init,
        device=device,
        batch_size=args.batch_size,
    )
    if not np.isfinite(predictions[["pred_log10_K", "pred_log10_G"]].to_numpy()).all():
        raise FloatingPointError("Prediction output contains non-finite values.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False)
    print(
        "[ok] device={} rows={} output={}".format(
            device, len(predictions), args.output.resolve()
        )
    )
    for prop in ("K", "G"):
        true_log = "true_log10_{}".format(prop)
        if true_log in predictions.columns:
            mae = float(
                np.mean(
                    np.abs(
                        predictions[true_log]
                        - predictions["pred_log10_{}".format(prop)]
                    )
                )
            )
            print("[metric] {} log10-MAE={:.6f}".format(prop, mae))


if __name__ == "__main__":
    main()
