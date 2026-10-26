"""Check the released data, model weights, inference and a backward step."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import ROOT as PACKAGE_ROOT, SPLIT_DIR, CIF_DIR
from .data import collate_graphs
from .predict import (
    DEFAULT_ATOM_INIT,
    DEFAULT_CIF_DIR,
    load_paper_model,
    predict_dataframe,
    prepared_dataset,
)
from .utils import load_checkpoint, move_inputs


EXPECTED_SPLITS = {
    "non_sse_train.csv": 9459,
    "finetune_train_core.csv": 536,
    "val_SSE.csv": 115,
    "test_B.csv": 115,
}
EXPECTED_HASHES = {
    "data/splits/non_sse_train.csv": "eee391fa13b872359ee1780c7851c18f868d9b77632d1cf3373ca913e4878531",
    "data/splits/finetune_train_core.csv": "dfd51dbb09766a10d502f4fced88e4b345a4d3d465dbd4a34ec69d50c3d8ab80",
    "data/splits/val_SSE.csv": "dc83a54d4ed76d0184a7e77246cf66a3e4d65c348510fbc6be155e7430759b0b",
    "data/splits/test_B.csv": "20d4e5e57e6ac7ab85372d7cea786258c99b667a2a68d0a63754a76c40989645",
    "checkpoints/K/model_best.pth.tar": "3c32325facc783edc42f0fc3078cceeffe29c777bdacafe6c423a030c3bf21bb",
    "checkpoints/G/model_best.pth.tar": "48b16c2c4f244951acf041afa59c2125290c5383b31de42f26f0a33f9a053e3e",
    "data/atom_init.json": "cb90bd2257dbf8cfc4e6b2b6d4348d4616e8e9680f1339b1461137032e0f1894",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_assets() -> None:
    failures = []
    split_ids = {}
    for name, expected_count in EXPECTED_SPLITS.items():
        path = SPLIT_DIR / name
        if not path.is_file():
            failures.append("missing split {}".format(path))
            continue
        frame = pd.read_csv(path)
        ids = frame["material_id"].astype(str)
        if len(frame) != expected_count:
            failures.append(
                "{} rows: expected {}, got {}".format(name, expected_count, len(frame))
            )
        if ids.duplicated().any():
            failures.append("{} contains duplicate material_id values".format(name))
        split_ids[name] = set(ids)

    all_ids = set().union(*split_ids.values()) if split_ids else set()
    missing_cifs = [
        mid for mid in sorted(all_ids) if not (CIF_DIR / (mid + ".cif")).is_file()
    ]
    packaged_cifs = list(CIF_DIR.glob("*.cif")) if CIF_DIR.is_dir() else []
    if missing_cifs:
        failures.append("{} referenced CIF files are missing".format(len(missing_cifs)))
    if len(packaged_cifs) != len(all_ids):
        failures.append(
            "CIF count mismatch: expected {}, got {}".format(
                len(all_ids), len(packaged_cifs)
            )
        )

    for relative, expected in EXPECTED_HASHES.items():
        path = PACKAGE_ROOT / relative
        if not path.is_file():
            failures.append("missing asset {}".format(relative))
        elif sha256(path) != expected:
            failures.append("SHA256 mismatch for {}".format(relative))

    checkpoint_expectations = {
        "K": (
            PACKAGE_ROOT / "checkpoints" / "K" / "model_best.pth.tar",
            "dslapm_edge_plain",
            141738,
        ),
        "G": (
            PACKAGE_ROOT / "checkpoints" / "G" / "model_best.pth.tar",
            "dslapm_edge_cgcnn",
            142314,
        ),
    }
    for prop, (path, arch, param_count) in checkpoint_expectations.items():
        if not path.is_file():
            continue
        checkpoint = load_checkpoint(path)
        if checkpoint.get("prop") != prop:
            failures.append("{} checkpoint prop mismatch".format(prop))
        if checkpoint.get("arch") != arch:
            failures.append("{} checkpoint arch mismatch".format(prop))
        if checkpoint.get("param_count") != param_count:
            failures.append("{} checkpoint param_count mismatch".format(prop))
        if checkpoint.get("args", {}).get("n_conv") != 4:
            failures.append("{} checkpoint n_conv mismatch".format(prop))

    if split_ids:
        train_ids = split_ids.get("non_sse_train.csv", set()) | split_ids.get(
            "finetune_train_core.csv", set()
        )
        val_ids = split_ids.get("val_SSE.csv", set())
        test_ids = split_ids.get("test_B.csv", set())
        if train_ids & val_ids:
            failures.append("training and validation IDs overlap")
        if train_ids & test_ids or val_ids & test_ids:
            failures.append("test_B overlaps training or validation")

    if failures:
        for failure in failures:
            print("[fail] {}".format(failure))
        raise SystemExit(1)
    print("[ok] split rows: {}".format(EXPECTED_SPLITS))
    print("[ok] unique material IDs and CIF files: {}".format(len(all_ids)))
    print("[ok] split disjointness")
    print("[ok] atom features and checkpoint SHA256 hashes")
    print("[ok] checkpoint metadata for K/plain and G/Fourier-F5")


def smoke_test() -> None:
    torch.manual_seed(42)
    split = (
        pd.read_csv(PACKAGE_ROOT / "data" / "splits" / "val_SSE.csv").iloc[:2].copy()
    )

    predictions = predict_dataframe(
        split,
        cif_dir=DEFAULT_CIF_DIR,
        atom_init=DEFAULT_ATOM_INIT,
        device=torch.device("cpu"),
        batch_size=2,
    )
    numeric = predictions[
        ["pred_log10_K", "pred_K_GPa", "pred_log10_G", "pred_G_GPa"]
    ].to_numpy()
    if numeric.shape != (2, 4) or not np.isfinite(numeric).all():
        raise AssertionError(
            "Checkpoint inference did not produce two finite K/G predictions."
        )

    with prepared_dataset(
        split, cif_dir=DEFAULT_CIF_DIR, atom_init=DEFAULT_ATOM_INIT
    ) as dataset:
        loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=collate_graphs,
            num_workers=0,
        )
        inputs, _, _ = next(iter(loader))
        model, _, checkpoint = load_paper_model("K", dataset, torch.device("cpu"))
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        before = model.fc_out.weight.detach().clone()
        optimizer.zero_grad()
        output = model(*move_inputs(inputs, torch.device("cpu")))
        target = torch.zeros_like(output)
        loss = nn.MSELoss()(output, target)
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        if not gradients or not all(
            torch.isfinite(gradient).all() for gradient in gradients
        ):
            raise AssertionError(
                "Backward pass produced missing or non-finite gradients."
            )
        if not any(torch.count_nonzero(gradient).item() > 0 for gradient in gradients):
            raise AssertionError("Backward pass produced only zero gradients.")
        optimizer.step()
        if torch.equal(before, model.fc_out.weight.detach()):
            raise AssertionError("Optimizer step did not update the prediction head.")

    print("[ok] packaged imports")
    print(
        "[ok] K plain-angle checkpoint epoch={} inference".format(checkpoint["epoch"])
    )
    print("[ok] G Fourier-angle (F=5) checkpoint inference")
    print("[ok] finite predictions for 2 real CIF structures")
    print("[ok] MSE backward pass and Adam optimizer step")
    print(
        predictions[["material_id", "pred_K_GPa", "pred_G_GPa"]].to_string(index=False)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--assets-only", action="store_true", help="Skip forward/backward checks."
    )
    args = parser.parse_args()
    verify_assets()
    if not args.assets_only:
        smoke_test()


if __name__ == "__main__":
    main()
