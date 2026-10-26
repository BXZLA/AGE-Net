"""Train a property-specific AGE-Net model on the fixed mixed training pool."""

import argparse
import csv
import shutil
import time
from pathlib import Path
from random import sample

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from . import config
from .data import CrystalDataset, collate_graphs
from .model import build_model
from .utils import Normalizer, load_checkpoint, move_inputs, resolve_device, set_seed


def prepare_training_folder(folder, frame, prop):
    """Materialize graph inputs while preserving the original row order."""
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "id_prop.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        for _, row in frame.iterrows():
            writer.writerow([row["material_id"], np.log10(float(row[prop + "_VRH"]))])
    for mid in frame["material_id"].astype(str):
        source = config.CIF_DIR / (mid + ".cif")
        destination = folder / source.name
        if not destination.exists():
            shutil.copy2(source, destination)
    shutil.copy2(config.ATOM_INIT, folder / "atom_init.json")


def build_normalizer(dataset):
    indices = sample(range(len(dataset)), min(500, len(dataset)))
    _, targets, _ = collate_graphs([dataset[i] for i in indices])
    return Normalizer(targets)


def run_epoch(loader, model, criterion, optimizer, normalizer, device, train):
    """Optimize MSE, reporting MAE in unstandardized log10 space."""
    model.train() if train else model.eval()
    total, count = 0.0, 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for inputs, targets, _ in loader:
            inputs = move_inputs(inputs, device)
            target_norm = normalizer.norm(targets).to(device)
            output = model(*inputs)
            if not torch.isfinite(output).all():
                raise FloatingPointError("Non-finite AGE-Net output.")
            loss = criterion(output, target_norm)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            pred = normalizer.denorm(output.detach().cpu())
            batch_mae = torch.mean(torch.abs(pred - targets)).item()
            total += batch_mae * targets.size(0)
            count += targets.size(0)
    return total / max(count, 1)


def evaluate(prop, split_name, frame, run_dir, device):
    checkpoint = load_checkpoint(run_dir / "model_best.pth.tar")
    folder = run_dir / "work" / ("eval_" + split_name)
    prepare_training_folder(folder, frame, prop)
    dataset = CrystalDataset(str(folder))
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_graphs,
        num_workers=0,
    )
    model = build_model(checkpoint["arch"], config.FC_EDGE, config.NUM_FREQUENCIES)(
        dataset, device.type == "cuda", n_conv=config.N_CONV
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    normalizer = Normalizer()
    normalizer.load_state_dict(checkpoint["normalizer"])
    ids, trues, predictions = [], [], []
    with torch.no_grad():
        for inputs, targets, material_ids in loader:
            pred = normalizer.denorm(model(*move_inputs(inputs, device)).cpu())
            ids.extend(str(mid) for mid in material_ids)
            trues.extend(targets.view(-1).tolist())
            predictions.extend(pred.view(-1).tolist())
    result = pd.DataFrame(
        {
            "material_id": ids,
            "true_log" + prop: trues,
            "pred_log" + prop: predictions,
        }
    )
    result.to_csv(run_dir / ("predictions_" + split_name + ".csv"), index=False)
    log_mae = float(np.mean(np.abs(np.asarray(trues) - np.asarray(predictions))))
    print(
        "[eval] {} {} log10-MAE={:.6f} n={}".format(
            prop, split_name, log_mae, len(result)
        )
    )


def train(args):
    prop = args.property
    device = resolve_device(args.device)
    set_seed(args.seed, device.type == "cuda")
    arch = config.ARCHITECTURES[prop]
    run_name = args.run_name or "train_{}_seed{}".format(prop, args.seed)
    if Path(run_name).name != run_name or run_name in (".", ".."):
        raise ValueError(
            "--run-name must be a directory name, without path separators."
        )
    run_dir = args.output_dir / run_name
    if run_dir.exists():
        raise FileExistsError(
            "Choose a new --run-name; this run already exists: {}".format(run_dir)
        )
    run_dir.mkdir(parents=True)

    train_df = (
        pd.concat(
            [
                pd.read_csv(args.split_dir / "non_sse_train.csv"),
                pd.read_csv(args.split_dir / "finetune_train_core.csv"),
            ],
            ignore_index=True,
        )
        .sample(frac=1, random_state=args.seed)
        .reset_index(drop=True)
    )
    val_df = pd.read_csv(args.split_dir / "val_SSE.csv")
    overlap = set(train_df["material_id"].astype(str)) & set(
        val_df["material_id"].astype(str)
    )
    if overlap:
        raise ValueError(
            "Training and validation overlap: {} materials".format(len(overlap))
        )
    train_folder = run_dir / "work" / "train"
    val_folder = run_dir / "work" / "validation"
    prepare_training_folder(train_folder, train_df, prop)
    prepare_training_folder(val_folder, val_df, prop)
    train_ds = CrystalDataset(str(train_folder))
    val_ds = CrystalDataset(str(val_folder))

    generator = torch.Generator().manual_seed(args.seed)
    # BatchNorm cannot train on a trailing batch containing one structure.
    drop_last = (len(train_ds) % config.BATCH_SIZE) == 1
    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_graphs,
        num_workers=0,
        generator=generator,
        drop_last=drop_last,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_graphs,
        num_workers=0,
    )
    model = build_model(arch, config.FC_EDGE, config.NUM_FREQUENCIES)(
        train_ds, device.type == "cuda", n_conv=config.N_CONV
    )
    normalizer = build_normalizer(train_ds)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    scheduler = MultiStepLR(optimizer, milestones=config.LR_MILESTONES, gamma=0.1)
    best_val, best_epoch, no_improve = float("inf"), 0, 0
    log_rows = []
    print(
        "[train] {}: {} training, {} validation, device={}".format(
            prop, len(train_ds), len(val_ds), device
        )
    )
    for epoch in range(1, args.max_epochs + 1):
        started = time.time()
        train_mae = run_epoch(
            train_loader, model, criterion, optimizer, normalizer, device, True
        )
        val_mae = run_epoch(
            val_loader, model, criterion, None, normalizer, device, False
        )
        scheduler.step()
        improved = val_mae < best_val
        if improved:
            best_val, best_epoch, no_improve = val_mae, epoch, 0
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "best_mae_error": best_val,
                    "normalizer": normalizer.state_dict(),
                    "prop": prop,
                    "seed": args.seed,
                    "protocol": "agenet",
                    "arch": arch,
                    "param_count": sum(
                        p.numel() for p in model.parameters() if p.requires_grad
                    ),
                    "args": {
                        "arch": arch,
                        "fc_edge": config.FC_EDGE,
                        "num_frequencies": config.NUM_FREQUENCIES,
                        "atom_fea_len": config.ATOM_FEA_LEN,
                        "h_fea_len": config.H_FEA_LEN,
                        "n_conv": config.N_CONV,
                        "n_h": config.N_H,
                        "max_epochs": args.max_epochs,
                        "early_stop": "val_SSE",
                        "train": "non_sse_train.csv + finetune_train_core.csv",
                    },
                },
                run_dir / "model_best.pth.tar",
            )
        else:
            no_improve += 1
        log_rows.append(
            {
                "epoch": epoch,
                "train_mae": train_mae,
                "valSSE_mae": val_mae,
                "best_valSSE_mae": best_val,
            }
        )
        if epoch == 1 or improved or epoch % 20 == 0:
            print(
                "  epoch {:03d} train={:.4f} val={:.4f} best={:.4f}@{} ({:.0f}s)".format(
                    epoch,
                    train_mae,
                    val_mae,
                    best_val,
                    best_epoch,
                    time.time() - started,
                )
            )
        if no_improve >= config.PATIENCE:
            print(
                "[early stop] no validation improvement for {} epochs".format(
                    config.PATIENCE
                )
            )
            break
    pd.DataFrame(log_rows).to_csv(run_dir / "train_log.csv", index=False)
    for name, filename in (
        ("traincore", "finetune_train_core.csv"),
        ("valSSE", "val_SSE.csv"),
    ):
        evaluate(prop, name, pd.read_csv(args.split_dir / filename), run_dir, device)
    if args.report_tests:
        evaluate(
            prop, "testB", pd.read_csv(args.split_dir / "test_B.csv"), run_dir, device
        )
    print("[saved] {}".format(run_dir))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--property", choices=["K", "G"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--split-dir", type=Path, default=config.SPLIT_DIR)
    parser.add_argument("--output-dir", type=Path, default=config.ROOT / "outputs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--report-tests",
        action="store_true",
        help="Write test predictions after model selection; reporting only.",
    )
    args = parser.parse_args()
    if args.max_epochs < 1:
        parser.error("--max-epochs must be positive")
    train(args)


if __name__ == "__main__":
    main()
