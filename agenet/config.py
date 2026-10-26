"""Paths and fixed settings for the released AGE-Net models."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT / "data" / "splits"
CIF_DIR = SPLIT_DIR / "cif_files"
ATOM_INIT = ROOT / "data" / "atom_init.json"
CHECKPOINTS = {
    prop: ROOT / "checkpoints" / prop / "model_best.pth.tar" for prop in ("K", "G")
}
ARCHITECTURES = {"K": "dslapm_edge_plain", "G": "dslapm_edge_cgcnn"}

ATOM_FEA_LEN = 64
H_FEA_LEN = 128
N_CONV = 4
N_H = 1
FC_EDGE = 64
NUM_FREQUENCIES = 5
BATCH_SIZE = 64
LR = 1e-3
LR_MILESTONES = [150, 220]
PATIENCE = 70
