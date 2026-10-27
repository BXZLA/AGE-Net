import csv
import functools
import json
import math
import os
import random
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset


try:
    import pymatgen.optimization.neighbors as _pon
    from pymatgen.optimization.neighbors import find_points_in_spheres as _orig_fps

    def _fps_safe(all_coords, center_coords, r, pbc, lattice, tol=1e-8):
        return _orig_fps(
            np.ascontiguousarray(all_coords, dtype=np.float64),
            np.ascontiguousarray(center_coords, dtype=np.float64),
            r,
            np.ascontiguousarray(pbc, dtype=np.int64),
            np.ascontiguousarray(lattice, dtype=np.float64),
            tol,
        )

    _pon.find_points_in_spheres = _fps_safe
except Exception:
    pass

from pymatgen.core.structure import Structure


EPS = 1e-10


class GaussianDistance(object):

    def __init__(self, dmin, dmax, step, var=None):
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax + step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):

        return np.exp(-((distances[..., np.newaxis] - self.filter) ** 2) / self.var**2)


class AtomInitializer(object):

    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {
            idx: atom_type for atom_type, idx in self._embedding.items()
        }

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, "_decodedict"):
            self._decodedict = {
                idx: atom_type for atom_type, idx in self._embedding.items()
            }
        return self._decodedict[idx]


class AtomCustomJSONInitializer(AtomInitializer):

    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)


class _CrystalDataset(Dataset):

    def __init__(
        self, root_dir, max_num_nbr=12, radius=8, dmin=0, step=0.2, random_seed=123
    ):
        self.root_dir = root_dir
        self.max_num_nbr, self.radius = max_num_nbr, radius
        assert os.path.exists(root_dir), "root_dir does not exist!"
        id_prop_file = os.path.join(self.root_dir, "id_prop.csv")
        assert os.path.exists(id_prop_file), "id_prop.csv does not exist!"
        with open(id_prop_file) as f:
            reader = csv.reader(f)
            self.id_prop_data = [row for row in reader]
        random.seed(random_seed)
        random.shuffle(self.id_prop_data)
        atom_init_file = os.path.join(self.root_dir, "atom_init.json")
        assert os.path.exists(atom_init_file), "atom_init.json does not exist!"
        self.ari = AtomCustomJSONInitializer(atom_init_file)
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)

    def __len__(self):
        return len(self.id_prop_data)

    @functools.lru_cache(maxsize=None)
    def __getitem__(self, idx):

        cif_id, target = self.id_prop_data[idx]

        crystal = Structure.from_file(os.path.join(self.root_dir, cif_id + ".cif"))

        atom_fea = np.vstack(
            [
                self.ari.get_atom_fea(crystal[i].specie.number)
                for i in range(len(crystal))
            ]
        )
        atom_fea = torch.Tensor(atom_fea)

        all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
        nbr_fea_idx, nbr_fea = [], []
        for nbr in all_nbrs:
            if len(nbr) < self.max_num_nbr:
                warnings.warn(
                    "{} not find enough neighbors to build graph. "
                    "If it happens frequently, consider increase "
                    "radius.".format(cif_id)
                )
                nbr_fea_idx.append(
                    list(map(lambda x: x[2], nbr)) + [0] * (self.max_num_nbr - len(nbr))
                )
                nbr_fea.append(
                    list(map(lambda x: x[1], nbr))
                    + [self.radius + 1.0] * (self.max_num_nbr - len(nbr))
                )
            else:
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr[: self.max_num_nbr])))
                nbr_fea.append(list(map(lambda x: x[1], nbr[: self.max_num_nbr])))
        nbr_fea_idx, nbr_fea = np.array(nbr_fea_idx), np.array(nbr_fea)

        nbr_fea = self.gdf.expand(nbr_fea)
        atom_fea = torch.Tensor(atom_fea)
        nbr_fea = torch.Tensor(nbr_fea)
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
        target = torch.Tensor([float(target)])

        return (atom_fea, nbr_fea, nbr_fea_idx), target, cif_id


def _angle(v1, v2):
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 <= EPS or n2 <= EPS:
        return None
    cosang = float(np.dot(v1, v2) / (n1 * n2))
    cosang = max(-1.0, min(1.0, cosang))
    return float(math.acos(cosang))


def sorted_neighbors(structure, radius, max_num_nbr):
    all_nbrs = structure.get_all_neighbors(radius, include_index=True)
    return [sorted(nbrs, key=lambda x: x[1])[:max_num_nbr] for nbrs in all_nbrs]


def build_linegraph_edge_pairs(
    structure, radius=8.0, max_num_nbr=12, max_triplets_per_atom=66
):

    lg_src, lg_dst, lg_angle, lg_center, lg_center_z = [], [], [], [], []
    all_nbrs = sorted_neighbors(structure, radius, max_num_nbr)
    for center_idx, nbrs in enumerate(all_nbrs):
        center = np.asarray(structure[center_idx].coords, dtype=np.float64)
        vectors = []
        slots = []
        for slot, nbr in enumerate(nbrs):
            dist = float(nbr[1])
            if dist <= EPS or dist > radius:
                continue
            vectors.append(np.asarray(nbr[0].coords, dtype=np.float64) - center)
            slots.append(slot)

        made = 0
        for a in range(len(vectors)):
            for b in range(a + 1, len(vectors)):
                if made >= max_triplets_per_atom:
                    break
                theta = _angle(vectors[a], vectors[b])
                if theta is None:
                    continue
                if not (0.0 <= theta <= math.pi):
                    raise ValueError(
                        "line-graph angle outside [0, pi]: {}".format(theta)
                    )
                edge_a = center_idx * max_num_nbr + slots[a]
                edge_b = center_idx * max_num_nbr + slots[b]
                for src, dst in ((edge_a, edge_b), (edge_b, edge_a)):
                    lg_src.append(src)
                    lg_dst.append(dst)
                    lg_angle.append(theta)
                    lg_center.append(center_idx)
                    lg_center_z.append(int(structure[center_idx].specie.number))
                made += 1
            if made >= max_triplets_per_atom:
                break

    if not lg_angle:
        return (
            torch.zeros(0, dtype=torch.long),
            torch.zeros(0, dtype=torch.long),
            torch.zeros(0, dtype=torch.float32),
            torch.zeros(0, dtype=torch.long),
            torch.zeros(0, dtype=torch.long),
        )
    angles = np.asarray(lg_angle, dtype=np.float32)
    if not np.isfinite(angles).all():
        raise ValueError("non-finite line-graph angles")
    return (
        torch.LongTensor(lg_src),
        torch.LongTensor(lg_dst),
        torch.Tensor(angles),
        torch.LongTensor(lg_center),
        torch.LongTensor(lg_center_z),
    )


def manual_angle_from_neighbors(
    structure, center_idx, nbr_slot_a, nbr_slot_b, radius=8.0, max_num_nbr=12
):
    nbrs = sorted_neighbors(structure, radius, max_num_nbr)[center_idx]
    center = np.asarray(structure[center_idx].coords, dtype=np.float64)
    v1 = np.asarray(nbrs[nbr_slot_a][0].coords, dtype=np.float64) - center
    v2 = np.asarray(nbrs[nbr_slot_b][0].coords, dtype=np.float64) - center
    theta = _angle(v1, v2)
    if theta is None:
        raise ValueError("degenerate manual angle")
    return theta


class CrystalDataset(_CrystalDataset):

    @functools.lru_cache(maxsize=None)
    def _linegraph(self, idx):
        cif_id, _ = self.id_prop_data[idx]
        structure = Structure.from_file(os.path.join(self.root_dir, cif_id + ".cif"))
        return build_linegraph_edge_pairs(
            structure,
            radius=self.radius,
            max_num_nbr=self.max_num_nbr,
            max_triplets_per_atom=66,
        )

    @functools.lru_cache(maxsize=None)
    def __getitem__(self, idx):
        (atom_fea, nbr_fea, nbr_fea_idx), target, cif_id = super(
            CrystalDataset, self
        ).__getitem__(idx)
        lg_src, lg_dst, lg_angle, lg_center, lg_center_z = self._linegraph(idx)
        if lg_angle.numel() > 0:
            if not torch.isfinite(lg_angle).all():
                raise ValueError("non-finite line-graph angle for {}".format(cif_id))
            if (
                torch.min(lg_angle).item() < -1e-6
                or torch.max(lg_angle).item() > math.pi + 1e-6
            ):
                raise ValueError(
                    "line-graph angle out of [0, pi] for {}".format(cif_id)
                )
        return (
            (
                atom_fea,
                nbr_fea,
                nbr_fea_idx,
                lg_src,
                lg_dst,
                lg_angle,
                lg_center,
                lg_center_z,
            ),
            target,
            cif_id,
        )


def collate_graphs(dataset_list):

    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx = [], [], []
    crystal_atom_idx, batch_target, batch_cif_ids = [], [], []
    lg_src_all, lg_dst_all, lg_angle_all, lg_center_all, lg_center_z_all = (
        [],
        [],
        [],
        [],
        [],
    )
    atom_base = 0
    edge_base = 0
    for (
        (
            atom_fea,
            nbr_fea,
            nbr_fea_idx,
            lg_src,
            lg_dst,
            lg_angle,
            lg_center,
            lg_center_z,
        ),
        target,
        cif_id,
    ) in dataset_list:
        n_i, max_num_nbr = nbr_fea_idx.shape
        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        batch_nbr_fea_idx.append(nbr_fea_idx + atom_base)
        crystal_atom_idx.append(torch.LongTensor(np.arange(n_i) + atom_base))
        if lg_angle.numel() > 0:
            lg_src_all.append(lg_src + edge_base)
            lg_dst_all.append(lg_dst + edge_base)
            lg_angle_all.append(lg_angle)
            lg_center_all.append(lg_center + atom_base)
            lg_center_z_all.append(lg_center_z)
        batch_target.append(target)
        batch_cif_ids.append(cif_id)
        atom_base += n_i
        edge_base += n_i * max_num_nbr

    if lg_angle_all:
        lg_src = torch.cat(lg_src_all, dim=0).long()
        lg_dst = torch.cat(lg_dst_all, dim=0).long()
        lg_angle = torch.cat(lg_angle_all, dim=0).float()
        lg_center = torch.cat(lg_center_all, dim=0).long()
        lg_center_z = torch.cat(lg_center_z_all, dim=0).long()
    else:
        lg_src = torch.zeros(0, dtype=torch.long)
        lg_dst = torch.zeros(0, dtype=torch.long)
        lg_angle = torch.zeros(0, dtype=torch.float32)
        lg_center = torch.zeros(0, dtype=torch.long)
        lg_center_z = torch.zeros(0, dtype=torch.long)

    return (
        (
            torch.cat(batch_atom_fea, dim=0),
            torch.cat(batch_nbr_fea, dim=0),
            torch.cat(batch_nbr_fea_idx, dim=0),
            crystal_atom_idx,
            lg_src,
            lg_dst,
            lg_angle,
            lg_center,
            lg_center_z,
        ),
        torch.stack(batch_target, dim=0),
        batch_cif_ids,
    )
