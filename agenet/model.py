"""AGE-Net architecture and checkpoint-compatible model construction.

ConvLayer derives from CGCNN (Copyright Tian Xie); see ../LICENSE_CGCNN
for its MIT license.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .angles import PlainAngleEncoder, angle_embedding


# Historical architecture identifiers are retained for saved checkpoints.
VALID_ARCHS = (
    "param_matched_cgcnn",
    "dslapm_edge_cgcnn",
    "dslapm_edge_plain",
)


class ConvLayer(nn.Module):
    """Gated atom messages with distance features and a residual update."""

    def __init__(self, atom_fea_len, nbr_fea_len):
        super(ConvLayer, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len

        self.fc_full = nn.Linear(
            2 * self.atom_fea_len + self.nbr_fea_len, 2 * self.atom_fea_len
        )
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2 * self.atom_fea_len)
        self.bn2 = nn.BatchNorm1d(self.atom_fea_len)
        self.softplus2 = nn.Softplus()

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        """Update atom states from neighbor features shaped (N, M, D)."""
        N, M = nbr_fea_idx.shape

        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]

        total_nbr_fea = torch.cat(
            [
                atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),
                atom_nbr_fea,
                nbr_fea,
            ],
            dim=2,
        )

        total_gated_fea = self.fc_full(total_nbr_fea)
        total_gated_fea = self.bn1(
            total_gated_fea.view(-1, self.atom_fea_len * 2)
        ).view(N, M, self.atom_fea_len * 2)
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)

        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)
        nbr_sumed = self.bn2(nbr_sumed)

        out = self.softplus2(atom_in_fea + nbr_sumed)
        return out


class AngleGatedEdgeConv(nn.Module):
    """Refine directed edge states using angle-gated line-graph messages."""

    def __init__(self, fc_features):
        super(AngleGatedEdgeConv, self).__init__()
        self.fc_features = fc_features
        self.bn = nn.BatchNorm1d(fc_features)
        self.bn_interaction = nn.BatchNorm1d(fc_features)
        self.nonlinear_full = nn.Sequential(
            nn.Linear(3 * fc_features, fc_features),
            nn.SiLU(),
            nn.Linear(fc_features, fc_features),
        )
        self.nonlinear = nn.Sequential(
            nn.Linear(3 * fc_features, fc_features),
            nn.SiLU(),
            nn.Linear(fc_features, fc_features),
        )
        self.last_gate = None
        self.last_angle = None

    def _bn_safe(self, bn, x):
        if self.training and x.shape[0] < 2:
            return x
        return bn(x)

    def forward(self, edge_fea, lg_src, lg_dst, angle_emb, lg_angle=None):
        if lg_src.numel() == 0:
            self.last_gate = edge_fea.new_zeros((0, self.fc_features))
            self.last_angle = edge_fea.new_zeros((0,))
            return F.relu(edge_fea)

        src_fea = edge_fea[lg_src]
        dst_fea = edge_fea[lg_dst]
        cat = torch.cat((dst_fea, src_fea, angle_emb), dim=1)
        gate = torch.sigmoid(
            self._bn_safe(self.bn_interaction, self.nonlinear_full(cat))
        )
        msg = gate * self.nonlinear(cat)
        agg = edge_fea.new_zeros(edge_fea.shape)
        agg.index_add_(0, lg_dst, msg)
        connected = torch.zeros(
            edge_fea.shape[0], dtype=torch.bool, device=edge_fea.device
        )
        connected[lg_dst] = True
        agg_bn = edge_fea.new_zeros(edge_fea.shape)
        agg_bn[connected] = self._bn_safe(self.bn, agg[connected])
        out = F.relu(edge_fea + agg_bn)
        self.last_gate = gate.detach()
        self.last_angle = None if lg_angle is None else lg_angle.detach()
        return out


class AGENet(nn.Module):
    """Angle-gated edge refinement followed by crystal graph convolutions."""

    def __init__(
        self,
        orig_atom_fea_len,
        nbr_fea_len,
        atom_fea_len=64,
        n_conv=4,
        h_fea_len=128,
        n_h=1,
        classification=False,
        arch="dslapm_edge_cgcnn",
        fc_edge=64,
        num_frequencies=5,
    ):
        super(AGENet, self).__init__()
        if arch not in VALID_ARCHS:
            raise ValueError("Unknown architecture: {}".format(arch))
        self.classification = classification
        self.arch = arch
        self.nbr_fea_len = nbr_fea_len
        self.fc_edge = fc_edge

        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.edge_proj = nn.Linear(nbr_fea_len, fc_edge)
        if arch == "dslapm_edge_plain":
            self.angle_encoder = PlainAngleEncoder(
                fc_edge, num_frequencies=num_frequencies
            )
        else:
            self.angle_encoder = angle_embedding(
                fc_edge, num_frequencies=num_frequencies
            )
        self.no_angle = nn.Parameter(torch.zeros(fc_edge), requires_grad=False)
        self.edge_conv = AngleGatedEdgeConv(fc_edge)
        self.edge_back = nn.Linear(fc_edge, nbr_fea_len)

        self.convs = nn.ModuleList(
            [
                ConvLayer(atom_fea_len=atom_fea_len, nbr_fea_len=nbr_fea_len)
                for _ in range(n_conv)
            ]
        )
        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()
        if n_h > 1:
            self.fcs = nn.ModuleList(
                [nn.Linear(h_fea_len, h_fea_len) for _ in range(n_h - 1)]
            )
            self.softpluses = nn.ModuleList([nn.Softplus() for _ in range(n_h - 1)])
        self.fc_out = nn.Linear(h_fea_len, 2 if classification else 1)
        if self.classification:
            self.logsoftmax = nn.LogSoftmax(dim=1)
            self.dropout = nn.Dropout()

    def _valid_edge_mask(self, nbr_fea):
        return torch.amax(torch.abs(nbr_fea), dim=2) > 1e-6

    def refine_edges(self, nbr_fea, lg_src, lg_dst, lg_angle):
        N, M, D = nbr_fea.shape
        edge_flat = nbr_fea.reshape(N * M, D)
        edge_hidden = self.edge_proj(edge_flat)

        if self.arch == "param_matched_cgcnn":
            angle_emb = self.no_angle.view(1, -1).expand(lg_angle.shape[0], -1)
        else:
            angle_emb = self.angle_encoder(lg_angle)
        refined = self.edge_conv(edge_hidden, lg_src, lg_dst, angle_emb, lg_angle)
        delta = self.edge_back(refined).reshape(N, M, D)
        valid_mask = self._valid_edge_mask(nbr_fea).unsqueeze(-1)
        return nbr_fea + torch.where(valid_mask, delta, torch.zeros_like(delta))

    def forward(
        self,
        atom_fea,
        nbr_fea,
        nbr_fea_idx,
        crystal_atom_idx,
        lg_src=None,
        lg_dst=None,
        lg_angle=None,
        lg_center=None,
        lg_center_z=None,
    ):
        atom_fea = self.embedding(atom_fea)
        if lg_src is not None and lg_dst is not None and lg_angle is not None:
            lg_src = lg_src.to(device=nbr_fea.device, dtype=torch.long)
            lg_dst = lg_dst.to(device=nbr_fea.device, dtype=torch.long)
            lg_angle = lg_angle.to(device=nbr_fea.device, dtype=nbr_fea.dtype)
            nbr_fea = self.refine_edges(nbr_fea, lg_src, lg_dst, lg_angle)
        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)
        crys_fea = self.pooling(atom_fea, crystal_atom_idx)
        crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea))
        crys_fea = self.conv_to_fc_softplus(crys_fea)
        if self.classification:
            crys_fea = self.dropout(crys_fea)
        if hasattr(self, "fcs") and hasattr(self, "softpluses"):
            for fc, softplus in zip(self.fcs, self.softpluses):
                crys_fea = softplus(fc(crys_fea))
        out = self.fc_out(crys_fea)
        if self.classification:
            out = self.logsoftmax(out)
        return out

    def pooling(self, atom_fea, crystal_atom_idx):
        assert (
            sum(len(idx_map) for idx_map in crystal_atom_idx) == atom_fea.data.shape[0]
        )
        pooled = [
            torch.mean(atom_fea[idx_map], dim=0, keepdim=True)
            for idx_map in crystal_atom_idx
        ]
        return torch.cat(pooled, dim=0)

    def get_angle_spectrum(self):
        if hasattr(self.angle_encoder, "frequency_weight_norms"):
            return self.angle_encoder.frequency_weight_norms()
        return []

    def get_last_gate(self):
        return self.edge_conv.last_gate, self.edge_conv.last_angle


def build_model(arch, fc_edge=64, num_frequencies=5):
    """Return a dataset-aware model factory with the saved training dimensions."""

    def _build_model(dataset, cuda, n_conv=4):
        structures, _, _ = dataset[0]
        model = AGENet(
            structures[0].shape[-1],
            structures[1].shape[-1],
            atom_fea_len=64,
            n_conv=n_conv,
            h_fea_len=128,
            n_h=1,
            classification=False,
            arch=arch,
            fc_edge=fc_edge,
            num_frequencies=num_frequencies,
        )
        return model.cuda() if cuda else model

    return _build_model
