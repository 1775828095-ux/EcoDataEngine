"""Trainable, permutation-invariant sparse point encoder for FNO inputs."""
import torch
from torch import nn


class SparseObservationEncoder(nn.Module):
    """Cross-attend grid queries to (x, y, value, quality) point tokens.

    Padded or missing stations are excluded by ``valid``. Future learned
    reliability can replace the explicit quality prior without changing FNO.
    """

    def __init__(self, channels=8):
        super().__init__()
        self.points = nn.Sequential(nn.Linear(4, channels), nn.GELU(), nn.Linear(channels, channels))
        self.queries = nn.Linear(2, channels)
        self.channels = channels

    def forward(self, points, valid, queries):
        tokens = self.points(points)
        query = self.queries(queries)
        logits = torch.einsum("qd,bnd->bqn", query, tokens) / self.channels ** 0.5
        logits = logits + points[..., 3].clamp_min(1e-8).log()[:, None, :]
        logits = logits.masked_fill(~valid[:, None, :], -1e9)
        weights = logits.softmax(-1) * valid[:, None, :]
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        return torch.einsum("bqn,bnd->bdq", weights, tokens)


class AttentionObservationEncoder(nn.Module):
    """PyTorch cross attention: grid features query location/value/QA tokens."""

    def __init__(self, channels, grid_channels, residual_gate=0.0, locality_scale=0.25,
                 cross_attention=True):
        super().__init__()
        self.points = nn.Sequential(nn.Linear(4, channels), nn.GELU(), nn.Linear(channels, channels))
        self.queries = nn.Linear(2, channels)
        self.grid = nn.Conv2d(grid_channels, channels, 1)
        self.attention = nn.MultiheadAttention(channels, 1, batch_first=True)
        self.residual_gate = residual_gate
        self.locality_scale = locality_scale
        self.cross_attention = cross_attention

    def forward(self, points, valid, queries, grid):
        tokens = self.points(points).masked_fill(~valid[..., None], 0)
        query = self.queries(queries)[None] + self.grid(grid).flatten(2).transpose(1, 2)
        empty = ~valid.any(dim=1)
        prior = points[..., 3].clamp_min(1e-8).log().masked_fill(~valid, float('-inf'))
        # Prevent all-masked attention NaNs; no-observation rows return zero.
        prior[empty, 0] = 0
        if self.cross_attention:
            result, _ = self.attention(query, tokens, tokens, key_padding_mask=prior, need_weights=False)
        else:
            # Ablate only the background-conditioned path. The independent
            # local observation residual remains controlled by residual_gate.
            result = torch.zeros_like(query)
        if self.residual_gate:
            # An observation-only residual bypasses background-conditioned
            # queries. Reuse the same official MHA; no extra network layers.
            distance = torch.cdist(queries[None].expand(points.shape[0], -1, -1), points[..., :2])
            local_prior = -distance.square() / (2 * self.locality_scale ** 2) + prior[:, None]
            obs_query = self.queries(queries)[None].expand(points.shape[0], -1, -1)
            local, _ = self.attention(obs_query, tokens, tokens, attn_mask=local_prior, need_weights=False)
            coverage = torch.exp(-distance.square() / (2 * self.locality_scale ** 2)).masked_fill(~valid[:, None], 0).amax(-1)
            result = result + self.residual_gate * coverage[..., None] * local
        return result.masked_fill(empty[:, None, None], 0).transpose(1, 2)
