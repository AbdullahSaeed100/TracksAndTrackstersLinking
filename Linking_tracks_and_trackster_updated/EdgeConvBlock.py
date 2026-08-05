import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadEdgeAttention(nn.Module):
    """Multi-head node attention with edge-dependent score contributions."""

    def __init__(self, node_dim, edge_dim, num_heads=4, dropout=0.1):
        super().__init__()
        if node_dim % num_heads != 0:
            raise ValueError("node_dim must be divisible by num_heads")

        self.num_heads = num_heads
        self.head_dim = node_dim // num_heads
        self.q_proj = nn.Linear(node_dim, node_dim)
        self.k_proj = nn.Linear(node_dim, node_dim)
        self.v_proj = nn.Linear(node_dim, node_dim)
        self.edge_proj = nn.Linear(edge_dim, num_heads)
        self.out_proj = nn.Linear(node_dim, node_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(node_dim)

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index[0].long(), edge_index[1].long()

        q = self.q_proj(x).view(-1, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(-1, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(-1, self.num_heads, self.head_dim)
        edge_score = self.edge_proj(edge_attr)

        scores = (q[src] * k[dst]).sum(dim=-1) / math.sqrt(self.head_dim)
        scores = scores + edge_score

        # Normalize over each source node's candidate neighbours.
        source_index = src.unsqueeze(-1).expand_as(scores)
        max_score = torch.full_like(x.new_zeros(x.size(0), self.num_heads), -torch.inf)
        max_score.scatter_reduce_(0, source_index, scores, reduce="amax", include_self=True)
        weights = torch.exp(scores - max_score[src])
        denominator = torch.zeros_like(max_score)
        denominator.scatter_add_(0, source_index, weights)
        weights = self.dropout(weights / denominator[src].clamp_min(1e-12))

        messages = (v[dst] * weights.unsqueeze(-1)).reshape(-1, self.num_heads * self.head_dim)
        aggregated = torch.zeros_like(x)
        aggregated.scatter_add_(0, src.unsqueeze(-1).expand_as(messages), messages)

        return self.norm(x + self.out_proj(aggregated))


class EdgeConvBlock(nn.Module):
    """Attention, edge-message processing, aggregation, and node update."""

    def __init__(self, in_feat, out_feat, edge_dim, dropout=0.2, heads=4):
        super().__init__()
        self.attention = MultiHeadEdgeAttention(in_feat, edge_dim, heads, dropout)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * in_feat + edge_dim, 2 * out_feat),
            nn.LayerNorm(2 * out_feat),
            nn.GLU(dim=-1),
            nn.Dropout(dropout),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(in_feat + out_feat, 2 * out_feat),
            nn.LayerNorm(2 * out_feat),
            nn.GLU(dim=-1),
            nn.Dropout(dropout),
        )
        self.skip = nn.Linear(in_feat, out_feat) if in_feat != out_feat else nn.Identity()
        self.norm = nn.LayerNorm(out_feat)

    def forward(self, x, edge_index, edge_attr):
        x_attended = self.attention(x, edge_index, edge_attr)
        src, dst = edge_index[0].long(), edge_index[1].long()

        edge_input = torch.cat([x_attended[src], x_attended[dst], edge_attr], dim=-1)
        edge_messages = self.edge_mlp(edge_input)

        aggregated = torch.zeros(
            x_attended.size(0), edge_messages.size(1),
            dtype=edge_messages.dtype, device=edge_messages.device,
        )
        aggregated.scatter_add_(
            0, src.unsqueeze(-1).expand_as(edge_messages), edge_messages
        )

        node_update = self.node_mlp(torch.cat([x_attended, aggregated], dim=-1))
        return self.norm(self.skip(x_attended) + node_update)
