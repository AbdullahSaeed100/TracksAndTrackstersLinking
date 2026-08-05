import torch
import torch.nn as nn

from EdgeConvBlock import EdgeConvBlock


class GNN_TrackLinkingNet(nn.Module):
    """Edge-logit classifier for the active 21-node/9-edge-feature schema."""

    def __init__(
        self,
        input_dim=21,
        hidden_dim=64,
        output_dim=1,
        num_layers=3,
        edge_feature_dim=9,
        heads=4,
        dropout=0.2,
    ):
        super().__init__()
        self.input_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
        )
        self.edge_input_net = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
        )
        self.convs = nn.ModuleList([
            EdgeConvBlock(hidden_dim, hidden_dim, hidden_dim, dropout, heads)
            for _ in range(num_layers)
        ])
        self.edge_classifier = nn.Sequential(
            nn.Linear(3 * hidden_dim + edge_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x, edge_index, edge_attr):
        if not torch.isfinite(x).all():
            bad_count = (~torch.isfinite(x)).sum().item()
            raise ValueError(
                f"x contains {bad_count} non-finite values before model execution"
            )

        if not torch.isfinite(edge_attr).all():
            bad_count = (~torch.isfinite(edge_attr)).sum().item()
            raise ValueError(
                f"edge_attr contains {bad_count} non-finite values before model execution"
            )
        edge_index = edge_index.long()

        x = self.input_net(x)
        edge_features = self.edge_input_net(edge_attr)

        for conv in self.convs:
            x = conv(x, edge_index, edge_features)

        src, dst = edge_index[0], edge_index[1]
        edge_embedding = torch.cat(
            [x[src], x[dst], edge_features, edge_attr],
            dim=-1,
        )
        return self.edge_classifier(edge_embedding).squeeze(-1)
