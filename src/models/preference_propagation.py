import torch
import torch.nn as nn
from torch_geometric.nn import RGCNConv

import torch
import torch.nn as nn
from torch_geometric.nn import RGCNConv


class PreferencePropagation(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_relations, num_layers=2):
        super(PreferencePropagation, self).__init__()
        self.layers = nn.ModuleList()

        # First RGCN layer with input_dim
        self.layers.append(RGCNConv(input_dim, hidden_dim, num_relations))

        # Hidden RGCN layers
        for _ in range(num_layers - 1):
            self.layers.append(RGCNConv(hidden_dim, hidden_dim, num_relations))

        # Output layer to produce a single logit per node for binary classification
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, edge_type):
        for layer in self.layers:
            x = layer(x, edge_index, edge_type)
            x = torch.relu(x)

        # Produce final output (single logit per node)
        return torch.sigmoid(self.output_layer(x)).squeeze(-1)  # Shape should be [num_nodes]
