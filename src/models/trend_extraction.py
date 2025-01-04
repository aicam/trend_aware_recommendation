import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GatedGraphConv, RGCNConv
from torch_geometric.data import HeteroData
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn.functional as F
import torch.nn as nn


class TrendExtractor(nn.Module):
    def __init__(self, config):
        super(TrendExtractor, self).__init__()
        input_dim = config['input_dim']
        attention_dim = config.get('attention_dim', input_dim)  # Define separate attention dimension if needed
        hidden_dim = config['hidden_dim']
        num_layers = config['num_layers']

        # Use a default structure if 'structure' is not specified in config
        layer_structure = config.get('structure', [hidden_dim])

        # Attention layer, using input_dim for compatibility with `items`
        self.attention_layer = nn.Linear(input_dim, attention_dim)
        self.gru = nn.GRU(attention_dim, hidden_dim, num_layers, batch_first=True)

        # Sequentially connect layers in layer_structure
        layers = []
        in_dim = hidden_dim
        for out_dim in layer_structure:
            layers.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim
        self.output_layers = nn.ModuleList(layers)

        # Final output layer
        self.final_output = nn.Linear(layer_structure[-1], config['output_dim'])

    def forward(self, time_segments):
        segment_representations = []

        for items in time_segments:
            attention_weights = F.softmax(self.attention_layer(items), dim=1)
            weighted_items = attention_weights * items  # Element-wise multiplication with `items`
            segment_representation = weighted_items.sum(dim=0)
            segment_representations.append(segment_representation)

        segments = torch.stack(segment_representations, dim=0).unsqueeze(0)
        _, h_n = self.gru(segments)

        x = h_n[-1]
        for layer in self.output_layers:
            x = F.relu(layer(x))
        trend_embedding = self.final_output(x)

        return trend_embedding
