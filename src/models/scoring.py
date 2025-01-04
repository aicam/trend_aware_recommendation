import torch
import torch.nn as nn
import torch.nn.functional as F
from models.trend_extraction import TrendExtractor
from models.preference_propagation import PreferencePropagation

class ScoringModule(nn.Module):
    def __init__(self, config):
        super(ScoringModule, self).__init__()

        # Initialize TrendExtractor with its configuration
        self.trend_extractor = TrendExtractor(config['trend_extractor'])

        # Initialize PreferencePropagation with its configuration
        self.preference_propagation = PreferencePropagation(
            input_dim=config['preference_propagation']['input_dim'],
            hidden_dim=config['preference_propagation']['hidden_dim'],
            num_relations=config['preference_propagation']['num_relations'],
            num_layers=config['preference_propagation']['num_layers']
        )

        # Define MLP for final scoring
        self.mlp = nn.Sequential(
            nn.Linear(config['scoring_module']['mlp_input_dim'], config['scoring_module']['mlp_hidden_dim']),
            nn.ReLU(),
            nn.Linear(config['scoring_module']['mlp_hidden_dim'], 1)
        )

    def forward(self, user_description, item_description, edge_index, edge_type, time_segments):
        # Extract trend embeddings for user's investment history
        trend_embedding = self.trend_extractor(time_segments)

        # Perform preference propagation through the graph
        combined_descriptions = torch.cat((user_description, item_description), dim=0)
        preference_embedding = self.preference_propagation(combined_descriptions, edge_index, edge_type)

        # Split the preference embedding into user and item parts
        user_preference = preference_embedding[:user_description.size(0)]
        item_preference = preference_embedding[user_description.size(0):]

        # Combine embeddings for user and item
        user_features = torch.cat((trend_embedding, user_description, user_preference), dim=1)
        item_features = torch.cat((item_description, item_preference), dim=1)

        # Concatenate user and item features for final score calculation
        final_features = torch.cat((user_features, item_features), dim=1)
        score = self.mlp(final_features).squeeze(-1)  # Final score for investment recommendation

        return score
