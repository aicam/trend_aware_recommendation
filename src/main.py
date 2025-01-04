import sys
import os
import torch
import torch.optim as optim
import torch.nn as nn
from preprocessing import DataPreprocessor
from models.trend_extraction import TrendExtractor
from models.preference_propagation import PreferencePropagation
from models.scoring import ScoringModule
import numpy as np


# Configuration for the modules
config = {
    "data": {
        "raw_path": "data/crunchbase",
        "processed_path": "data/processed"
    },
    "doc2vec": {
        "vector_size": 128,
        "window": 5,
        "min_count": 2,
        "workers": 4,
        "epochs": 20
    },
    "trend_extractor": {
        "input_dim": 128,
        "hidden_dim": 80,
        "num_layers": 2,
        "output_dim": 20
    },
    "preference_propagation": {
        "input_dim": 128,
        "hidden_dim": 80,
        "num_relations": 4,
        "num_layers": 2
    },
    "scoring_module": {
        "trend_dim": 20,
        "preference_dim": 80,
        "mlp_input_dim": 120,
        "mlp_hidden_dim": 64,
        "output_dim": 1
    },
    "training": {
        "epochs": 10,
        "batch_size": 16,
        "learning_rate": 0.001
    }
}


# Initialize modules
trend_extractor = TrendExtractor(config["trend_extractor"])
preference_propagation = PreferencePropagation(
    input_dim=config["preference_propagation"]["input_dim"],
    hidden_dim=config["preference_propagation"]["hidden_dim"],
    num_relations=config["preference_propagation"]["num_relations"],
    num_layers=config["preference_propagation"]["num_layers"]
)
scoring_module = ScoringModule(config)


# Utility: Validate Edge Index
def validate_edge_index(edge_index, max_nodes):
    if edge_index.max() >= max_nodes:
        raise ValueError(
            f"Invalid edge_index: Max index {edge_index.max()} exceeds available nodes {max_nodes}."
        )


# Data Preprocessing
def load_and_preprocess_data(file_path):
    preprocessor = DataPreprocessor(config)
    data = preprocessor.load_data(file_path)
    data = preprocessor.extract_descriptions(data)

    # Convert description vectors to tensor
    print("Converting description vectors to tensor...")
    description_tensor = torch.tensor(
        np.stack(data['description_vector'].values), dtype=torch.float32
    )

    # Generate graph data and validate edges
    graph_data = preprocessor.create_graph(data)
    graph_data['organization'].x = description_tensor

    edge_index = graph_data['organization', 'located_in_country_code', 'country_code'].edge_index
    validate_edge_index(edge_index, graph_data['organization'].x.shape[0])

    print(f"Edge Index Shape: {edge_index.shape}")
    print(f"Max Edge Index Value: {edge_index.max()}")
    print(f"Node Feature Shape: {graph_data['organization'].x.shape}")

    return data, graph_data

def train_pipeline(data, graph_data, epochs, batch_size):
    optimizer = optim.Adam(
        list(trend_extractor.parameters()) +
        list(preference_propagation.parameters()) +
        list(scoring_module.parameters()),
        lr=config["training"]["learning_rate"]
    )
    criterion = nn.BCELoss()

    labels = torch.randint(0, 2, (graph_data['organization'].x.shape[0],)).float()

    edge_index = graph_data['organization', 'located_in_country_code', 'country_code'].edge_index
    edge_type = torch.zeros(edge_index.shape[1], dtype=torch.long)

    for epoch in range(epochs):
        total_loss = 0
        for i in range(0, graph_data['organization'].x.shape[0], batch_size):
            # Extract batch inputs
            batch_data = graph_data['organization'].x[i:i + batch_size]
            batch_labels = labels[i:i + batch_size]
            batch_mask = (edge_index[0] >= i) & (edge_index[0] < i + batch_size)
            batch_edge_index = edge_index[:, batch_mask]
            batch_edge_type = edge_type[batch_mask]

            # Normalize edge indices to batch range
            batch_edge_index -= i

            optimizer.zero_grad()

            # Forward pass
            preference_embedding = preference_propagation(batch_data, batch_edge_index, batch_edge_type)
            time_segments = [torch.randn(10, config["trend_extractor"]["input_dim"]) for _ in range(batch_data.shape[0])]
            trend_embedding = trend_extractor(time_segments)
            score = scoring_module(preference_embedding, trend_embedding)

            # Loss computation
            loss = criterion(score, batch_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(labels):.4f}")


# Main Execution
if __name__ == "__main__":
    file_path = os.path.join(config["data"]["raw_path"], "investments_VC.csv")
    data, graph_data = load_and_preprocess_data(file_path)
    print("Graph structure created successfully.")
    train_pipeline(data, graph_data, config["training"]["epochs"], config["training"]["batch_size"])
