import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from models.preference_propagation import PreferencePropagation


def generate_synthetic_data(num_nodes, input_dim):
    x = torch.randn(num_nodes, input_dim)
    edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
    edge_type = torch.randint(0, 4, (num_nodes * 2,))
    labels = torch.randint(0, 2, (num_nodes,))
    return x, edge_index, edge_type, labels


def train_model(model, data, labels, epochs, config_name):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()  # Binary cross-entropy for binary classification

    for epoch in tqdm(range(epochs), desc=f"Training {config_name}"):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        output = model(data[0], data[1], data[2])

        # Calculate loss
        loss = criterion(output, labels.float())
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

    return model


def test_preference_propagation():
    config = {'input_dim': 40, 'hidden_dim': 80, 'num_relations': 4, 'num_layers': 2, 'epochs': 10}
    x, edge_index, edge_type, labels = generate_synthetic_data(num_nodes=100, input_dim=config['input_dim'])
    preference_model = PreferencePropagation(config['input_dim'], config['hidden_dim'], config['num_relations'],
                                             config['num_layers'])

    print("\nTesting Preference Propagation with Synthetic Data")
    model = train_model(preference_model, (x, edge_index, edge_type), labels, config['epochs'], "PreferencePropagation")

    print("\nTesting completed. Model training and evaluation complete.")


if __name__ == "__main__":
    test_preference_propagation()
