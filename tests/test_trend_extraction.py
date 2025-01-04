
import torch
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import os
import sys
import numpy as np
import os

from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from models.trend_extraction import TrendExtractor  # Use the correct path to import
def load_config():
    with open('config/config.yaml', 'r') as f:
        return yaml.safe_load(f)['trend_extractor']


def test_trend_extractor():
    # Load configuration
    config = load_config()

    # Display model configuration
    print("\n===== Testing TrendExtractor Model =====")
    print(
        f"Model Configuration:\n - Input Dimension: {config['input_dim']}\n - Hidden Dimension: {config['hidden_dim']}\n - GRU Layers: {config['num_layers']}")

    # Generate synthetic data with specified dimensions
    num_segments = 5
    segment_length = 10
    input_dim = config['input_dim']
    time_segments = [torch.randn(segment_length, input_dim) for _ in range(num_segments)]

    print("\nGenerated Sample Time Segments:")
    print(f" - Number of Segments: {num_segments}")
    print(f" - Each Segment Shape: ({segment_length}, {input_dim})")
    print(f" - Example Segment Data:\n{time_segments[0][:3]} ... [truncated]")

    # Initialize TrendExtractor model with configuration
    trend_extractor = TrendExtractor(config)
    print("\nInitialized TrendExtractor Model.")

    # Run the model and get the trend embedding
    trend_embedding = trend_extractor(time_segments)

    # Output results
    print("\n===== Model Output =====")
    print("Trend Embedding Shape:", trend_embedding.shape)
    print("Trend Embedding (First 5 Values):", trend_embedding[0][:5].detach().numpy())

    # Update assertion to use config-based shape
    assert trend_embedding.shape == (
    1, config['output_dim']), f"Unexpected trend embedding shape: {trend_embedding.shape}"

    print("\nTest Passed! The TrendExtractor successfully produced an embedding with the expected shape.")
    print("========================================\n")


# Generate synthetic data
def generate_synthetic_data(num_samples=100, num_segments=5, segment_length=10, input_dim=40):
    data = []
    labels = torch.randint(0, 2, (num_samples,))  # Binary labels for classification
    for _ in range(num_samples):
        sample = [torch.randn(segment_length, input_dim) for _ in range(num_segments)]
        data.append(sample)
    return data, labels


# Define TrendModel with a classifier layer on top
class TrendModel(nn.Module):
    def __init__(self, config):
        super(TrendModel, self).__init__()
        self.trend_extractor = TrendExtractor(config)
        self.classifier = nn.Linear(config['output_dim'], 2)  # Binary classifier

    def forward(self, time_segments):
        embedding = self.trend_extractor(time_segments)
        return self.classifier(embedding)



def train_and_evaluate(model, data, labels, config_name, epochs, save_path="models/"):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()  # Binary cross-entropy for two classes (0, 1)
    model.train()

    # Training loop with tqdm progress bar
    for epoch in tqdm(range(epochs), desc=f"Training {config_name}", leave=True):
        total_loss = 0
        for i, (sample, label) in enumerate(zip(data, labels)):
            optimizer.zero_grad()
            output = model(sample)  # Shape should be [1, 2] for binary classification
            loss = criterion(output, label.unsqueeze(0))  # `label` should be of shape [1]
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Config {config_name} | Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(data):.4f}")

    # Save the model and results
    os.makedirs(save_path, exist_ok=True)
    model_save_path = os.path.join(save_path, f"trend_model_{config_name}_epochs_{epochs}.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
# Main testing function to load configurations, train, and evaluate
def load_config(config_name="trend_extractor"):
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config[config_name]


def test_trend_model():
    # List of model configurations to test
    config_names = ["trend_extractor", "trend_extractor_alternate", "trend_extractor_deep"]

    # Generate synthetic data
    data, labels = generate_synthetic_data(num_samples=100, num_segments=5, segment_length=10, input_dim=40)

    # Train and evaluate each configuration
    for i, config_name in enumerate(config_names, 1):
        config = load_config(config_name)
        model = TrendModel(config)

        print(f"\nTraining configuration {i}: Structure - {config['structure']} with {config['epochs']} epochs")
        train_and_evaluate(model, data, labels, config_name=f"{config_name}_{i}", epochs=config['epochs'],
                           save_path="models/configs")


# Run the full test
if __name__ == "__main__":
    test_trend_extractor()
    test_trend_model()
