import os
import sys
import yaml
import pandas as pd
import numpy as np  # Import numpy to check array instances
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from preprocessing import DataPreprocessor

# Add the src directory to the system path

import matplotlib.pyplot as plt
import networkx as nx
import random
from torch_geometric.utils import to_networkx


def plot_sample_graph(graph_data, num_nodes=10):
    """
    Plots a sample of nodes and edges from the heterogeneous graph.

    Args:
        graph_data (HeteroData): The heterogeneous graph data.
        num_nodes (int): The number of nodes to sample for visualization.
    """
    # Initialize a new NetworkX graph
    G = nx.Graph()

    # Select a sample of organization nodes
    org_nodes = range(min(num_nodes, graph_data['organization'].x.size(0)))
    for idx in org_nodes:
        G.add_node(f"organization_{idx}", type="organization")

    # Select a sample of location nodes and add edges
    for location_type in ['country_code', 'state_code', 'region', 'city']:
        if location_type in graph_data.node_types:
            location_nodes = range(min(num_nodes, graph_data[location_type].x.size(0)))
            for idx in location_nodes:
                G.add_node(f"{location_type}_{idx}", type=location_type)

            # Add edges between organizations and location nodes for this location type
            edge_index = graph_data['organization', f'located_in_{location_type}', location_type].edge_index
            for i in range(edge_index.size(1)):
                src, dst = edge_index[:, i]
                if src < num_nodes and dst < num_nodes:
                    G.add_edge(f"organization_{src}", f"{location_type}_{dst}", relation=f"located_in_{location_type}")

    # Plotting the graph
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)

    # Define color map for node types
    color_map = {
        'organization': 'blue',
        'country_code': 'red',
        'state_code': 'purple',
        'region': 'orange',
        'city': 'brown'
    }

    # Draw nodes by type
    for node_type, color in color_map.items():
        nodes = [node for node, attr in G.nodes(data=True) if attr['type'] == node_type]
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=color, label=node_type, node_size=100)

    # Draw edges and labels
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos, font_size=8)

    plt.title("Sample Nodes and Edges in Heterogeneous Graph")
    plt.legend()
    plt.show()


# Testing function to visualize the graph
def test_graph_creation_with_plot(preprocessor, data):
    graph_data = preprocessor.create_graph(data)
    print("Plotting sample graph structure...")
    plot_sample_graph(graph_data)

def load_config():
    """Loads the configuration file from config/config.yaml."""
    config_path = os.path.join("config", "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def test_data_loading(preprocessor, file_path):
    """Tests data loading from Crunchbase."""
    print("Testing data loading...")
    data = preprocessor.load_data(file_path)
    assert isinstance(data, pd.DataFrame), "Data loading failed, not a DataFrame."
    assert not data.empty, "Loaded data is empty."
    print("Data loading test passed.")
    print("\nSample of loaded data:")
    print(data.head())
    print("\nData info:")
    print(data.info())


def test_description_extraction(preprocessor, data):
    """Tests description extraction using Doc2Vec."""
    print("Testing description extraction...")
    processed_data = preprocessor.extract_descriptions(data)
    assert "description_vector" in processed_data.columns, "Description vector column missing."
    assert all(isinstance(vec, np.ndarray) for vec in processed_data["description_vector"]), \
        "Description vectors are not NumPy arrays."
    print("Description extraction test passed.")
    print("\nSample of processed data with description vectors:")
    print(processed_data[['permalink', 'name', 'description_vector']].head())


def test_graph_creation(preprocessor, data):
    print("Testing graph creation...")
    graph_data = preprocessor.create_graph(data)

    # Verify that 'organization' is one of the node types in graph_data
    assert 'organization' in graph_data.node_types, "Graph node type 'organization' missing."
    print("Organization nodes successfully created.")

    # Check for market nodes and edges if available
    if 'market' in data.columns and 'market' in graph_data.node_types:
        print("Market nodes successfully created.")
        assert 'market' in graph_data.node_types, "Graph node type 'market' missing."

        if ('organization', 'belongs_to_market', 'market') in graph_data.edge_types:
            assert hasattr(graph_data['organization', 'belongs_to_market', 'market'], "edge_index"), \
                "Graph creation failed, edge index missing between organizations and markets."
            print("Edges successfully created between organizations and markets.")
        else:
            print("No edges were created between organizations and markets.")

    # Check location node types and edges
    for location_col in ['country_code', 'state_code', 'region', 'city']:
        if location_col in data.columns and location_col in graph_data.node_types:
            print(f"{location_col} location nodes successfully created.")
            if ('organization', f'located_in_{location_col}', location_col) in graph_data.edge_types:
                print(f"Edges successfully created between organizations and {location_col} locations.")
            else:
                print(f"No edges were created between organizations and {location_col} locations.")

    print("\nSample nodes in the graph:")
    print("Organization nodes:", graph_data['organization'].x[:5])

    if 'market' in graph_data.node_types:
        print("Market nodes:", graph_data['market'].x[:5])


def run_tests():
    """Runs all preprocessing tests."""
    config = load_config()
    preprocessor = DataPreprocessor(config)
    file_path = os.path.join(config['data']['raw_path'], 'investments_VC.csv')  # Use path from config

    # Step 1: Test data loading
    data = preprocessor.load_data(file_path)
    test_data_loading(preprocessor, file_path)

    # Step 2: Test description extraction
    test_description_extraction(preprocessor, data)

    # Step 3: Test graph creation
    test_graph_creation(preprocessor, data)

    test_graph_creation_with_plot(preprocessor,data)


if __name__ == "__main__":
    run_tests()
