from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import torch
from torch_geometric.data import HeteroData
import numpy as np
from tqdm import tqdm
import os
import pandas as pd
import pickle


class DataPreprocessor:
    def __init__(self, config):
        self.config = config
        self.raw_path = config['data']['raw_path']
        self.processed_path = config['data']['processed_path']
        self.doc2vec_params = config['doc2vec']

    def load_data(self, file_path):
        print("\n==================== Step 1: Data Loading ====================")
        print("Loading data from Crunchbase dataset...")

        investment_data = pd.read_csv(file_path, encoding='ISO-8859-1')
        print(f"Columns found in data: {investment_data.columns.tolist()}")

        required_columns = [
            'permalink', 'name', 'category_list', 'market', 'status',
            'country_code', 'state_code', 'region', 'city',
            'funding_total_usd', 'funding_rounds', 'founded_at',
            'first_funding_at', 'last_funding_at', 'round_A', 'round_B',
            'round_C', 'round_D', 'round_E', 'round_F', 'round_G', 'round_H'
        ]
        available_columns = [col for col in required_columns if col in investment_data.columns]

        investment_data = investment_data[available_columns]
        investment_data.fillna(0, inplace=True)  # Fill missing values for numerical columns with 0

        # Compute total funding
        funding_columns = ['round_A', 'round_B', 'round_C', 'round_D', 'round_E', 'round_F', 'round_G', 'round_H']
        if all(col in investment_data.columns for col in funding_columns):
            investment_data['total_funding'] = investment_data[funding_columns].sum(axis=1)

        print(f"Loaded {len(investment_data)} records with columns: {available_columns + ['total_funding']}")
        return investment_data

    def extract_descriptions(self, data):
        print("\n==================== Step 2: Description Extraction ====================")
        print("Preparing tagged documents for Doc2Vec training...")

        documents = [
            TaggedDocument(words=str(categories).split('|'), tags=[str(i)])
            for i, categories in tqdm(enumerate(data['category_list'].fillna('')), desc="Preparing documents")
        ]

        print("Building Doc2Vec vocabulary...")
        doc2vec_model = Doc2Vec(
            vector_size=self.doc2vec_params['vector_size'],
            window=self.doc2vec_params['window'],
            min_count=self.doc2vec_params['min_count'],
            workers=self.doc2vec_params['workers']
        )
        doc2vec_model.build_vocab(documents)

        print("Training Doc2Vec model on descriptions...")
        doc2vec_model.train(documents, total_examples=doc2vec_model.corpus_count, epochs=self.doc2vec_params['epochs'])

        print("Generating description vectors...")
        tqdm.pandas(desc="Vectorizing descriptions")
        data['description_vector'] = data['category_list'].progress_apply(
            lambda desc: doc2vec_model.infer_vector(str(desc).split('|'))
        )

        # Ensure each vector is converted to a NumPy array explicitly
        data['description_vector'] = data['description_vector'].apply(lambda x: np.array(x))

        print("Description extraction completed.")
        return data

    def create_graph(self, data):
        print("\n==================== Step 3: Graph Creation ====================")
        print("Creating graph structure...")

        hetero_data = HeteroData()

        # Add organization nodes
        print("Adding organization nodes...")
        organization_ids = data['permalink'].unique()
        org_map = {org: idx for idx, org in enumerate(organization_ids)}
        hetero_data['organization'].x = torch.tensor(
            np.random.randn(len(organization_ids), 40), dtype=torch.float32  # Example feature dimensions
        )

        # Add location nodes and edges
        location_columns = ['country_code', 'state_code', 'region', 'city']
        for location_col in location_columns:
            if location_col in data.columns:
                print(f"Adding location nodes and edges for {location_col}...")
                location_ids = data[location_col].dropna().unique()
                location_map = {loc: idx for idx, loc in enumerate(location_ids)}
                hetero_data[location_col].x = torch.tensor(
                    np.random.randn(len(location_ids), 40), dtype=torch.float32  # Example feature dimensions
                )

                organization_to_location = []
                for _, row in data.iterrows():
                    if pd.notna(row[location_col]):
                        org_idx = org_map.get(row['permalink'], None)
                        loc_idx = location_map.get(row[location_col], None)
                        if org_idx is not None and loc_idx is not None:
                            organization_to_location.append([org_idx, loc_idx])

                if organization_to_location:
                    edge_index = torch.tensor(organization_to_location, dtype=torch.long).t().contiguous()

                    # Validate edge indices
                    max_index = hetero_data['organization'].x.shape[0]
                    if edge_index.max() >= max_index:
                        raise ValueError(
                            f"Invalid edge index found: max index {edge_index.max()} exceeds {max_index - 1}")

                    hetero_data['organization', f'located_in_{location_col}', location_col].edge_index = edge_index

        print("Final node types in hetero_data:", hetero_data.node_types)
        print("Final edge types in hetero_data:", hetero_data.edge_types)
        print("Graph creation completed.")
        return hetero_data

    def save_processed_data(self, data, graph_data):
        print("\n==================== Step 4: Saving Processed Data ====================")
        print("Saving processed data...")
        os.makedirs(self.processed_path, exist_ok=True)
        data.to_csv(os.path.join(self.processed_path, 'processed_data.csv'), index=False)

        with open(os.path.join(self.processed_path, 'graph_data.pkl'), 'wb') as f:
            pickle.dump(graph_data, f)

        print("Data and graph saved successfully.")
