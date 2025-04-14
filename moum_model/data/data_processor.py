"""
Data processing utilities for multi-omics data integration.

This module provides functions for loading, preprocessing, and preparing multi-omics data
for use with GNN-based integration models.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class MultiOmicsDataset:
    """Dataset class for multi-omics data."""
    
    def __init__(self, data_dir, omics_types=None, response_file=None, test_size=0.2, random_state=42):
        """
        Initialize the multi-omics dataset.
        
        Args:
            data_dir (str): Directory containing omics data files
            omics_types (list, optional): List of omics types to include. 
                                         If None, all available omics data will be included.
            response_file (str, optional): Path to drug response or target variable file
            test_size (float, optional): Proportion of data to use for testing. Defaults to 0.2.
            random_state (int, optional): Random seed for reproducibility. Defaults to 42.
        """
        self.data_dir = data_dir
        self.omics_types = omics_types
        self.response_file = response_file
        self.test_size = test_size
        self.random_state = random_state
        
        self.omics_data = {}
        self.response_data = None
        self.sample_ids = None
        self.feature_dims = {}
        
        # Preprocessing components
        self.scalers = {}
        
        # Train-test split indices
        self.train_indices = None
        self.test_indices = None
        
        # Load data if paths are provided
        if data_dir:
            self._load_omics_data()
        
        if response_file:
            self._load_response_data()
            
        if self.omics_data and self.response_data is not None:
            self._align_samples()
            self._create_train_test_split()
    
    def _load_omics_data(self):
        """Load omics data from files in the data directory."""
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory {self.data_dir} not found.")
        
        # If specific omics types are not provided, look for all available data files
        if self.omics_types is None:
            self.omics_types = []
            for file in os.listdir(self.data_dir):
                if file.endswith('.csv') or file.endswith('.tsv'):
                    omics_type = os.path.splitext(file)[0]
                    self.omics_types.append(omics_type)
        
        # Load each omics data file
        for omics_type in self.omics_types:
            csv_path = os.path.join(self.data_dir, f"{omics_type}.csv")
            tsv_path = os.path.join(self.data_dir, f"{omics_type}.tsv")
            
            if os.path.exists(csv_path):
                self.omics_data[omics_type] = pd.read_csv(csv_path, index_col=0)
            elif os.path.exists(tsv_path):
                self.omics_data[omics_type] = pd.read_csv(tsv_path, sep='\t', index_col=0)
            else:
                print(f"Warning: No data file found for {omics_type}")
                continue
            
            # Record dimensionality of each omics type
            self.feature_dims[omics_type] = self.omics_data[omics_type].shape[1]
            
            # If this is the first omics data loaded, set the sample IDs
            if self.sample_ids is None:
                self.sample_ids = self.omics_data[omics_type].index.tolist()
    
    def _load_response_data(self):
        """Load drug response or target variable data."""
        if not os.path.exists(self.response_file):
            raise FileNotFoundError(f"Response file {self.response_file} not found.")
        
        # Determine file format from extension
        if self.response_file.endswith('.csv'):
            self.response_data = pd.read_csv(self.response_file, index_col=0)
        elif self.response_file.endswith('.tsv'):
            self.response_data = pd.read_csv(self.response_file, sep='\t', index_col=0)
        else:
            raise ValueError("Response file must be CSV or TSV format.")
        
        # If no omics data loaded yet, set sample IDs from response data
        if self.sample_ids is None:
            self.sample_ids = self.response_data.index.tolist()
    
    def _align_samples(self):
        """Align samples across omics data and response data."""
        if not self.omics_data:
            raise ValueError("No omics data loaded.")
        
        # Find samples that are present in all datasets
        common_samples = set(self.sample_ids)
        
        for omics_type, df in self.omics_data.items():
            common_samples = common_samples.intersection(set(df.index))
        
        if self.response_data is not None:
            common_samples = common_samples.intersection(set(self.response_data.index))
        
        # Filter all datasets to include only common samples
        self.sample_ids = sorted(list(common_samples))
        
        if not self.sample_ids:
            raise ValueError("No common samples found across datasets.")
        
        for omics_type in self.omics_data:
            self.omics_data[omics_type] = self.omics_data[omics_type].loc[self.sample_ids]
        
        if self.response_data is not None:
            self.response_data = self.response_data.loc[self.sample_ids]
    
    def _create_train_test_split(self):
        """Create train-test split indices."""
        self.train_indices, self.test_indices = train_test_split(
            range(len(self.sample_ids)),
            test_size=self.test_size,
            random_state=self.random_state
        )
    
    def preprocess_data(self):
        """Preprocess omics data (standardization, etc.)."""
        for omics_type in self.omics_data:
            # Create and fit a scaler on training data
            scaler = StandardScaler()
            train_data = self.omics_data[omics_type].iloc[self.train_indices].values
            scaler.fit(train_data)
            self.scalers[omics_type] = scaler
            
            # Transform all data
            self.omics_data[omics_type] = pd.DataFrame(
                scaler.transform(self.omics_data[omics_type].values),
                index=self.omics_data[omics_type].index,
                columns=self.omics_data[omics_type].columns
            )
    
    def create_adjacency_matrices(self, thresholds=None):
        """
        Create adjacency matrices for relationships between omics features.
        
        Args:
            thresholds (dict, optional): Dictionary mapping omics type pairs to correlation thresholds.
                Format: {(type1, type2): threshold}. Defaults to None.
        
        Returns:
            dict: Dictionary mapping relation names to adjacency matrices (edge indices)
        """
        if thresholds is None:
            thresholds = {}
        
        edge_indices_dict = {}
        
        # Create correlation-based edges between different omics types
        for i, src_type in enumerate(self.omics_types):
            for j, dst_type in enumerate(self.omics_types):
                if src_type == dst_type:
                    continue  # Skip self-relationships for simplicity
                
                relation_name = f"{src_type}_to_{dst_type}"
                
                # Calculate correlations between features
                src_data = self.omics_data[src_type].iloc[self.train_indices].values
                dst_data = self.omics_data[dst_type].iloc[self.train_indices].values
                
                # Compute correlation matrix (simplified approach)
                # For more sophisticated methods, consider mutual information, etc.
                corr_matrix = np.abs(np.corrcoef(src_data.T, dst_data.T)[
                    :src_data.shape[1],
                    src_data.shape[1]:
                ])
                
                # Apply threshold to create edges
                threshold = thresholds.get((src_type, dst_type), 0.5)  # Default threshold of 0.5
                edges = np.where(corr_matrix > threshold)
                
                # Create edge indices
                if len(edges[0]) > 0:
                    edge_index = torch.tensor([edges[0], edges[1]], dtype=torch.long)
                    edge_indices_dict[relation_name] = edge_index
        
        return edge_indices_dict
    
    def get_train_data(self):
        """Get training data."""
        return self._get_data_subset(self.train_indices)
    
    def get_test_data(self):
        """Get testing data."""
        return self._get_data_subset(self.test_indices)
    
    def _get_data_subset(self, indices):
        """Get a subset of data based on indices."""
        omics_subset = {}
        for omics_type in self.omics_data:
            omics_subset[omics_type] = torch.tensor(
                self.omics_data[omics_type].iloc[indices].values,
                dtype=torch.float32
            )
        
        if self.response_data is not None:
            response_subset = torch.tensor(
                self.response_data.iloc[indices].values,
                dtype=torch.float32
            )
        else:
            response_subset = None
        
        return omics_subset, response_subset


def load_sample_data(data_dir, random_seed=42, create_synthetic=False):
    """
    Load sample multi-omics data for testing or demonstration.
    If real data is not available, synthetic data can be generated.
    
    Args:
        data_dir (str): Directory to load data from or save synthetic data to
        random_seed (int, optional): Random seed for reproducibility. Defaults to 42.
        create_synthetic (bool, optional): Whether to create synthetic data if real data
            is not available. Defaults to False.
    
    Returns:
        MultiOmicsDataset: Dataset containing the loaded or generated data
    """
    # Check if data directory exists
    if not os.path.exists(data_dir) and create_synthetic:
        os.makedirs(data_dir)
    
    # Define omics types to look for
    omics_types = ['gene_expression', 'methylation', 'copy_number']
    all_files_exist = True
    
    # Check if all necessary files exist
    for omics_type in omics_types:
        file_path = os.path.join(data_dir, f"{omics_type}.csv")
        if not os.path.exists(file_path):
            all_files_exist = False
            break
    
    response_file = os.path.join(data_dir, "drug_response.csv")
    if not os.path.exists(response_file):
        all_files_exist = False
    
    # If data doesn't exist and create_synthetic is True, generate synthetic data
    if not all_files_exist and create_synthetic:
        print("Generating synthetic multi-omics data...")
        np.random.seed(random_seed)
        
        # Sample parameters
        n_samples = 100
        n_genes = 1000
        n_cpg_sites = 800
        n_copy_number_regions = 500
        n_drugs = 5
        
        # Generate sample IDs
        sample_ids = [f"SAMPLE_{i}" for i in range(n_samples)]
        
        # Generate gene expression data
        gene_expr = np.random.normal(0, 1, (n_samples, n_genes))
        gene_ids = [f"GENE_{i}" for i in range(n_genes)]
        gene_expr_df = pd.DataFrame(gene_expr, index=sample_ids, columns=gene_ids)
        gene_expr_df.to_csv(os.path.join(data_dir, "gene_expression.csv"))
        
        # Generate methylation data
        methylation = np.random.beta(2, 5, (n_samples, n_cpg_sites))
        cpg_ids = [f"CPG_{i}" for i in range(n_cpg_sites)]
        methylation_df = pd.DataFrame(methylation, index=sample_ids, columns=cpg_ids)
        methylation_df.to_csv(os.path.join(data_dir, "methylation.csv"))
        
        # Generate copy number data
        copy_number = np.random.choice([-2, -1, 0, 1, 2], (n_samples, n_copy_number_regions))
        cn_region_ids = [f"CN_{i}" for i in range(n_copy_number_regions)]
        copy_number_df = pd.DataFrame(copy_number, index=sample_ids, columns=cn_region_ids)
        copy_number_df.to_csv(os.path.join(data_dir, "copy_number.csv"))
        
        # Generate drug response data
        # For simplicity, just generate random values representing IC50 or AUC
        drug_response = np.random.lognormal(0, 1, (n_samples, n_drugs))
        drug_ids = [f"DRUG_{i}" for i in range(n_drugs)]
        drug_response_df = pd.DataFrame(drug_response, index=sample_ids, columns=drug_ids)
        drug_response_df.to_csv(response_file)
        
        print("Synthetic data generation complete.")
    
    # Load the data using the MultiOmicsDataset class
    dataset = MultiOmicsDataset(
        data_dir=data_dir,
        omics_types=omics_types,
        response_file=response_file
    )
    
    return dataset
