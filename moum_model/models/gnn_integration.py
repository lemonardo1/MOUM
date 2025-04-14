"""
Multi-omics data integration using Graph Neural Networks.

This module implements a GNN-based approach for integrating multiple types of omics data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv
from torch_geometric.data import Data, Batch


class OmicsEncoder(nn.Module):
    """Base encoder for individual omics data types."""
    
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.2):
        """
        Initialize the encoder for a specific omics data type.
        
        Args:
            input_dim (int): Input dimension of the omics data
            hidden_dim (int): Hidden layer dimension
            output_dim (int): Output embedding dimension
            dropout (float, optional): Dropout rate. Defaults to 0.2.
        """
        super(OmicsEncoder, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim)
        )
        
    def forward(self, x):
        """Forward pass through the encoder."""
        return self.layers(x)


class MultiOmicsGNN(nn.Module):
    """GNN-based model for multi-omics data integration."""
    
    def __init__(self, omics_dims, hidden_dim=256, embedding_dim=128, gnn_type='GCN', 
                 num_gnn_layers=2, dropout=0.2, task_type='regression', num_tasks=1):
        """
        Initialize the GNN-based multi-omics integration model.
        
        Args:
            omics_dims (dict): Dictionary mapping omics types to their input dimensions
            hidden_dim (int, optional): Hidden layer dimension. Defaults to 256.
            embedding_dim (int, optional): Dimension of the integrated embedding. Defaults to 128.
            gnn_type (str, optional): Type of GNN layer ('GCN', 'GAT', 'SAGE', 'GIN'). Defaults to 'GCN'.
            num_gnn_layers (int, optional): Number of GNN layers. Defaults to 2.
            dropout (float, optional): Dropout rate. Defaults to 0.2.
            task_type (str, optional): Type of task ('regression' or 'classification'). Defaults to 'regression'.
            num_tasks (int, optional): Number of tasks (e.g., number of drugs for drug response). Defaults to 1.
        """
        super(MultiOmicsGNN, self).__init__()
        
        self.omics_types = list(omics_dims.keys())
        self.embedding_dim = embedding_dim
        self.task_type = task_type
        self.num_tasks = num_tasks
        
        # Create encoders for each omics type
        self.encoders = nn.ModuleDict({
            omics_type: OmicsEncoder(
                input_dim=dim,
                hidden_dim=hidden_dim,
                output_dim=embedding_dim,
                dropout=dropout
            )
            for omics_type, dim in omics_dims.items()
        })
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        
        for _ in range(num_gnn_layers):
            if gnn_type == 'GCN':
                self.gnn_layers.append(GCNConv(embedding_dim, embedding_dim))
            elif gnn_type == 'GAT':
                self.gnn_layers.append(GATConv(embedding_dim, embedding_dim))
            elif gnn_type == 'SAGE':
                self.gnn_layers.append(SAGEConv(embedding_dim, embedding_dim))
            elif gnn_type == 'GIN':
                nn_layer = nn.Sequential(
                    nn.Linear(embedding_dim, embedding_dim),
                    nn.ReLU(),
                    nn.Linear(embedding_dim, embedding_dim)
                )
                self.gnn_layers.append(GINConv(nn_layer))
            else:
                raise ValueError(f"Unsupported GNN type: {gnn_type}")
        
        # Prediction layers
        self.prediction_layers = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_tasks)
        )
        
        if task_type == 'classification':
            self.final_activation = nn.Sigmoid()
        else:  # regression
            self.final_activation = nn.Identity()
    
    def _construct_graph(self, omics_embeddings, adjacency_matrix=None):
        """
        Construct a graph from omics embeddings.
        
        Args:
            omics_embeddings (dict): Dictionary mapping omics types to their embeddings
            adjacency_matrix (torch.Tensor, optional): Predefined adjacency matrix for the graph.
                If None, a fully connected graph is created. Defaults to None.
        
        Returns:
            torch_geometric.data.Data: The constructed graph
        """
        # Concatenate all node features (omics embeddings)
        nodes = []
        node_types = []
        
        for i, (omics_type, embedding) in enumerate(omics_embeddings.items()):
            nodes.append(embedding)
            node_types.extend([i] * embedding.shape[0])
        
        x = torch.cat(nodes, dim=0)
        node_types = torch.tensor(node_types, device=x.device)
        
        # Create edges (either from adjacency matrix or fully connected)
        if adjacency_matrix is not None:
            edge_index = adjacency_matrix.nonzero().t().contiguous()
        else:
            # Create a fully connected graph
            num_nodes = x.shape[0]
            source_nodes = []
            target_nodes = []
            
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j:  # Exclude self-loops
                        source_nodes.append(i)
                        target_nodes.append(j)
            
            edge_index = torch.tensor([source_nodes, target_nodes], device=x.device)
        
        return Data(x=x, edge_index=edge_index, node_type=node_types)
    
    def forward(self, omics_data, adjacency_matrix=None):
        """
        Forward pass through the multi-omics GNN model.
        
        Args:
            omics_data (dict): Dictionary mapping omics types to their data tensors
            adjacency_matrix (torch.Tensor, optional): Predefined adjacency matrix. Defaults to None.
        
        Returns:
            torch.Tensor: Model predictions
        """
        # Encode each omics type
        omics_embeddings = {}
        for omics_type in self.omics_types:
            if omics_type in omics_data:
                omics_embeddings[omics_type] = self.encoders[omics_type](omics_data[omics_type])
        
        # Construct graph
        graph = self._construct_graph(omics_embeddings, adjacency_matrix)
        
        # Apply GNN layers
        x, edge_index = graph.x, graph.edge_index
        for gnn_layer in self.gnn_layers:
            x = gnn_layer(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.2, training=self.training)
        
        # Readout: average the node embeddings
        graph_embedding = torch.mean(x, dim=0)
        
        # Prediction
        predictions = self.prediction_layers(graph_embedding)
        
        return self.final_activation(predictions)


class HeterogeneousOmicsGNN(nn.Module):
    """
    Heterogeneous Graph Neural Network for multi-omics integration.
    This model handles different types of nodes (omics) and relationships between them.
    """
    
    def __init__(self, omics_dims, hidden_dim=256, embedding_dim=128, 
                 num_gnn_layers=2, dropout=0.2, task_type='regression', num_tasks=1):
        """
        Initialize the heterogeneous GNN model for multi-omics integration.
        
        Args:
            omics_dims (dict): Dictionary mapping omics types to their input dimensions
            hidden_dim (int, optional): Hidden layer dimension. Defaults to 256.
            embedding_dim (int, optional): Dimension of the integrated embedding. Defaults to 128.
            num_gnn_layers (int, optional): Number of GNN layers. Defaults to 2.
            dropout (float, optional): Dropout rate. Defaults to 0.2.
            task_type (str, optional): Type of task ('regression' or 'classification'). Defaults to 'regression'.
            num_tasks (int, optional): Number of tasks. Defaults to 1.
        """
        super(HeterogeneousOmicsGNN, self).__init__()
        
        self.omics_types = list(omics_dims.keys())
        self.embedding_dim = embedding_dim
        self.task_type = task_type
        self.num_tasks = num_tasks
        
        # Create encoders for each omics type
        self.encoders = nn.ModuleDict({
            omics_type: OmicsEncoder(
                input_dim=dim,
                hidden_dim=hidden_dim,
                output_dim=embedding_dim,
                dropout=dropout
            )
            for omics_type, dim in omics_dims.items()
        })
        
        # Heterogeneous GNN layers (simplified version)
        # In a real implementation, you would use a framework like PyTorch Geometric's 
        # HeteroConv or DGL's RelGraphConv for proper heterogeneous graph convolutions
        self.hgnn_layers = nn.ModuleList()
        for _ in range(num_gnn_layers):
            # One GCNConv per omics type relation
            relation_convs = nn.ModuleDict()
            for src_type in self.omics_types:
                for dst_type in self.omics_types:
                    relation_name = f"{src_type}_to_{dst_type}"
                    relation_convs[relation_name] = GCNConv(embedding_dim, embedding_dim)
            
            self.hgnn_layers.append(relation_convs)
        
        # Prediction layers (similar to the homogeneous case)
        self.prediction_layers = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_tasks)
        )
        
        if task_type == 'classification':
            self.final_activation = nn.Sigmoid()
        else:  # regression
            self.final_activation = nn.Identity()
    
    def forward(self, omics_data, edge_indices_dict):
        """
        Forward pass through the heterogeneous multi-omics GNN model.
        
        Args:
            omics_data (dict): Dictionary mapping omics types to their data tensors
            edge_indices_dict (dict): Dictionary mapping relation names to edge indices
                Format: {f"{src_type}_to_{dst_type}": edge_index_tensor}
        
        Returns:
            torch.Tensor: Model predictions
        """
        # Encode each omics type
        node_features = {}
        for omics_type in self.omics_types:
            if omics_type in omics_data:
                node_features[omics_type] = self.encoders[omics_type](omics_data[omics_type])
        
        # Apply heterogeneous GNN layers
        for layer in self.hgnn_layers:
            new_features = {omics_type: torch.zeros_like(feat) for omics_type, feat in node_features.items()}
            
            # Aggregate messages from each relation
            for src_type in self.omics_types:
                for dst_type in self.omics_types:
                    relation_name = f"{src_type}_to_{dst_type}"
                    
                    if relation_name in edge_indices_dict:
                        edge_index = edge_indices_dict[relation_name]
                        conv = layer[relation_name]
                        
                        # Apply the appropriate convolution
                        src_features = node_features[src_type]
                        dst_features = conv(src_features, edge_index)
                        
                        # Add to the destination features
                        new_features[dst_type] += dst_features
            
            # Apply non-linearity and update features
            for omics_type in self.omics_types:
                node_features[omics_type] = F.relu(new_features[omics_type])
                node_features[omics_type] = F.dropout(node_features[omics_type], p=0.2, training=self.training)
        
        # Readout: average the node embeddings from all omics types
        all_embeddings = torch.cat([node_features[omics_type] for omics_type in self.omics_types], dim=0)
        graph_embedding = torch.mean(all_embeddings, dim=0)
        
        # Prediction
        predictions = self.prediction_layers(graph_embedding)
        
        return self.final_activation(predictions)
