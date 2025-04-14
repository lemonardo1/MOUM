"""
GNN-based generator for multi-omics data integration.

This module implements a GNN-based generator that learns to generate integrated representations
of multi-omics data while preserving the relationships between different omics types.

Key Features:
1. Individual omics data generation
2. GNN-based integration of multiple omics types
3. Conditional generation based on cell type or drug information
4. Flexible graph construction (fully connected or predefined adjacency)
5. Support for various GNN architectures (GCN, GAT, SAGE, GIN)

Example Usage:
    # Initialize generator
    omics_dims = {
        'gene_expression': 1000,
        'methylation': 500,
        'copy_number': 200
    }
    generator = MultiOmicsGenerator(omics_dims)
    
    # Generate data
    latent_vectors = torch.randn(len(omics_dims), latent_dim)
    generated_data = generator(latent_vectors)
    
    # Conditional generation
    condition = torch.randn(condition_dim)
    generated_data = conditional_generator(latent_vectors, condition)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv
from torch_geometric.data import Data, Batch


class OmicsGenerator(nn.Module):
    """
    Generator for individual omics data types.
    
    This class implements a generator that transforms latent vectors into omics data
    for a specific omics type. It uses a simple feed-forward neural network with
    batch normalization and dropout for regularization.
    
    Attributes:
        layers (nn.Sequential): Neural network layers for data generation
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.2):
        """
        Initialize the generator for a specific omics data type.
        
        Args:
            input_dim (int): Input dimension (latent space)
            hidden_dim (int): Hidden layer dimension
            output_dim (int): Output dimension (omics data dimension)
            dropout (float, optional): Dropout rate. Defaults to 0.2.
        """
        super(OmicsGenerator, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim)
        )
        
    def forward(self, x):
        """
        Forward pass through the generator.
        
        Args:
            x (torch.Tensor): Input latent vector
            
        Returns:
            torch.Tensor: Generated omics data
        """
        return self.layers(x)


class MultiOmicsGenerator(nn.Module):
    """
    GNN-based generator for multi-omics data integration.
    
    This class implements a generator that learns to generate integrated representations
    of multi-omics data while preserving the relationships between different omics types.
    It uses GNN layers to refine the latent space and individual generators for each
    omics type.
    
    Attributes:
        omics_types (list): List of omics types
        latent_dim (int): Dimension of the latent space
        generators (nn.ModuleDict): Dictionary of generators for each omics type
        gnn_layers (nn.ModuleList): List of GNN layers for latent space refinement
    """
    
    def __init__(self, omics_dims, latent_dim=64, hidden_dim=256, gnn_type='GCN', 
                 num_gnn_layers=2, dropout=0.2):
        """
        Initialize the GNN-based multi-omics generator.
        
        Args:
            omics_dims (dict): Dictionary mapping omics types to their output dimensions
            latent_dim (int, optional): Dimension of the latent space. Defaults to 64.
            hidden_dim (int, optional): Hidden layer dimension. Defaults to 256.
            gnn_type (str, optional): Type of GNN layer ('GCN', 'GAT', 'SAGE', 'GIN'). Defaults to 'GCN'.
            num_gnn_layers (int, optional): Number of GNN layers. Defaults to 2.
            dropout (float, optional): Dropout rate. Defaults to 0.2.
        """
        super(MultiOmicsGenerator, self).__init__()
        
        self.omics_types = list(omics_dims.keys())
        self.latent_dim = latent_dim
        
        # Create generators for each omics type
        self.generators = nn.ModuleDict({
            omics_type: OmicsGenerator(
                input_dim=latent_dim,
                hidden_dim=hidden_dim,
                output_dim=dim,
                dropout=dropout
            )
            for omics_type, dim in omics_dims.items()
        })
        
        # GNN layers for latent space refinement
        self.gnn_layers = nn.ModuleList()
        
        for _ in range(num_gnn_layers):
            if gnn_type == 'GCN':
                self.gnn_layers.append(GCNConv(latent_dim, latent_dim))
            elif gnn_type == 'GAT':
                self.gnn_layers.append(GATConv(latent_dim, latent_dim))
            elif gnn_type == 'SAGE':
                self.gnn_layers.append(SAGEConv(latent_dim, latent_dim))
            elif gnn_type == 'GIN':
                nn_layer = nn.Sequential(
                    nn.Linear(latent_dim, latent_dim),
                    nn.ReLU(),
                    nn.Linear(latent_dim, latent_dim)
                )
                self.gnn_layers.append(GINConv(nn_layer))
            else:
                raise ValueError(f"Unsupported GNN type: {gnn_type}")
    
    def _construct_latent_graph(self, latent_vectors, adjacency_matrix=None):
        """
        Construct a graph in the latent space.
        
        This method creates a graph structure in the latent space, either using
        a predefined adjacency matrix or creating a fully connected graph.
        
        Args:
            latent_vectors (torch.Tensor): Latent vectors for each omics type
            adjacency_matrix (torch.Tensor, optional): Predefined adjacency matrix.
                If None, a fully connected graph is created. Defaults to None.
        
        Returns:
            torch_geometric.data.Data: The constructed graph
        """
        if adjacency_matrix is not None:
            edge_index = adjacency_matrix.nonzero().t().contiguous()
        else:
            # Create a fully connected graph
            num_nodes = latent_vectors.shape[0]
            source_nodes = []
            target_nodes = []
            
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j:  # Exclude self-loops
                        source_nodes.append(i)
                        target_nodes.append(j)
            
            edge_index = torch.tensor([source_nodes, target_nodes], device=latent_vectors.device)
        
        return Data(x=latent_vectors, edge_index=edge_index)
    
    def forward(self, latent_vectors, adjacency_matrix=None):
        """
        Forward pass through the multi-omics generator.
        
        This method takes latent vectors and refines them through GNN layers to
        generate integrated multi-omics data.
        
        Args:
            latent_vectors (torch.Tensor): Latent vectors for each omics type
            adjacency_matrix (torch.Tensor, optional): Predefined adjacency matrix. Defaults to None.
        
        Returns:
            dict: Dictionary mapping omics types to their generated data
        """
        # Refine latent vectors through GNN layers
        graph = self._construct_latent_graph(latent_vectors, adjacency_matrix)
        x, edge_index = graph.x, graph.edge_index
        
        for gnn_layer in self.gnn_layers:
            x = gnn_layer(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.2, training=self.training)
        
        # Generate omics data for each type
        generated_data = {}
        for i, omics_type in enumerate(self.omics_types):
            generated_data[omics_type] = self.generators[omics_type](x[i])
        
        return generated_data


class ConditionalMultiOmicsGenerator(MultiOmicsGenerator):
    """
    Conditional GNN-based generator for multi-omics data integration.
    
    This class extends the MultiOmicsGenerator to support conditional generation
    based on additional information such as cell type or drug information.
    
    Attributes:
        condition_encoder (nn.Sequential): Neural network for encoding condition information
    """
    
    def __init__(self, omics_dims, condition_dim, latent_dim=64, hidden_dim=256, 
                 gnn_type='GCN', num_gnn_layers=2, dropout=0.2):
        """
        Initialize the conditional GNN-based multi-omics generator.
        
        Args:
            omics_dims (dict): Dictionary mapping omics types to their output dimensions
            condition_dim (int): Dimension of the condition vector
            latent_dim (int, optional): Dimension of the latent space. Defaults to 64.
            hidden_dim (int, optional): Hidden layer dimension. Defaults to 256.
            gnn_type (str, optional): Type of GNN layer. Defaults to 'GCN'.
            num_gnn_layers (int, optional): Number of GNN layers. Defaults to 2.
            dropout (float, optional): Dropout rate. Defaults to 0.2.
        """
        super(ConditionalMultiOmicsGenerator, self).__init__(
            omics_dims=omics_dims,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            gnn_type=gnn_type,
            num_gnn_layers=num_gnn_layers,
            dropout=dropout
        )
        
        # Condition encoder
        self.condition_encoder = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
    
    def forward(self, latent_vectors, condition, adjacency_matrix=None):
        """
        Forward pass through the conditional multi-omics generator.
        
        This method takes latent vectors and condition information, encodes the condition,
        and generates integrated multi-omics data conditioned on the input condition.
        
        Args:
            latent_vectors (torch.Tensor): Latent vectors for each omics type
            condition (torch.Tensor): Condition vector
            adjacency_matrix (torch.Tensor, optional): Predefined adjacency matrix. Defaults to None.
        
        Returns:
            dict: Dictionary mapping omics types to their generated data
        """
        # Encode condition
        condition_embedding = self.condition_encoder(condition)
        
        # Combine latent vectors with condition
        conditioned_latent = latent_vectors + condition_embedding.unsqueeze(0)
        
        # Generate omics data using the conditioned latent vectors
        return super().forward(conditioned_latent, adjacency_matrix) 