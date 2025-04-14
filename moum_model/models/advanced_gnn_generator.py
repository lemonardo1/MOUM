"""
Advanced GNN-based generator for multi-omics data integration.

This module implements an enhanced version of the GNN-based generator with:
1. Attention-based graph construction
2. Multi-head attention for conditional generation
3. Residual connections for better gradient flow
4. Layer normalization for stable training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv
from torch_geometric.data import Data, Batch


class AttentionGraphConstructor(nn.Module):
    """
    Attention-based graph constructor.
    
    This module learns to construct a graph by computing attention scores
    between nodes in the latent space.
    """
    
    def __init__(self, latent_dim, num_heads=4, dropout=0.2):
        """
        Initialize the attention-based graph constructor.
        
        Args:
            latent_dim (int): Dimension of the latent space
            num_heads (int, optional): Number of attention heads. Defaults to 4.
            dropout (float, optional): Dropout rate. Defaults to 0.2.
        """
        super(AttentionGraphConstructor, self).__init__()
        
        self.num_heads = num_heads
        self.latent_dim = latent_dim
        self.head_dim = latent_dim // num_heads
        
        # Linear transformations for query, key, and value
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(latent_dim, latent_dim)
        self.value = nn.Linear(latent_dim, latent_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(latent_dim)
        
    def forward(self, x):
        """
        Compute attention scores and construct the graph.
        
        Args:
            x (torch.Tensor): Input latent vectors
            
        Returns:
            torch.Tensor: Attention-based adjacency matrix
        """
        batch_size = x.size(0)
        
        # Linear transformations
        q = self.query(x).view(batch_size, -1, self.num_heads, self.head_dim)
        k = self.key(x).view(batch_size, -1, self.num_heads, self.head_dim)
        v = self.value(x).view(batch_size, -1, self.num_heads, self.head_dim)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Compute weighted sum
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.latent_dim)
        
        # Layer normalization
        out = self.layer_norm(out)
        
        return out, attn


class ResidualGNNLayer(nn.Module):
    """
    GNN layer with residual connections.
    
    This class implements a GNN layer with residual connections and layer normalization
    for better gradient flow and stable training.
    """
    
    def __init__(self, in_channels, out_channels, gnn_type='GCN'):
        """
        Initialize the residual GNN layer.
        
        Args:
            in_channels (int): Input dimension
            out_channels (int): Output dimension
            gnn_type (str, optional): Type of GNN layer. Defaults to 'GCN'.
        """
        super(ResidualGNNLayer, self).__init__()
        
        if gnn_type == 'GCN':
            self.gnn = GCNConv(in_channels, out_channels)
        elif gnn_type == 'GAT':
            self.gnn = GATConv(in_channels, out_channels)
        elif gnn_type == 'SAGE':
            self.gnn = SAGEConv(in_channels, out_channels)
        elif gnn_type == 'GIN':
            nn_layer = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels)
            )
            self.gnn = GINConv(nn_layer)
        else:
            raise ValueError(f"Unsupported GNN type: {gnn_type}")
        
        self.layer_norm = nn.LayerNorm(out_channels)
        
    def forward(self, x, edge_index):
        """
        Forward pass through the residual GNN layer.
        
        Args:
            x (torch.Tensor): Node features
            edge_index (torch.Tensor): Edge indices
            
        Returns:
            torch.Tensor: Updated node features
        """
        # GNN operation
        out = self.gnn(x, edge_index)
        
        # Residual connection and layer normalization
        if x.size(-1) == out.size(-1):
            out = self.layer_norm(x + out)
        else:
            out = self.layer_norm(out)
            
        return out


class AdvancedMultiOmicsGenerator(nn.Module):
    """
    Advanced GNN-based generator for multi-omics data integration.
    
    This class implements an enhanced version of the multi-omics generator with
    attention-based graph construction, residual connections, and multi-head attention
    for conditional generation.
    """
    
    def __init__(self, omics_dims, latent_dim=64, hidden_dim=256, gnn_type='GCN',
                 num_gnn_layers=2, num_heads=4, dropout=0.2):
        """
        Initialize the advanced multi-omics generator.
        
        Args:
            omics_dims (dict): Dictionary mapping omics types to their output dimensions
            latent_dim (int, optional): Dimension of the latent space. Defaults to 64.
            hidden_dim (int, optional): Hidden layer dimension. Defaults to 256.
            gnn_type (str, optional): Type of GNN layer. Defaults to 'GCN'.
            num_gnn_layers (int, optional): Number of GNN layers. Defaults to 2.
            num_heads (int, optional): Number of attention heads. Defaults to 4.
            dropout (float, optional): Dropout rate. Defaults to 0.2.
        """
        super(AdvancedMultiOmicsGenerator, self).__init__()
        
        self.omics_types = list(omics_dims.keys())
        self.latent_dim = latent_dim
        
        # Attention-based graph constructor
        self.graph_constructor = AttentionGraphConstructor(
            latent_dim=latent_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
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
        
        # Residual GNN layers
        self.gnn_layers = nn.ModuleList([
            ResidualGNNLayer(latent_dim, latent_dim, gnn_type)
            for _ in range(num_gnn_layers)
        ])
    
    def forward(self, latent_vectors, adjacency_matrix=None):
        """
        Forward pass through the advanced multi-omics generator.
        
        Args:
            latent_vectors (torch.Tensor): Latent vectors for each omics type
            adjacency_matrix (torch.Tensor, optional): Predefined adjacency matrix. Defaults to None.
        
        Returns:
            dict: Dictionary mapping omics types to their generated data
        """
        # Construct graph using attention
        x, attn = self.graph_constructor(latent_vectors)
        
        # Use predefined adjacency matrix if provided
        if adjacency_matrix is not None:
            edge_index = adjacency_matrix.nonzero().t().contiguous()
        else:
            # Use attention scores to construct edges
            edge_index = attn.topk(k=5, dim=-1)[1].view(-1, 2).t()
        
        # Apply residual GNN layers
        for gnn_layer in self.gnn_layers:
            x = gnn_layer(x, edge_index)
        
        # Generate omics data for each type
        generated_data = {}
        for i, omics_type in enumerate(self.omics_types):
            generated_data[omics_type] = self.generators[omics_type](x[i])
        
        return generated_data


class AdvancedConditionalMultiOmicsGenerator(AdvancedMultiOmicsGenerator):
    """
    Advanced conditional GNN-based generator for multi-omics data integration.
    
    This class extends the AdvancedMultiOmicsGenerator to support conditional generation
    using multi-head attention.
    """
    
    def __init__(self, omics_dims, condition_dim, latent_dim=64, hidden_dim=256,
                 gnn_type='GCN', num_gnn_layers=2, num_heads=4, dropout=0.2):
        """
        Initialize the advanced conditional multi-omics generator.
        
        Args:
            omics_dims (dict): Dictionary mapping omics types to their output dimensions
            condition_dim (int): Dimension of the condition vector
            latent_dim (int, optional): Dimension of the latent space. Defaults to 64.
            hidden_dim (int, optional): Hidden layer dimension. Defaults to 256.
            gnn_type (str, optional): Type of GNN layer. Defaults to 'GCN'.
            num_gnn_layers (int, optional): Number of GNN layers. Defaults to 2.
            num_heads (int, optional): Number of attention heads. Defaults to 4.
            dropout (float, optional): Dropout rate. Defaults to 0.2.
        """
        super(AdvancedConditionalMultiOmicsGenerator, self).__init__(
            omics_dims=omics_dims,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            gnn_type=gnn_type,
            num_gnn_layers=num_gnn_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Multi-head attention for condition integration
        self.condition_attention = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.condition_projection = nn.Sequential(
            nn.Linear(condition_dim, latent_dim),
            nn.LayerNorm(latent_dim)
        )
    
    def forward(self, latent_vectors, condition, adjacency_matrix=None):
        """
        Forward pass through the advanced conditional multi-omics generator.
        
        Args:
            latent_vectors (torch.Tensor): Latent vectors for each omics type
            condition (torch.Tensor): Condition vector
            adjacency_matrix (torch.Tensor, optional): Predefined adjacency matrix. Defaults to None.
        
        Returns:
            dict: Dictionary mapping omics types to their generated data
        """
        # Project condition to latent space
        condition_embedding = self.condition_projection(condition)
        
        # Apply multi-head attention between condition and latent vectors
        condition_embedding = condition_embedding.unsqueeze(0)  # Add batch dimension
        latent_vectors = latent_vectors.unsqueeze(0)  # Add batch dimension
        
        attended_latent, _ = self.condition_attention(
            query=latent_vectors,
            key=condition_embedding,
            value=condition_embedding
        )
        
        # Remove batch dimension
        attended_latent = attended_latent.squeeze(0)
        
        # Generate data using the attended latent vectors
        return super().forward(attended_latent, adjacency_matrix) 