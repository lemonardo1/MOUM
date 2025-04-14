"""
Advanced GNN-based generator for multi-omics data integration in TensorFlow.

This module implements an enhanced version of the GNN-based generator with:
1. Attention-based graph construction
2. Multi-head attention for conditional generation
3. Residual connections for better gradient flow
4. Layer normalization for stable training
"""

import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, LayerNormalization, MultiHeadAttention


class AttentionGraphConstructor(Model):
    """
    Attention-based graph constructor in TensorFlow.
    
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
        
        # Multi-head attention
        self.mha = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=self.head_dim,
            dropout=dropout
        )
        
        self.layer_norm = LayerNormalization()
        self.dropout = Dropout(dropout)
        
    def call(self, x, training=False):
        """
        Compute attention scores and construct the graph.
        
        Args:
            x (tf.Tensor): Input latent vectors
            training (bool, optional): Whether the model is in training mode. Defaults to False.
        
        Returns:
            tuple: (Updated node features, attention scores)
        """
        batch_size = tf.shape(x)[0]
        
        # Apply multi-head attention
        attended_output, attention_scores = self.mha(
            x, x, x,
            return_attention_scores=True,
            training=training
        )
        
        # Add & norm
        x = self.layer_norm(x + attended_output)
        x = self.dropout(x, training=training)
        
        return x, attention_scores


class ResidualGNNLayer(Model):
    """
    GNN layer with residual connections in TensorFlow.
    
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
            self.gnn = tfgnn.layers.GCNConv(out_channels)
        elif gnn_type == 'GAT':
            self.gnn = tfgnn.layers.GATConv(out_channels)
        else:
            raise ValueError(f"Unsupported GNN type: {gnn_type}")
        
        self.layer_norm = LayerNormalization()
        
    def call(self, graph, training=False):
        """
        Forward pass through the residual GNN layer.
        
        Args:
            graph (tfgnn.GraphTensor): Input graph
            training (bool, optional): Whether the model is in training mode. Defaults to False.
        
        Returns:
            tfgnn.GraphTensor: Updated graph
        """
        # GNN operation
        x = graph.node_sets['nodes']['features']
        out = self.gnn(graph, node_set_name='nodes', feature_name='features')
        
        # Residual connection and layer normalization
        if x.shape[-1] == out.shape[-1]:
            out = self.layer_norm(x + out)
        else:
            out = self.layer_norm(out)
        
        # Update graph features
        return graph.replace_features(
            node_sets={
                'nodes': {
                    'features': out
                }
            }
        )


class AdvancedMultiOmicsGenerator(Model):
    """
    Advanced GNN-based generator for multi-omics data integration in TensorFlow.
    
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
        self.generators = {
            omics_type: OmicsGenerator(
                input_dim=latent_dim,
                hidden_dim=hidden_dim,
                output_dim=dim,
                dropout=dropout
            )
            for omics_type, dim in omics_dims.items()
        }
        
        # Residual GNN layers
        self.gnn_layers = [
            ResidualGNNLayer(latent_dim, latent_dim, gnn_type)
            for _ in range(num_gnn_layers)
        ]
    
    def call(self, latent_vectors, adjacency_matrix=None, training=False):
        """
        Forward pass through the advanced multi-omics generator.
        
        Args:
            latent_vectors (tf.Tensor): Latent vectors for each omics type
            adjacency_matrix (tf.Tensor, optional): Predefined adjacency matrix. Defaults to None.
            training (bool, optional): Whether the model is in training mode. Defaults to False.
        
        Returns:
            dict: Dictionary mapping omics types to their generated data
        """
        # Construct graph using attention
        x, attn = self.graph_constructor(latent_vectors, training=training)
        
        # Use predefined adjacency matrix if provided
        if adjacency_matrix is not None:
            edge_index = tf.where(adjacency_matrix)
        else:
            # Use attention scores to construct edges
            edge_index = tf.stack([
                tf.reshape(tf.argsort(attn, axis=-1)[..., -5:], [-1]),
                tf.reshape(tf.tile(tf.range(tf.shape(attn)[-1]), [5]), [-1])
            ], axis=0)
        
        # Create graph tensor
        graph = tfgnn.GraphTensor.from_pieces(
            node_sets={
                'nodes': tfgnn.NodeSet.from_fields(
                    sizes=tf.shape(x)[0:1],
                    features={'features': x}
                )
            },
            edge_sets={
                'edges': tfgnn.EdgeSet.from_fields(
                    sizes=tf.shape(edge_index)[1:2],
                    adjacency=tfgnn.Adjacency.from_indices(
                        source=('nodes', edge_index[0]),
                        target=('nodes', edge_index[1])
                    )
                )
            }
        )
        
        # Apply residual GNN layers
        for gnn_layer in self.gnn_layers:
            graph = gnn_layer(graph, training=training)
        
        # Generate omics data for each type
        x = graph.node_sets['nodes']['features']
        generated_data = {}
        for i, omics_type in enumerate(self.omics_types):
            generated_data[omics_type] = self.generators[omics_type](x[i], training=training)
        
        return generated_data


class AdvancedConditionalMultiOmicsGenerator(AdvancedMultiOmicsGenerator):
    """
    Advanced conditional GNN-based generator for multi-omics data integration in TensorFlow.
    
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
        self.condition_attention = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=latent_dim // num_heads,
            dropout=dropout
        )
        
        self.condition_projection = [
            Dense(latent_dim),
            LayerNormalization()
        ]
    
    def call(self, latent_vectors, condition, adjacency_matrix=None, training=False):
        """
        Forward pass through the advanced conditional multi-omics generator.
        
        Args:
            latent_vectors (tf.Tensor): Latent vectors for each omics type
            condition (tf.Tensor): Condition vector
            adjacency_matrix (tf.Tensor, optional): Predefined adjacency matrix. Defaults to None.
            training (bool, optional): Whether the model is in training mode. Defaults to False.
        
        Returns:
            dict: Dictionary mapping omics types to their generated data
        """
        # Project condition to latent space
        condition_embedding = condition
        for layer in self.condition_projection:
            condition_embedding = layer(condition_embedding)
        
        # Apply multi-head attention between condition and latent vectors
        condition_embedding = tf.expand_dims(condition_embedding, 0)  # Add batch dimension
        latent_vectors = tf.expand_dims(latent_vectors, 0)  # Add batch dimension
        
        attended_latent, _ = self.condition_attention(
            query=latent_vectors,
            key=condition_embedding,
            value=condition_embedding,
            training=training
        )
        
        # Remove batch dimension
        attended_latent = tf.squeeze(attended_latent, 0)
        
        # Generate data using the attended latent vectors
        return super().call(attended_latent, adjacency_matrix, training=training) 