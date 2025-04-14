"""
GNN-based generator for multi-omics data integration in TensorFlow.

This module implements a GNN-based generator that learns to generate integrated representations
of multi-omics data while preserving the relationships between different omics types.
"""

import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, LayerNormalization


class OmicsGenerator(Model):
    """
    Generator for individual omics data types in TensorFlow.
    
    This class implements a generator that transforms latent vectors into omics data
    for a specific omics type.
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
        
        self.layers = [
            Dense(hidden_dim),
            BatchNormalization(),
            layers.ReLU(),
            Dropout(dropout),
            Dense(output_dim),
            BatchNormalization()
        ]
        
    def call(self, x, training=False):
        """
        Forward pass through the generator.
        
        Args:
            x (tf.Tensor): Input latent vector
            training (bool, optional): Whether the model is in training mode. Defaults to False.
        
        Returns:
            tf.Tensor: Generated omics data
        """
        for layer in self.layers:
            if isinstance(layer, (BatchNormalization, Dropout)):
                x = layer(x, training=training)
            else:
                x = layer(x)
        return x


class MultiOmicsGenerator(Model):
    """
    GNN-based generator for multi-omics data integration in TensorFlow.
    
    This class implements a generator that learns to generate integrated representations
    of multi-omics data while preserving the relationships between different omics types.
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
        self.generators = {
            omics_type: OmicsGenerator(
                input_dim=latent_dim,
                hidden_dim=hidden_dim,
                output_dim=dim,
                dropout=dropout
            )
            for omics_type, dim in omics_dims.items()
        }
        
        # GNN layers for latent space refinement
        self.gnn_layers = []
        for _ in range(num_gnn_layers):
            if gnn_type == 'GCN':
                self.gnn_layers.append(tfgnn.layers.GCNConv(latent_dim))
            elif gnn_type == 'GAT':
                self.gnn_layers.append(tfgnn.layers.GATConv(latent_dim))
            else:
                raise ValueError(f"Unsupported GNN type: {gnn_type}")
    
    def _construct_latent_graph(self, latent_vectors, adjacency_matrix=None):
        """
        Construct a graph in the latent space.
        
        Args:
            latent_vectors (tf.Tensor): Latent vectors for each omics type
            adjacency_matrix (tf.Tensor, optional): Predefined adjacency matrix.
                If None, a fully connected graph is created. Defaults to None.
        
        Returns:
            tfgnn.GraphTensor: The constructed graph
        """
        if adjacency_matrix is not None:
            edge_index = tf.where(adjacency_matrix)
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
            
            edge_index = tf.stack([source_nodes, target_nodes], axis=0)
        
        # Create graph tensor
        graph = tfgnn.GraphTensor.from_pieces(
            node_sets={
                'nodes': tfgnn.NodeSet.from_fields(
                    sizes=tf.shape(latent_vectors)[0:1],
                    features={'features': latent_vectors}
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
        
        return graph
    
    def call(self, latent_vectors, adjacency_matrix=None, training=False):
        """
        Forward pass through the multi-omics generator.
        
        Args:
            latent_vectors (tf.Tensor): Latent vectors for each omics type
            adjacency_matrix (tf.Tensor, optional): Predefined adjacency matrix. Defaults to None.
            training (bool, optional): Whether the model is in training mode. Defaults to False.
        
        Returns:
            dict: Dictionary mapping omics types to their generated data
        """
        # Refine latent vectors through GNN layers
        graph = self._construct_latent_graph(latent_vectors, adjacency_matrix)
        x = graph.node_sets['nodes']['features']
        
        for gnn_layer in self.gnn_layers:
            x = gnn_layer(graph, node_set_name='nodes', feature_name='features')
            x = tf.nn.relu(x)
            x = tf.nn.dropout(x, rate=0.2, training=training)
        
        # Generate omics data for each type
        generated_data = {}
        for i, omics_type in enumerate(self.omics_types):
            generated_data[omics_type] = self.generators[omics_type](x[i], training=training)
        
        return generated_data


class ConditionalMultiOmicsGenerator(MultiOmicsGenerator):
    """
    Conditional GNN-based generator for multi-omics data integration in TensorFlow.
    
    This class extends the MultiOmicsGenerator to support conditional generation
    based on additional information such as cell type or drug information.
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
        self.condition_encoder = [
            Dense(hidden_dim, activation='relu'),
            Dense(latent_dim)
        ]
    
    def call(self, latent_vectors, condition, adjacency_matrix=None, training=False):
        """
        Forward pass through the conditional multi-omics generator.
        
        Args:
            latent_vectors (tf.Tensor): Latent vectors for each omics type
            condition (tf.Tensor): Condition vector
            adjacency_matrix (tf.Tensor, optional): Predefined adjacency matrix. Defaults to None.
            training (bool, optional): Whether the model is in training mode. Defaults to False.
        
        Returns:
            dict: Dictionary mapping omics types to their generated data
        """
        # Encode condition
        condition_embedding = condition
        for layer in self.condition_encoder:
            condition_embedding = layer(condition_embedding)
        
        # Combine latent vectors with condition
        conditioned_latent = latent_vectors + tf.expand_dims(condition_embedding, 0)
        
        # Generate omics data using the conditioned latent vectors
        return super().call(conditioned_latent, adjacency_matrix, training=training) 