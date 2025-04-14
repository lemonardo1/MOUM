"""
Multi-omics data integration using Graph Neural Networks in TensorFlow.

This module implements a GNN-based approach for integrating multiple types of omics data
using TensorFlow and TF-GNN.
"""

import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, LayerNormalization
import numpy as np


class OmicsEncoder(Model):
    """Base encoder for individual omics data types in TensorFlow."""
    
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
        
        self.dense1 = Dense(hidden_dim)
        self.bn1 = BatchNormalization()
        self.dropout = Dropout(dropout)
        self.dense2 = Dense(output_dim)
        self.bn2 = BatchNormalization()
        
    def call(self, x, training=False):
        """Forward pass through the encoder."""
        x = self.dense1(x)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        x = self.bn2(x, training=training)
        return x


class MultiOmicsGNN(Model):
    """GNN-based model for multi-omics data integration in TensorFlow."""
    
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
            num_tasks (int, optional): Number of tasks. Defaults to 1.
        """
        super(MultiOmicsGNN, self).__init__()
        
        self.omics_types = list(omics_dims.keys())
        self.embedding_dim = embedding_dim
        self.task_type = task_type
        self.num_tasks = num_tasks
        
        # Create encoders for each omics type
        self.encoders = {
            omics_type: OmicsEncoder(
                input_dim=dim,
                hidden_dim=hidden_dim,
                output_dim=embedding_dim,
                dropout=dropout
            )
            for omics_type, dim in omics_dims.items()
        }
        
        # GNN layers
        self.gnn_layers = []
        for _ in range(num_gnn_layers):
            if gnn_type == 'GCN':
                self.gnn_layers.append(tfgnn.layers.GCNConv(embedding_dim))
            elif gnn_type == 'GAT':
                self.gnn_layers.append(tfgnn.layers.GATConv(embedding_dim))
            else:
                raise ValueError(f"Unsupported GNN type: {gnn_type}")
        
        # Prediction layers
        self.prediction_layers = [
            Dense(hidden_dim, activation='relu'),
            Dropout(dropout),
            Dense(num_tasks)
        ]
        
        if task_type == 'classification':
            self.final_activation = tf.nn.sigmoid
        else:  # regression
            self.final_activation = tf.identity
    
    def _construct_graph(self, omics_embeddings, adjacency_matrix=None):
        """
        Construct a graph from omics embeddings.
        
        Args:
            omics_embeddings (dict): Dictionary mapping omics types to their embeddings
            adjacency_matrix (tf.Tensor, optional): Predefined adjacency matrix for the graph.
                If None, a fully connected graph is created. Defaults to None.
        
        Returns:
            tfgnn.GraphTensor: The constructed graph
        """
        # Concatenate all node features (omics embeddings)
        nodes = []
        node_types = []
        
        for i, (omics_type, embedding) in enumerate(omics_embeddings.items()):
            nodes.append(embedding)
            node_types.extend([i] * embedding.shape[0])
        
        x = tf.concat(nodes, axis=0)
        node_types = tf.convert_to_tensor(node_types, dtype=tf.int32)
        
        # Create edges (either from adjacency matrix or fully connected)
        if adjacency_matrix is not None:
            edge_index = tf.where(adjacency_matrix)
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
            
            edge_index = tf.stack([source_nodes, target_nodes], axis=0)
        
        # Create graph tensor
        graph = tfgnn.GraphTensor.from_pieces(
            node_sets={
                'nodes': tfgnn.NodeSet.from_fields(
                    sizes=tf.shape(x)[0:1],
                    features={'features': x, 'type': node_types}
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
    
    def call(self, omics_data, adjacency_matrix=None, training=False):
        """
        Forward pass through the multi-omics GNN model.
        
        Args:
            omics_data (dict): Dictionary mapping omics types to their data tensors
            adjacency_matrix (tf.Tensor, optional): Predefined adjacency matrix. Defaults to None.
            training (bool, optional): Whether the model is in training mode. Defaults to False.
        
        Returns:
            tf.Tensor: Model predictions
        """
        # Encode each omics type
        omics_embeddings = {}
        for omics_type in self.omics_types:
            if omics_type in omics_data:
                omics_embeddings[omics_type] = self.encoders[omics_type](
                    omics_data[omics_type], training=training
                )
        
        # Construct graph
        graph = self._construct_graph(omics_embeddings, adjacency_matrix)
        
        # Apply GNN layers
        x = graph.node_sets['nodes']['features']
        for gnn_layer in self.gnn_layers:
            x = gnn_layer(graph, node_set_name='nodes', feature_name='features')
            x = tf.nn.relu(x)
            x = tf.nn.dropout(x, rate=0.2, training=training)
        
        # Readout: average the node embeddings
        graph_embedding = tf.reduce_mean(x, axis=0)
        
        # Prediction
        for layer in self.prediction_layers:
            graph_embedding = layer(graph_embedding, training=training)
        
        return self.final_activation(graph_embedding)


class HeterogeneousOmicsGNN(Model):
    """
    Heterogeneous Graph Neural Network for multi-omics integration in TensorFlow.
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
        self.encoders = {
            omics_type: OmicsEncoder(
                input_dim=dim,
                hidden_dim=hidden_dim,
                output_dim=embedding_dim,
                dropout=dropout
            )
            for omics_type, dim in omics_dims.items()
        }
        
        # Heterogeneous GNN layers
        self.hgnn_layers = []
        for _ in range(num_gnn_layers):
            # One GCNConv per omics type relation
            relation_convs = {}
            for src_type in self.omics_types:
                for dst_type in self.omics_types:
                    relation_name = f"{src_type}_to_{dst_type}"
                    relation_convs[relation_name] = tfgnn.layers.GCNConv(embedding_dim)
            
            self.hgnn_layers.append(relation_convs)
        
        # Prediction layers
        self.prediction_layers = [
            Dense(hidden_dim, activation='relu'),
            Dropout(dropout),
            Dense(num_tasks)
        ]
        
        if task_type == 'classification':
            self.final_activation = tf.nn.sigmoid
        else:  # regression
            self.final_activation = tf.identity
    
    def call(self, omics_data, edge_indices_dict, training=False):
        """
        Forward pass through the heterogeneous multi-omics GNN model.
        
        Args:
            omics_data (dict): Dictionary mapping omics types to their data tensors
            edge_indices_dict (dict): Dictionary mapping relation names to edge indices
                Format: {f"{src_type}_to_{dst_type}": edge_index_tensor}
            training (bool, optional): Whether the model is in training mode. Defaults to False.
        
        Returns:
            tf.Tensor: Model predictions
        """
        # Encode each omics type
        node_features = {}
        for omics_type in self.omics_types:
            if omics_type in omics_data:
                node_features[omics_type] = self.encoders[omics_type](
                    omics_data[omics_type], training=training
                )
        
        # Apply heterogeneous GNN layers
        for layer in self.hgnn_layers:
            new_features = {
                omics_type: tf.zeros_like(feat)
                for omics_type, feat in node_features.items()
            }
            
            # Aggregate messages from each relation
            for src_type in self.omics_types:
                for dst_type in self.omics_types:
                    relation_name = f"{src_type}_to_{dst_type}"
                    
                    if relation_name in edge_indices_dict:
                        edge_index = edge_indices_dict[relation_name]
                        conv = layer[relation_name]
                        
                        # Create graph for this relation
                        graph = tfgnn.GraphTensor.from_pieces(
                            node_sets={
                                'nodes': tfgnn.NodeSet.from_fields(
                                    sizes=tf.shape(node_features[src_type])[0:1],
                                    features={'features': node_features[src_type]}
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
                        
                        # Apply the appropriate convolution
                        dst_features = conv(graph, node_set_name='nodes', feature_name='features')
                        new_features[dst_type] += dst_features
            
            # Apply non-linearity and update features
            for omics_type in self.omics_types:
                node_features[omics_type] = tf.nn.relu(new_features[omics_type])
                node_features[omics_type] = tf.nn.dropout(
                    node_features[omics_type], rate=0.2, training=training
                )
        
        # Readout: average the node embeddings from all omics types
        all_embeddings = tf.concat(
            [node_features[omics_type] for omics_type in self.omics_types], axis=0
        )
        graph_embedding = tf.reduce_mean(all_embeddings, axis=0)
        
        # Prediction
        for layer in self.prediction_layers:
            graph_embedding = layer(graph_embedding, training=training)
        
        return self.final_activation(graph_embedding) 