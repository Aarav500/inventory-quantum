"""
Graph Neural Networks for Supply Chain Modeling.

Models warehouse-store relationships as a graph.
Reference: Kipf & Welling (2017) - Semi-Supervised Classification with GCN
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class SupplyChainNode:
    """Node in supply chain graph."""
    node_id: str
    node_type: str  # 'warehouse', 'store', 'supplier'
    features: np.ndarray
    demand: Optional[float] = None
    capacity: Optional[float] = None


@dataclass
class SupplyChainEdge:
    """Edge representing flow/connection."""
    source: str
    target: str
    weight: float  # e.g., shipping cost, distance
    lead_time: int


class GraphConvolutionLayer:
    """
    Graph Convolution Layer.
    
    H' = σ(D^(-1/2) A D^(-1/2) H W)
    """
    
    def __init__(self, in_features: int, out_features: int):
        self.in_features = in_features
        self.out_features = out_features
        # Xavier initialization
        self.W = np.random.randn(in_features, out_features) * np.sqrt(2 / (in_features + out_features))
        self.bias = np.zeros(out_features)
    
    def forward(self, H: np.ndarray, A_normalized: np.ndarray) -> np.ndarray:
        """
        Forward pass.
        
        Args:
            H: Node features (N, in_features)
            A_normalized: Normalized adjacency matrix
        """
        # Message passing: aggregate neighbor features
        aggregated = A_normalized @ H
        
        # Transform
        output = aggregated @ self.W + self.bias
        
        # ReLU activation
        return np.maximum(0, output)


class SupplyChainGNN:
    """
    Graph Neural Network for supply chain demand propagation.
    
    Key insight: Demand at stores affects upstream warehouse planning.
    """
    
    def __init__(self, node_features: int, hidden_dim: int = 32, output_dim: int = 1):
        self.layer1 = GraphConvolutionLayer(node_features, hidden_dim)
        self.layer2 = GraphConvolutionLayer(hidden_dim, hidden_dim)
        self.layer3 = GraphConvolutionLayer(hidden_dim, output_dim)
        
        self.nodes: Dict[str, SupplyChainNode] = {}
        self.edges: List[SupplyChainEdge] = []
        self.A = None
        self.A_normalized = None
    
    def add_node(self, node: SupplyChainNode):
        """Add node to graph."""
        self.nodes[node.node_id] = node
    
    def add_edge(self, edge: SupplyChainEdge):
        """Add edge to graph."""
        self.edges.append(edge)
    
    def build_adjacency_matrix(self) -> np.ndarray:
        """Build adjacency matrix from edges."""
        node_ids = list(self.nodes.keys())
        n = len(node_ids)
        node_idx = {nid: i for i, nid in enumerate(node_ids)}
        
        A = np.zeros((n, n))
        for edge in self.edges:
            if edge.source in node_idx and edge.target in node_idx:
                i, j = node_idx[edge.source], node_idx[edge.target]
                A[i, j] = edge.weight
                A[j, i] = edge.weight  # Undirected
        
        self.A = A
        return A
    
    def normalize_adjacency(self) -> np.ndarray:
        """
        Symmetric normalization: D^(-1/2) A D^(-1/2)
        """
        if self.A is None:
            self.build_adjacency_matrix()
        
        # Add self-loops
        A_hat = self.A + np.eye(len(self.A))
        
        # Degree matrix
        D = np.diag(np.sum(A_hat, axis=1))
        D_inv_sqrt = np.diag(1 / np.sqrt(np.diag(D) + 1e-8))
        
        self.A_normalized = D_inv_sqrt @ A_hat @ D_inv_sqrt
        return self.A_normalized
    
    def get_node_features(self) -> np.ndarray:
        """Stack all node features."""
        node_ids = list(self.nodes.keys())
        return np.vstack([self.nodes[nid].features for nid in node_ids])
    
    def forward(self) -> np.ndarray:
        """
        Forward pass through GNN.
        
        Returns node-level predictions (e.g., demand propagation).
        """
        if self.A_normalized is None:
            self.normalize_adjacency()
        
        H = self.get_node_features()
        
        H = self.layer1.forward(H, self.A_normalized)
        H = self.layer2.forward(H, self.A_normalized)
        H = self.layer3.forward(H, self.A_normalized)
        
        return H
    
    def propagate_demand(self) -> Dict[str, float]:
        """
        Propagate store demand to warehouses.
        
        Uses message passing to aggregate downstream demand upstream.
        """
        predictions = self.forward()
        node_ids = list(self.nodes.keys())
        
        result = {}
        for i, nid in enumerate(node_ids):
            node = self.nodes[nid]
            if node.node_type == 'warehouse':
                # Aggregate demand from connected stores
                result[nid] = float(predictions[i, 0])
        
        return result


def build_sample_supply_chain() -> SupplyChainGNN:
    """Build a sample supply chain graph for demonstration."""
    gnn = SupplyChainGNN(node_features=4, hidden_dim=16, output_dim=1)
    
    # Warehouse nodes
    gnn.add_node(SupplyChainNode(
        node_id='WH1',
        node_type='warehouse',
        features=np.array([1, 0, 0, 1000]),  # type_encoding, lat, lon, capacity
        capacity=10000
    ))
    
    gnn.add_node(SupplyChainNode(
        node_id='WH2',
        node_type='warehouse',
        features=np.array([1, 0, 1, 800]),
        capacity=8000
    ))
    
    # Store nodes
    for i in range(5):
        gnn.add_node(SupplyChainNode(
            node_id=f'STORE{i}',
            node_type='store',
            features=np.array([0, i*0.1, i*0.1, 100+i*20]),
            demand=50 + np.random.rand() * 50
        ))
    
    # Edges (warehouse to stores)
    for i in range(3):
        gnn.add_edge(SupplyChainEdge('WH1', f'STORE{i}', weight=1.0, lead_time=2))
    for i in range(3, 5):
        gnn.add_edge(SupplyChainEdge('WH2', f'STORE{i}', weight=1.0, lead_time=3))
    
    # Inter-warehouse edge
    gnn.add_edge(SupplyChainEdge('WH1', 'WH2', weight=0.5, lead_time=1))
    
    return gnn
