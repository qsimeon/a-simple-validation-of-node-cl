"""Graph Dimensionality Reduction Module

This module implements a graph-based dimensionality reduction algorithm that
reconstructs node coordinates from edge distance features. Given a fully connected
graph where edges contain Euclidean distances between nodes, this module can:

1. Recover the original Euclidean coordinates of nodes from distance matrices
2. Perform dimensionality reduction by embedding high-dimensional points into
   lower-dimensional spaces while preserving pairwise distances
3. Validate the quality of coordinate recovery through visualization and metrics

The core algorithm uses Multidimensional Scaling (MDS) to reconstruct coordinates
from pairwise distance matrices.

Example:
    >>> from graph_dim_red import GraphDimensionalityReduction
    >>> gdr = GraphDimensionalityReduction(n_components=2)
    >>> true_coords = np.random.randn(50, 3)  # 50 points in 3D
    >>> recovered_coords = gdr.fit_transform(true_coords)
    >>> gdr.plot_comparison(true_coords, recovered_coords)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import pdist, squareform
from typing import Tuple, Optional
import warnings


class GraphDimensionalityReduction:
    """Graph-based dimensionality reduction using distance matrices.
    
    This class implements a dimensionality reduction technique that treats
    data points as nodes in a fully connected graph, where edge weights
    represent Euclidean distances between nodes. It uses MDS to recover
    coordinates from these distances.
    
    Attributes:
        n_components (int): Target dimensionality for the embedding
        distance_matrix (np.ndarray): Computed pairwise distance matrix
        mds (MDS): Fitted MDS model
        stress (float): MDS stress value (measure of fit quality)
    """
    
    def __init__(self, n_components: int = 2, random_state: Optional[int] = 42):
        """Initialize the GraphDimensionalityReduction.
        
        Args:
            n_components: Number of dimensions for the embedding (default: 2)
            random_state: Random seed for reproducibility (default: 42)
        """
        self.n_components = n_components
        self.random_state = random_state
        self.distance_matrix = None
        self.mds = None
        self.stress = None
        
    def compute_distance_matrix(self, coordinates: np.ndarray) -> np.ndarray:
        """Compute pairwise Euclidean distance matrix from coordinates.
        
        Args:
            coordinates: Array of shape (n_samples, n_features) containing
                        the coordinates of points
        
        Returns:
            Distance matrix of shape (n_samples, n_samples)
        """
        # Use scipy's pdist and squareform for efficient computation
        distance_matrix = squareform(pdist(coordinates, metric='euclidean'))
        self.distance_matrix = distance_matrix
        return distance_matrix
    
    def fit_transform(self, coordinates: np.ndarray) -> np.ndarray:
        """Fit MDS and transform coordinates to lower dimensions.
        
        Args:
            coordinates: Array of shape (n_samples, n_features) containing
                        the original coordinates
        
        Returns:
            Recovered coordinates of shape (n_samples, n_components)
        """
        # Compute distance matrix
        distance_matrix = self.compute_distance_matrix(coordinates)
        
        # Apply MDS to recover coordinates from distances
        self.mds = MDS(
            n_components=self.n_components,
            dissimilarity='precomputed',
            random_state=self.random_state,
            max_iter=300,
            n_init=4
        )
        
        recovered_coordinates = self.mds.fit_transform(distance_matrix)
        self.stress = self.mds.stress_
        
        return recovered_coordinates
    
    def transform_from_distances(self, distance_matrix: np.ndarray) -> np.ndarray:
        """Transform a precomputed distance matrix to coordinates.
        
        Args:
            distance_matrix: Precomputed distance matrix of shape (n_samples, n_samples)
        
        Returns:
            Recovered coordinates of shape (n_samples, n_components)
        """
        self.distance_matrix = distance_matrix
        
        self.mds = MDS(
            n_components=self.n_components,
            dissimilarity='precomputed',
            random_state=self.random_state,
            max_iter=300,
            n_init=4
        )
        
        recovered_coordinates = self.mds.fit_transform(distance_matrix)
        self.stress = self.mds.stress_
        
        return recovered_coordinates
    
    def align_coordinates(self, true_coords: np.ndarray, 
                         recovered_coords: np.ndarray) -> np.ndarray:
        """Align recovered coordinates to true coordinates using Procrustes.
        
        MDS can recover coordinates up to rotation, reflection, and translation.
        This method aligns the recovered coordinates to match the true ones.
        
        Args:
            true_coords: Original coordinates
            recovered_coords: Recovered coordinates from MDS
        
        Returns:
            Aligned recovered coordinates
        """
        # Center both coordinate sets
        true_centered = true_coords - true_coords.mean(axis=0)
        recovered_centered = recovered_coords - recovered_coords.mean(axis=0)
        
        # Compute optimal rotation using SVD (Procrustes)
        U, _, Vt = np.linalg.svd(recovered_centered.T @ true_centered)
        rotation = U @ Vt
        
        # Apply rotation and translation
        aligned = recovered_centered @ rotation + true_coords.mean(axis=0)
        
        return aligned
    
    def compute_reconstruction_error(self, true_coords: np.ndarray,
                                    recovered_coords: np.ndarray,
                                    align: bool = True) -> float:
        """Compute reconstruction error between true and recovered coordinates.
        
        Args:
            true_coords: Original coordinates
            recovered_coords: Recovered coordinates
            align: Whether to align coordinates before computing error (default: True)
        
        Returns:
            Root mean squared error between coordinates
        """
        if align:
            recovered_coords = self.align_coordinates(true_coords, recovered_coords)
        
        rmse = np.sqrt(np.mean((true_coords - recovered_coords) ** 2))
        return rmse
    
    def compute_distance_preservation(self, true_coords: np.ndarray,
                                     recovered_coords: np.ndarray) -> float:
        """Compute how well pairwise distances are preserved.
        
        Args:
            true_coords: Original coordinates
            recovered_coords: Recovered coordinates
        
        Returns:
            Correlation coefficient between true and recovered distances
        """
        true_distances = squareform(pdist(true_coords))
        recovered_distances = squareform(pdist(recovered_coords))
        
        # Compute correlation between distance matrices
        correlation = np.corrcoef(true_distances.flatten(), 
                                 recovered_distances.flatten())[0, 1]
        
        return correlation
    
    def plot_comparison(self, true_coords: np.ndarray, 
                       recovered_coords: np.ndarray,
                       align: bool = True,
                       figsize: Tuple[int, int] = (16, 6),
                       save_path: Optional[str] = None) -> plt.Figure:
        """Plot comparison between true and recovered coordinates.
        
        Args:
            true_coords: Original coordinates (must be 2D or will use first 2 dims)
            recovered_coords: Recovered coordinates (must be 2D or will use first 2 dims)
            align: Whether to align coordinates before plotting (default: True)
            figsize: Figure size (default: (16, 6))
            save_path: Path to save figure (optional)
        
        Returns:
            Matplotlib figure object
        """
        # Ensure we're working with 2D coordinates for plotting
        if true_coords.shape[1] > 2:
            warnings.warn(f"True coordinates have {true_coords.shape[1]} dimensions. "
                         "Using only first 2 dimensions for plotting.")
            true_coords = true_coords[:, :2]
        
        if recovered_coords.shape[1] > 2:
            warnings.warn(f"Recovered coordinates have {recovered_coords.shape[1]} dimensions. "
                         "Using only first 2 dimensions for plotting.")
            recovered_coords = recovered_coords[:, :2]
        
        if align:
            recovered_coords = self.align_coordinates(true_coords, recovered_coords)
        
        # Compute metrics
        rmse = self.compute_reconstruction_error(true_coords, recovered_coords, align=False)
        dist_corr = self.compute_distance_preservation(true_coords, recovered_coords)
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Plot true coordinates
        axes[0].scatter(true_coords[:, 0], true_coords[:, 1], 
                       c=range(len(true_coords)), cmap='viridis', 
                       s=100, alpha=0.7, edgecolors='black', linewidth=1)
        axes[0].set_title('True Coordinates', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Dimension 1', fontsize=12)
        axes[0].set_ylabel('Dimension 2', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_aspect('equal', adjustable='box')
        
        # Plot recovered coordinates
        axes[1].scatter(recovered_coords[:, 0], recovered_coords[:, 1],
                       c=range(len(recovered_coords)), cmap='viridis',
                       s=100, alpha=0.7, edgecolors='black', linewidth=1)
        axes[1].set_title(f'Recovered Coordinates\nRMSE: {rmse:.4f} | '
                         f'Distance Correlation: {dist_corr:.4f}',
                         fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Dimension 1', fontsize=12)
        axes[1].set_ylabel('Dimension 2', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_distance_comparison(self, true_coords: np.ndarray,
                                recovered_coords: np.ndarray,
                                figsize: Tuple[int, int] = (10, 8),
                                save_path: Optional[str] = None) -> plt.Figure:
        """Plot comparison of distance matrices.
        
        Args:
            true_coords: Original coordinates
            recovered_coords: Recovered coordinates
            figsize: Figure size (default: (10, 8))
            save_path: Path to save figure (optional)
        
        Returns:
            Matplotlib figure object
        """
        true_distances = squareform(pdist(true_coords))
        recovered_distances = squareform(pdist(recovered_coords))
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # True distance matrix
        im0 = axes[0, 0].imshow(squareform(true_distances), cmap='viridis')
        axes[0, 0].set_title('True Distance Matrix', fontweight='bold')
        plt.colorbar(im0, ax=axes[0, 0])
        
        # Recovered distance matrix
        im1 = axes[0, 1].imshow(squareform(recovered_distances), cmap='viridis')
        axes[0, 1].set_title('Recovered Distance Matrix', fontweight='bold')
        plt.colorbar(im1, ax=axes[0, 1])
        
        # Difference matrix
        diff = squareform(true_distances) - squareform(recovered_distances)
        im2 = axes[1, 0].imshow(diff, cmap='RdBu_r', vmin=-np.abs(diff).max(), 
                                vmax=np.abs(diff).max())
        axes[1, 0].set_title('Difference (True - Recovered)', fontweight='bold')
        plt.colorbar(im2, ax=axes[1, 0])
        
        # Scatter plot of distances
        axes[1, 1].scatter(true_distances, recovered_distances, alpha=0.5, s=10)
        axes[1, 1].plot([true_distances.min(), true_distances.max()],
                       [true_distances.min(), true_distances.max()],
                       'r--', linewidth=2, label='Perfect Recovery')
        axes[1, 1].set_xlabel('True Distances', fontsize=12)
        axes[1, 1].set_ylabel('Recovered Distances', fontsize=12)
        axes[1, 1].set_title('Distance Correlation', fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def generate_test_data(n_nodes: int = 50, true_dim: int = 2, 
                       random_state: Optional[int] = 42) -> np.ndarray:
    """Generate random test data for validation.
    
    Args:
        n_nodes: Number of nodes/points to generate (default: 50)
        true_dim: Dimensionality of the space (default: 2)
        random_state: Random seed for reproducibility (default: 42)
    
    Returns:
        Array of shape (n_nodes, true_dim) with random coordinates
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    coordinates = np.random.randn(n_nodes, true_dim)
    return coordinates


def validate_node_clustering(n_nodes: int = 50, true_dim: int = 2,
                            target_dim: Optional[int] = None,
                            random_state: int = 42,
                            plot: bool = True) -> dict:
    """Validate that node clustering can recover true coordinates from distances.
    
    This function demonstrates the core concept: given a fully connected graph
    where edges contain Euclidean distances, we can recover the original node
    coordinates (up to rotation, reflection, and translation).
    
    Args:
        n_nodes: Number of nodes in the graph (default: 50)
        true_dim: True dimensionality of the space (default: 2)
        target_dim: Target dimensionality for recovery (default: same as true_dim)
        random_state: Random seed for reproducibility (default: 42)
        plot: Whether to generate plots (default: True)
    
    Returns:
        Dictionary containing validation results and metrics
    """
    if target_dim is None:
        target_dim = true_dim
    
    # Generate random points in true_dim space
    print(f"Generating {n_nodes} random points in {true_dim}D space...")
    true_coordinates = generate_test_data(n_nodes, true_dim, random_state)
    
    # Create graph dimensionality reduction object
    gdr = GraphDimensionalityReduction(n_components=target_dim, 
                                       random_state=random_state)
    
    # Compute distance matrix
    print(f"Computing pairwise distance matrix...")
    distance_matrix = gdr.compute_distance_matrix(true_coordinates)
    print(f"Distance matrix shape: {distance_matrix.shape}")
    
    # Recover coordinates from distances
    print(f"Recovering {target_dim}D coordinates from distances using MDS...")
    recovered_coordinates = gdr.transform_from_distances(distance_matrix)
    print(f"Recovered coordinates shape: {recovered_coordinates.shape}")
    print(f"MDS stress: {gdr.stress:.4f}")
    
    # Compute metrics
    rmse = gdr.compute_reconstruction_error(true_coordinates, recovered_coordinates)
    dist_corr = gdr.compute_distance_preservation(true_coordinates, recovered_coordinates)
    
    print(f"\nValidation Results:")
    print(f"  RMSE (after alignment): {rmse:.6f}")
    print(f"  Distance correlation: {dist_corr:.6f}")
    print(f"  MDS stress: {gdr.stress:.4f}")
    
    # Generate plots if requested
    if plot and true_dim >= 2 and target_dim >= 2:
        print(f"\nGenerating comparison plots...")
        gdr.plot_comparison(true_coordinates, recovered_coordinates)
        gdr.plot_distance_comparison(true_coordinates, recovered_coordinates)
        plt.show()
    
    return {
        'true_coordinates': true_coordinates,
        'recovered_coordinates': recovered_coordinates,
        'distance_matrix': distance_matrix,
        'rmse': rmse,
        'distance_correlation': dist_corr,
        'mds_stress': gdr.stress,
        'gdr_object': gdr
    }


def dimensionality_reduction_demo(n_nodes: int = 100, 
                                 source_dim: int = 10,
                                 target_dim: int = 2,
                                 random_state: int = 42,
                                 plot: bool = True) -> dict:
    """Demonstrate graph-based dimensionality reduction.
    
    This function shows how the technique can be used for dimensionality reduction:
    taking high-dimensional data and embedding it in a lower-dimensional space
    while preserving pairwise distances.
    
    Args:
        n_nodes: Number of data points (default: 100)
        source_dim: Original dimensionality (default: 10)
        target_dim: Target dimensionality (default: 2)
        random_state: Random seed for reproducibility (default: 42)
        plot: Whether to generate plots (default: True)
    
    Returns:
        Dictionary containing results and metrics
    """
    print(f"\n{'='*60}")
    print(f"Graph Dimensionality Reduction Demo")
    print(f"{'='*60}")
    print(f"Reducing {n_nodes} points from {source_dim}D to {target_dim}D")
    print(f"{'='*60}\n")
    
    # Generate high-dimensional data
    high_dim_data = generate_test_data(n_nodes, source_dim, random_state)
    
    # Apply graph dimensionality reduction
    gdr = GraphDimensionalityReduction(n_components=target_dim, 
                                       random_state=random_state)
    low_dim_data = gdr.fit_transform(high_dim_data)
    
    # Compute distance preservation
    dist_corr = gdr.compute_distance_preservation(high_dim_data, low_dim_data)
    
    print(f"Results:")
    print(f"  Original shape: {high_dim_data.shape}")
    print(f"  Reduced shape: {low_dim_data.shape}")
    print(f"  Distance correlation: {dist_corr:.6f}")
    print(f"  MDS stress: {gdr.stress:.4f}")
    
    # Plot if target dimension is 2
    if plot and target_dim == 2:
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(low_dim_data[:, 0], low_dim_data[:, 1],
                           c=range(len(low_dim_data)), cmap='viridis',
                           s=100, alpha=0.7, edgecolors='black', linewidth=1)
        ax.set_title(f'Graph Dimensionality Reduction: {source_dim}D → {target_dim}D\n'
                    f'Distance Correlation: {dist_corr:.4f}',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Dimension 1', fontsize=12)
        ax.set_ylabel('Dimension 2', fontsize=12)
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Point Index')
        plt.tight_layout()
        plt.show()
    
    return {
        'high_dim_data': high_dim_data,
        'low_dim_data': low_dim_data,
        'distance_correlation': dist_corr,
        'mds_stress': gdr.stress,
        'gdr_object': gdr
    }


if __name__ == '__main__':
    # Example 1: Validate coordinate recovery in 2D
    print("\n" + "="*60)
    print("Example 1: Validate Node Clustering (2D → 2D)")
    print("="*60)
    results_2d = validate_node_clustering(n_nodes=50, true_dim=2, plot=True)
    
    # Example 2: Validate coordinate recovery in 3D
    print("\n" + "="*60)
    print("Example 2: Validate Node Clustering (3D → 3D)")
    print("="*60)
    results_3d = validate_node_clustering(n_nodes=50, true_dim=3, plot=False)
    
    # Example 3: Dimensionality reduction from high dimensions
    print("\n" + "="*60)
    print("Example 3: Graph Dimensionality Reduction (10D → 2D)")
    print("="*60)
    results_reduction = dimensionality_reduction_demo(
        n_nodes=100, 
        source_dim=10, 
        target_dim=2, 
        plot=True
    )
    
    # Example 4: Compare different target dimensions
    print("\n" + "="*60)
    print("Example 4: Dimensionality Reduction Comparison")
    print("="*60)
    source_dim = 20
    target_dims = [2, 3, 5, 10]
    
    print(f"\nReducing from {source_dim}D to various target dimensions:")
    print(f"{'Target Dim':<12} {'Dist. Corr.':<15} {'MDS Stress':<15}")
    print("-" * 42)
    
    for target_dim in target_dims:
        result = dimensionality_reduction_demo(
            n_nodes=100,
            source_dim=source_dim,
            target_dim=target_dim,
            plot=False
        )
        print(f"{target_dim:<12} {result['distance_correlation']:<15.6f} "
              f"{result['mds_stress']:<15.2f}")
    
    print("\n" + "="*60)
    print("All examples completed successfully!")
    print("="*60)
