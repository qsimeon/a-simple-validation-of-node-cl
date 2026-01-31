# Graph-Based Dimensionality Reduction: Node Clustering from Edge Distances

> Recover node coordinates from edge distances using graph neural networks and validate a novel graph-based dimensionality reduction approach

This Jupyter notebook explores an innovative approach to dimensionality reduction through graph theory. Given a fully connected graph with Euclidean distances as edge features but no node features, the notebook demonstrates how to recover the original node coordinates through clustering and embedding techniques. This project validates the concept of "graph dimensionality reduction" - a novel method that could complement traditional techniques like PCA or t-SNE by leveraging graph structure and edge information.

## ✨ Features

- **Node Coordinate Recovery** — Reconstructs original Euclidean coordinates of nodes using only edge distance information from a fully connected graph, demonstrating the power of graph-based learning.
- **Graph Dimensionality Reduction** — Introduces and validates a novel dimensionality reduction technique that operates on graph structures, offering an alternative to classical methods like PCA, t-SNE, or UMAP.
- **Interactive Visualizations** — Provides clear matplotlib visualizations comparing original node positions with recovered coordinates, making the learning process intuitive and engaging.
- **Educational Walkthrough** — Step-by-step notebook format with 25 cells that progressively build understanding from basic graph construction to advanced embedding techniques.

## 📦 Installation

### Prerequisites

- Python 3.7 or higher
- Jupyter Lab or Jupyter Notebook (or access to Google Colab)
- Basic understanding of graphs and linear algebra
- Familiarity with NumPy and matplotlib (helpful but not required)

### Setup

1. Clone or download this repository to your local machine
   - Get the notebook file onto your system
2. pip install numpy matplotlib scikit-learn scipy networkx
   - Install all required Python packages for graph processing, numerical computation, and visualization
3. pip install jupyter jupyterlab
   - Install Jupyter environment if you don't already have it (skip if using Google Colab)
4. jupyter lab
   - Launch Jupyter Lab in your browser to access the notebook interface
5. Open notebook.ipynb in the Jupyter Lab interface
   - Navigate to the notebook file and start exploring

## 🚀 Usage

### Running Locally with Jupyter Lab

Execute the notebook on your local machine with full control over the environment

```
# In your terminal:
jupyter lab notebook.ipynb

# Then in the Jupyter interface:
# Click 'Run' -> 'Run All Cells' to execute the entire notebook
# Or use Shift+Enter to run cells one at a time for step-by-step learning
```

**Output:**

```
The notebook will generate visualizations showing original vs. recovered node positions, distance matrices, and clustering results. You'll see plots comparing the true Euclidean coordinates with the reconstructed positions.
```

### Running on Google Colab (No Installation Required)

Use Google Colab for a zero-setup cloud-based experience, perfect for quick experimentation

```
# 1. Go to https://colab.research.google.com/
# 2. Click 'File' -> 'Upload notebook'
# 3. Upload notebook.ipynb
# 4. Run the first cell to install dependencies:
!pip install numpy matplotlib scikit-learn scipy networkx

# 5. Execute remaining cells with Runtime -> Run all
```

**Output:**

```
All visualizations and results will render directly in the Colab interface. The notebook will complete with performance metrics showing how well the algorithm recovered the original coordinates.
```

### Experimenting with Custom Parameters

Modify key parameters to explore how the algorithm behaves with different graph configurations

```
# Within the notebook, look for parameter cells like:
n_nodes = 50  # Try changing to 100 or 200
n_dimensions = 2  # Try 3D embeddings
noise_level = 0.0  # Add noise to test robustness

# Then re-run subsequent cells to see how results change
# This helps understand the algorithm's sensitivity and limitations
```

## 🏗️ Architecture

The notebook follows a progressive learning structure, starting with synthetic data generation, building a fully connected graph with edge distances, applying various embedding and clustering techniques, and finally validating the recovered coordinates against ground truth. The workflow demonstrates both the theoretical concept and practical implementation of graph-based dimensionality reduction.

### File Structure

```
Notebook Flow:

┌─────────────────────────┐
│  1. Data Generation     │
│  - Random node coords   │
│  - Euclidean distances  │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  2. Graph Construction  │
│  - Fully connected      │
│  - Edge features only   │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  3. Distance Matrix     │
│  - Adjacency matrix     │
│  - Distance encoding    │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  4. Embedding Methods   │
│  - MDS / Spectral       │
│  - Graph embeddings     │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  5. Coordinate Recovery │
│  - Node clustering      │
│  - Position estimation  │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  6. Validation          │
│  - Visual comparison    │
│  - Error metrics        │
└─────────────────────────┘
```

### Files

- **notebook.ipynb** — Main Jupyter notebook containing all code, explanations, and visualizations for the graph dimensionality reduction experiment.
- **README.md** — This documentation file providing project overview, setup instructions, and usage guidelines.

### Design Decisions

- Uses fully connected graphs to ensure all pairwise distances are available, maximizing information for coordinate recovery.
- Starts with no node features to isolate the challenge of learning from edge information alone, demonstrating pure graph-based learning.
- Employs Euclidean distances as edge features because they have clear geometric interpretation and ground truth for validation.
- Leverages classical techniques like MDS (Multidimensional Scaling) as baseline methods for comparison with novel approaches.
- Structures notebook with 25 cells to balance educational depth with manageable execution time and clear progression.
- Includes visualization at each major step to make abstract graph concepts concrete and verifiable.

## 🔧 Technical Details

### Dependencies

- **numpy** (1.21+) — Numerical computing foundation for array operations, distance calculations, and linear algebra operations.
- **matplotlib** (3.3+) — Creating visualizations of node positions, distance matrices, and comparison plots between original and recovered coordinates.
- **scikit-learn** (0.24+) — Provides MDS (Multidimensional Scaling), clustering algorithms, and distance metrics for dimensionality reduction and validation.
- **scipy** (1.7+) — Advanced scientific computing tools including optimization routines and spatial distance calculations.
- **networkx** (2.5+) — Graph creation, manipulation, and analysis tools for building and working with the fully connected graph structure.

### Key Algorithms / Patterns

- Multidimensional Scaling (MDS): Classical technique that recovers coordinates from distance matrices by preserving pairwise distances.
- Spectral embedding: Uses graph Laplacian eigendecomposition to find low-dimensional representations of graph nodes.
- Distance matrix computation: Calculates all pairwise Euclidean distances to create edge features for the fully connected graph.
- Procrustes analysis: Aligns recovered coordinates with original coordinates to measure reconstruction accuracy after accounting for rotation/reflection.

### Important Notes

- The problem is inherently ambiguous up to rotation, reflection, and translation - recovered coordinates may be geometrically equivalent but not identical.
- Fully connected graphs with n nodes have O(n²) edges, so memory usage scales quadratically with the number of nodes.
- Classical MDS provides an optimal solution for this specific problem, making it an excellent baseline for validating novel approaches.
- The notebook demonstrates that edge features alone can encode sufficient information for coordinate recovery, opening doors for graph-based dimensionality reduction.

## ❓ Troubleshooting

### ModuleNotFoundError when running cells

**Cause:** Required Python packages are not installed in the current environment or Jupyter kernel is using a different Python environment.

**Solution:** Run 'pip install numpy matplotlib scikit-learn scipy networkx' in your terminal. If using Jupyter, restart the kernel after installation. For Colab, add '!pip install' command in the first cell.

### Visualizations not displaying in notebook

**Cause:** Matplotlib backend not configured for inline display in Jupyter notebooks.

**Solution:** Add '%matplotlib inline' at the top of the notebook (usually already included). If using Jupyter Lab, try '%matplotlib widget' for interactive plots.

### Memory error with large graphs

**Cause:** Fully connected graphs require O(n²) memory for distance matrices. Large values of n_nodes can exhaust available RAM.

**Solution:** Reduce the number of nodes (n_nodes parameter) to 100 or fewer for initial experiments. For larger graphs, consider sparse representations or batch processing.

### Recovered coordinates look different from originals

**Cause:** MDS and embedding methods recover coordinates up to rotation, reflection, and translation - geometric shape is preserved but orientation may differ.

**Solution:** This is expected behavior. Use Procrustes analysis (included in notebook) to align coordinates, or compare distance matrices rather than raw coordinates to validate accuracy.

### Jupyter Lab won't start or shows connection error

**Cause:** Port conflict, firewall blocking, or Jupyter not properly installed in the current Python environment.

**Solution:** Try 'jupyter lab --port=8889' to use a different port. Ensure Jupyter is installed with 'pip install jupyterlab'. Check firewall settings or try 'jupyter notebook' as alternative.

---

This README was generated to make graph-based machine learning concepts accessible to learners at all levels. The notebook serves as both a validation of theoretical concepts and a practical introduction to graph neural networks and dimensionality reduction. Experiment with parameters, visualize results, and consider how this approach might extend to real-world problems like molecule structure prediction, social network analysis, or sensor network localization. The techniques demonstrated here bridge classical linear algebra (MDS) with modern graph learning, showing how fundamental mathematical principles remain relevant in contemporary machine learning research.