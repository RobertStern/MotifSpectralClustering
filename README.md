# Motif Spectral Clustering

A Python library for higher-order network analysis and clustering using motif-based spectral methods.

## Overview

This project implements spectral clustering techniques based on network motifs. It provides tools for:

1. Creating and analyzing undirected motifs in graphs
2. Finding motif occurrences in networks
3. Creating motif adjacency matrices
4. Computing Laplacian matrices and Fiedler vectors for spectral clustering
5. Performing various clustering algorithms on the resulting embeddings

The library supports multiple clustering approaches:
- Traditional bisection with Fiedler vector
- K-means clustering
- Mini-batch K-means clustering
- Gaussian Mixture Models (GMM)
- Motif-based Approximate Personalized PageRank (MAPPR) for local clustering

## Installation

### Requirements

- Python 3.6+
- NetworkX
- NumPy
- SciPy
- Matplotlib
- pandas
- scikit-learn

### Setup

```bash
# Clone the repository
git clone https://github.com/RobertStern/MotifSpectralClustering.git
cd MotifSpectralClustering

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python scripts/motif_spectral_analysis.py --network data/examples/network.txt --motif triangle
```

This will:
1. Load the network from the specified file
2. Find all triangle motifs in the network
3. Create a motif adjacency matrix
4. Compute the Fiedler vector for spectral bisection
5. Compute conductance for the resulting partition
6. Save results to the `data/results` directory

### Running All Motifs

For a comprehensive analysis of all available motifs on a network, use the `run_all_motifs.py` script:

```bash
python scripts/run_all_motifs.py --network data/examples/network.txt
```

This will:
1. Process all available motifs (triangle, square, diamond, cycle_chord, complete4)
2. Calculate optimal k values using eigen-gap, elbow, and silhouette methods
3. Apply multiple clustering algorithms (K-means, Mini-batch K-means, GMM, MAPPR)
4. Save detailed results to CSV and a human-readable summary file

#### Batch Processing

You can also process multiple network files in a directory structure:

```bash
python scripts/run_all_motifs.py --granger-dir
```

This will find all `.edge` files in the `data/granger` directory and its subdirectories, and process each one with a unique output name based on the subfolder structure.

#### Specifying Seed Methods for MAPPR

```bash
python scripts/run_all_motifs.py --network data/examples/network.txt --seed-method first last middle
```

Available seed methods:
- `first`: Uses the first few nodes as seeds
- `last`: Uses the last few nodes as seeds
- `middle`: Uses nodes from the middle of the matrix as seeds

If no seed methods are specified, the script uses "combined" and "fiedler" methods to find good seed nodes.

#### Specifying Optimal K

```bash
python scripts/run_all_motifs.py --network data/examples/network.txt --optimal-k 5
```

This uses the specified value for k instead of calculating optimal k. The value will be used for k-means, mini-batch k-means, GMM, and as the number of seed nodes for MAPPR.

### Available Motifs

The following motifs are available:
- `triangle`: Complete graph on 3 nodes
- `star`: Star with 3 leaves (4 nodes, 3 edges)
- `square`: Cycle on 4 nodes
- `diamond`: 4-node graph with 4 edges (cycle with one chord)
- `cycle_chord`: 4-node graph with 5 edges (cycle with one chord)
- `complete4`: Complete graph on 4 nodes

### Clustering Options

#### K-means Clustering

```bash
python scripts/motif_spectral_analysis.py --network data/examples/network.txt --motif triangle --algorithm kmeans --clusters 5
```

#### Mini-batch K-means (for large networks)

```bash
python scripts/motif_spectral_analysis.py --network data/examples/network.txt --motif triangle --algorithm minibatch --clusters 5 --batch-size 100
```

#### Gaussian Mixture Models

```bash
python scripts/motif_spectral_analysis.py --network data/examples/network.txt --motif triangle --algorithm gmm --clusters 5 --covariance-type full
```

#### Automatic Cluster Number Selection with Eigengap Heuristic

```bash
python scripts/motif_spectral_analysis.py --network data/examples/network.txt --motif triangle --use-eigengap --max-k 20
```

#### Local Clustering with MAPPR

```bash
python scripts/motif_spectral_analysis.py --network data/examples/network.txt --motif triangle --algorithm mappr --seed-nodes node1 node2
```

Or automatically find good seed nodes:

```bash
python scripts/motif_spectral_analysis.py --network data/examples/network.txt --motif triangle --algorithm mappr --auto-seed --num-seeds 10
```

## Output Files

Results are saved in the `data/results/<network_name>` directory:

- Motif adjacency matrices: `motif_<motif_name>_<number>.csv`
- Fiedler vectors: `motif_<motif_name>_<number>_fiedler_vector.txt`
- Cluster assignments: `motif_<motif_name>_<network_name>_cluster_<number>_nodes.txt`
- Conductance results: `motif_<motif_name>_<network_name>_min_conductance_nodes.txt`
- MAPPR results: `motif_<motif_name>_<network_name>_mappr_cluster_<seed>_nodes.txt`

For the `run_all_motifs.py` script, results are saved in the `data/results/all_motifs` directory:

- Summary files: `<network_name>_summary.txt`
- Detailed CSV results: `<network_name>_motifs.csv`

The summary file includes:
- Network properties (nodes, edges)
- Motif occurrences and processing time
- Optimal k values found by each method
- Clustering results for each method (K-means, Mini-batch K-means, GMM, MAPPR)
- Cluster sizes and statistics

## Examples

### Example 1: Basic Spectral Bisection

```bash
python scripts/motif_spectral_analysis.py --network data/examples/karate.txt --motif triangle
```

### Example 2: Multi-way Clustering with GMM

```bash
python scripts/motif_spectral_analysis.py --network data/examples/karate.txt --motif triangle --algorithm gmm --clusters 4
```

### Example 3: Local Clustering with MAPPR

```bash
python scripts/motif_spectral_analysis.py --network data/examples/karate.txt --motif triangle --algorithm mappr --seed-nodes 0
```

### Example 4: Comprehensive Analysis of All Motifs

```bash
python scripts/run_all_motifs.py --network data/examples/karate.txt
```

### Example 5: Batch Processing of Multiple Networks

```bash
python scripts/run_all_motifs.py --granger-dir
```

## References

- Benson, A. R., Gleich, D. F., & Leskovec, J. (2016). Higher-order organization of complex networks. Science, 353(6295), 163-166.
- Spielman, D. A., & Teng, S. H. (2007). Spectral partitioning works: Planar graphs and finite element meshes. Linear Algebra and its Applications, 421(2-3), 284-305.
- Andersen, R., Chung, F., & Lang, K. (2006). Local graph partitioning using PageRank vectors. In 2006 47th Annual IEEE Symposium on Foundations of Computer Science (FOCS'06) (pp. 475-486).

## License

This project is licensed under the MIT License - see the LICENSE file for details.