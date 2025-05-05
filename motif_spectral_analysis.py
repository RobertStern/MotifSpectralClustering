#!/usr/bin/env python3
"""
Motif Spectral Analysis

This module combines functionality for motif analysis and spectral clustering.
It provides tools for:
1. Creating, saving, and visualizing undirected motifs
2. Finding motif occurrences in graphs
3. Creating motif adjacency matrices
4. Computing Laplacian matrices and Fiedler vectors for spectral clustering
5. Performing k-means clustering on eigenvectors for multi-cluster spectral clustering
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import os
import time
import itertools
from typing import Dict, List, Tuple, Optional, Union, Set, Any, Hashable
import pandas as pd
from numpy import ndarray, dtype
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh
from numpy.linalg import eigh
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.mixture import GaussianMixture
from BonusFunctions import approximate_pagerank, sweep_cut, mappr

# Import at the top level to avoid circular imports
if 'load_network_from_file' not in globals():
    from network_loader import load_network_from_file


class UndirectedMotifs:
    """
    Class for handling undirected motifs in graphs.

    This class provides functionality for:
    - Creating and managing motif structures
    - Computing motif adjacency matrices
    - Spectral analysis of motif-based graphs
    - Clustering using various algorithms:
        - Traditional bisection with Fiedler vector
        - K-means clustering
        - Mini-batch K-means clustering (more efficient for large datasets)
    """

    def __init__(self):
        """Initialize the UndirectedMotifs class."""
        self.motifs = {}  # Dictionary to store motifs: {name: graph}
        self.motif_dir = os.path.join("data", "motifs")
        self.output_file = ""

        # Create directory for saved motifs if it doesn't exist
        if not os.path.exists(self.motif_dir):
            os.makedirs(self.motif_dir)

        # Create basic motifs
        # Motif 1: Two edges (path)
        path = nx.Graph()
        path.add_nodes_from([0, 1, 2])
        path.add_edge(0, 1)
        path.add_edge(1, 2)
        self.motifs["path"] = path

        # Motif 2: Three edges (triangle/complete graph)
        triangle = nx.Graph()
        triangle.add_nodes_from([0, 1, 2])
        triangle.add_edge(0, 1)
        triangle.add_edge(1, 2)
        triangle.add_edge(0, 2)
        self.motifs["triangle"] = triangle

        # Motif 3: Three edges (path of length 3)
        path3 = nx.Graph()
        path3.add_nodes_from([0, 1, 2, 3])
        path3.add_edge(0, 1)
        path3.add_edge(1, 2)
        path3.add_edge(2, 3)
        self.motifs["path3"] = path3

        # Motif 4: Three edges (star)
        star = nx.Graph()
        star.add_nodes_from([0, 1, 2, 3])
        star.add_edge(0, 1)
        star.add_edge(0, 2)
        star.add_edge(0, 3)
        self.motifs["star"] = star

        # Motif 5: Four edges (cycle/square)
        square = nx.Graph()
        square.add_nodes_from([0, 1, 2, 3])
        square.add_edge(0, 1)
        square.add_edge(1, 2)
        square.add_edge(2, 3)
        square.add_edge(3, 0)
        self.motifs["square"] = square

        # Motif 6: Four edges (diamond)
        diamond = nx.Graph()
        diamond.add_nodes_from([0, 1, 2, 3])
        diamond.add_edge(0, 1)
        diamond.add_edge(1, 2)
        diamond.add_edge(1, 3)
        diamond.add_edge(2, 3)
        self.motifs["diamond"] = diamond

        # Motif 7: Five edges (cycle with chord)
        cycle_chord = nx.Graph()
        cycle_chord.add_nodes_from([0, 1, 2, 3])
        cycle_chord.add_edge(0, 1)
        cycle_chord.add_edge(1, 2)
        cycle_chord.add_edge(2, 3)
        cycle_chord.add_edge(3, 0)
        cycle_chord.add_edge(0, 2)
        self.motifs["cycle_chord"] = cycle_chord

        # Motif 8: Six edges (complete graph)
        complete4 = nx.Graph()
        complete4.add_nodes_from([0, 1, 2, 3])
        complete4.add_edge(0, 1)
        complete4.add_edge(0, 2)
        complete4.add_edge(0, 3)
        complete4.add_edge(1, 2)
        complete4.add_edge(1, 3)
        complete4.add_edge(2, 3)
        self.motifs["complete4"] = complete4

    def list_motifs(self) -> List[str]:
        """Return a list of all available motif names."""
        return list(self.motifs.keys())

    def get_motif(self, name: str) -> nx.Graph:
        """
        Get a motif by name.

        Args:
            name: Name of the motif to get

        Returns:
            The graph object for the requested motif

        Raises:
            ValueError: If the motif does not exist
        """
        if name not in self.motifs:
            raise ValueError(f"Motif '{name}' does not exist")
        return self.motifs[name]

    def compute_conductance(self, adjacency, fiedler, node_mapping, network_name, motif_name):
        # Sort nodes by Fiedler vector
        sorted_idx = np.argsort(fiedler)
        n = len(sorted_idx)
        # Check if adjacency is sparse
        is_sparse = sp.issparse(adjacency)

        # Precompute node volumes (sum of row/column for each node)
        print("Precomputing node volumes...")
        if is_sparse:
            # For sparse matrices, compute row sums efficiently
            node_volumes = np.array(adjacency.sum(axis=1)).flatten()
        else:
            # For dense matrices, use numpy's sum
            node_volumes = adjacency.sum(axis=1)

        # Total volume
        vol_total = node_volumes.sum()

        # Calculate motif conductance for each prefix
        print("Computing conductance for each prefix...")
        results = []

        # Create arrays for efficient indexing
        sorted_idx_array = np.array(sorted_idx)

        # Initialize volumes for S and T
        vol_S = 0
        vol_T = vol_total

        # For efficient cut computation, we'll use a different approach based on matrix structure
        if is_sparse:
            # For sparse matrices, we'll compute cuts incrementally
            # This avoids materializing large submatrices
            for k in range(1, n):
                # Update node moving from T to S
                node_idx = sorted_idx[k-1]
                vol_S += node_volumes[node_idx]
                vol_T -= node_volumes[node_idx]

                # Compute cut efficiently for sparse matrices
                # Get the nodes in S and T
                S_nodes = sorted_idx_array[:k]
                T_nodes = sorted_idx_array[k:]

                # Compute cut by summing specific elements
                # This is more efficient than materializing the full submatrix
                cut = 0
                for i in S_nodes:
                    # Get the row for node i
                    row = adjacency[i].toarray().flatten()
                    # Sum connections to nodes in T
                    cut += sum(row[j] for j in T_nodes)

                # Compute conductance
                phi = cut / min(vol_S, vol_T) if min(vol_S, vol_T) > 0 else 0

                # Store results
                results.append({
                    'k': k,
                    'prefix_nodes': list(S_nodes),
                    'cut': cut,
                    'vol_S': vol_S,
                    'vol_T': vol_T,
                    'conductance': phi
                })

                # Print progress for large networks
                if n > 1000 and k % 100 == 0:
                    print(f"Processed {k}/{n-1} prefixes ({k/(n-1)*100:.1f}%)...")
        else:
            # For dense matrices, we can use more vectorized operations
            # Precompute the sorted adjacency matrix for efficient slicing
            sorted_adjacency = adjacency[sorted_idx][:, sorted_idx]

            for k in range(1, n):
                # Update volumes incrementally
                node_idx = sorted_idx[k-1]
                vol_S += node_volumes[node_idx]
                vol_T -= node_volumes[node_idx]

                # Compute cut using slicing operations
                # This is much faster than nested loops
                S_slice = slice(0, k)
                T_slice = slice(k, n)
                cut = np.sum(sorted_adjacency[S_slice, T_slice])

                # Compute conductance
                phi = cut / min(vol_S, vol_T) if min(vol_S, vol_T) > 0 else 0

                # Store results
                results.append({
                    'k': k,
                    'prefix_nodes': list(sorted_idx[:k]),
                    'cut': cut,
                    'vol_S': vol_S,
                    'vol_T': vol_T,
                    'conductance': phi
                })

                # Print progress for large networks
                if n > 1000 and k % 100 == 0:
                    print(f"Processed {k}/{n-1} prefixes ({k/(n-1)*100:.1f}%)...")

        # Create DataFrame and print results
        df = pd.DataFrame(results)
        print("Conductance computation complete.")

        # Only print the adjacency matrix for small networks to avoid flooding the console
        if n <= 20:
            print(adjacency)
            print(f'Fiedler: {np.round(fiedler,4)}')
            print(df.to_string(index=False))
        else:
            print(f"Network too large to print ({n} nodes). Showing only conductance summary.")
            # Print summary statistics
            min_conductance = df['conductance'].min()
            min_idx = df['conductance'].idxmin()
            min_k = df.loc[min_idx, 'k']
            print(f"Minimum conductance: {min_conductance} at k={min_k}")

            # Print first few and last few rows
            print("\nFirst 5 rows:")
            print(df.head().to_string(index=False))
            print("\nLast 5 rows:")
            print(df.tail().to_string(index=False))
        self.save_conductance(df, network_name, motif_name, node_mapping)

    def compute_multiway_conductance(self, adjacency, clusters, node_mapping, network_name, motif_name):
        """
        Compute motif-conductance for multi-way clustering.

        For each cluster, compute the conductance as the ratio of the cut between the cluster
        and the rest of the graph to the minimum of the volume of the cluster and the volume
        of the rest of the graph.

        Args:
            adjacency: Adjacency matrix
            clusters: List of lists, where each inner list contains the node indices for a cluster
            node_mapping: Mapping from indices to node names
            network_name: Name of the network
            motif_name: Name of the motif

        Returns:
            DataFrame with conductance results for each cluster
        """
        # Check if adjacency is sparse
        is_sparse = sp.issparse(adjacency)

        # Precompute node volumes (sum of row/column for each node)
        print("Precomputing node volumes...")
        if is_sparse:
            # For sparse matrices, compute row sums efficiently
            node_volumes = np.array(adjacency.sum(axis=1)).flatten()
        else:
            # For dense matrices, use numpy's sum
            node_volumes = adjacency.sum(axis=1)

        # Total volume
        vol_total = node_volumes.sum()

        # Calculate motif conductance for each cluster
        print("Computing conductance for each cluster...")
        results = []

        for i, cluster in enumerate(clusters):
            # Skip empty clusters
            if not cluster:
                continue

            # Calculate volume of the cluster
            vol_cluster = sum(node_volumes[node_idx] for node_idx in cluster)

            # Calculate volume of the complement
            vol_complement = vol_total - vol_cluster

            # Calculate cut between the cluster and the rest of the graph
            cut = 0

            # For sparse matrices, compute cut efficiently
            if is_sparse:
                for node_idx in cluster:
                    # Get the row for node_idx
                    row = adjacency[node_idx].toarray().flatten()
                    # Sum connections to nodes outside the cluster
                    for j in range(len(row)):
                        if j not in cluster and row[j] > 0:
                            cut += row[j]
            else:
                # For dense matrices, use vectorized operations
                # Create a mask for the cluster
                cluster_mask = np.zeros(adjacency.shape[0], dtype=bool)
                cluster_mask[cluster] = True

                # Compute cut using the mask
                for node_idx in cluster:
                    cut += np.sum(adjacency[node_idx, ~cluster_mask])

            # Compute conductance
            phi = cut / min(vol_cluster, vol_complement) if min(vol_cluster, vol_complement) > 0 else 0

            # Store results
            results.append({
                'cluster': i + 1,
                'size': len(cluster),
                'cut': cut,
                'vol_cluster': vol_cluster,
                'vol_complement': vol_complement,
                'conductance': phi
            })

        # Create DataFrame and print results
        df = pd.DataFrame(results)
        print("Multi-way conductance computation complete.")

        # Print the results
        if df.empty:
            print("No valid clusters found.")
        else:
            print(df.to_string(index=False))

            # Print summary statistics
            max_conductance = df['conductance'].max()
            max_idx = df['conductance'].idxmax()
            max_cluster = df.loc[max_idx, 'cluster']
            print(f"\nMaximum conductance: {max_conductance} for cluster {max_cluster}")

            min_conductance = df['conductance'].min()
            min_idx = df['conductance'].idxmin()
            min_cluster = df.loc[min_idx, 'cluster']
            print(f"Minimum conductance: {min_conductance} for cluster {min_cluster}")

            avg_conductance = df['conductance'].mean()
            print(f"Average conductance: {avg_conductance}")

        # Save the results
        self.save_multiway_conductance(df, network_name, motif_name, node_mapping, clusters)



    def create_motif_adjacency_matrix(self, G: nx.Graph, motif_name: str, network_file: Optional[str] = None) -> lil_matrix | ndarray[tuple[int, int], dtype[Any]] | tuple[ndarray[tuple[int, int], dtype[Any]], dict[int, Hashable | Any]]:
        # Get the motif
        motif = self.get_motif(motif_name)

        print(f"Finding occurrences of motif '{motif_name}' in the graph...")
        start_time = time.time()
        # Find all occurrences of the motif in the graph
        motif_occurrences = self._find_motif_occurrences(G, motif)
        elapsed_time = time.time() - start_time
        print(f"Found {len(motif_occurrences)} occurrences in {elapsed_time:.2f} seconds")

        # Create the adjacency matrix
        print("Creating motif adjacency matrix...")
        start_time = time.time()
        # Initialize the adjacency matrix with zeros
        n = G.number_of_nodes()
        node_to_idx = {node: i for i, node in enumerate(G.nodes())}
        idx_to_node = {i: node for node, i in node_to_idx.items()}
        # Import scipy.sparse for potential use with large networks
        import scipy.sparse as sp
        # Use sparse matrix for large networks to save memory
        if n > 1000:
            print("Using sparse matrix representation for large network...")
            adj_matrix = sp.lil_matrix((n, n), dtype=int)
        else:
            adj_matrix = np.zeros((n, n), dtype=int)
        # Process occurrences in batches to avoid memory issues
        batch_size = 10000
        total_occurrences = len(motif_occurrences)

        if total_occurrences == 0:
            print("No occurrences of the motif found in the graph.")
            return adj_matrix, idx_to_node

        for i in range(0, total_occurrences, batch_size):
            batch_end = min(i + batch_size, total_occurrences)
            batch = motif_occurrences[i:batch_end]
            # Print progress
            print(f"Processing occurrences {i+1}-{batch_end} of {total_occurrences} ({(batch_end/total_occurrences)*100:.1f}%)...")
            # Fill the adjacency matrix for this batch
            for occurrence in batch:
                # For each pair of nodes in the occurrence, increment the count
                for u, v in itertools.combinations(occurrence, 2):
                    u_idx = node_to_idx[u]
                    v_idx = node_to_idx[v]
                    adj_matrix[u_idx, v_idx] += 1
                    adj_matrix[v_idx, u_idx] += 1  # Symmetric for undirected graph

        # Convert sparse matrix to dense if needed for output
        if isinstance(adj_matrix, sp.spmatrix):
            print("Converting sparse matrix to dense format for output...")
            adj_matrix = adj_matrix.toarray()

        elapsed_time = time.time() - start_time
        print(f"Adjacency matrix created in {elapsed_time:.2f} seconds")
        self.save_adjacency_matrix(adj_matrix, motif_name, motif_occurrences, node_to_idx, batch_size)
        # Convert adjacency matrix to float for better numerical stability
        adj_matrix_float = adj_matrix.astype(float)

        return adj_matrix_float, idx_to_node

    def save_edges_within_set(self, G: nx.Graph, node_set: List[int], node_mapping: Optional[dict], 
                           output_dir: str, motif_name: str, network_name: str, 
                           set_label: str, set_type: str) -> None:
        """
        Save edges within a set of nodes to a file.

        Args:
            G: The original graph
            node_set: List of node indices in the set
            node_mapping: Optional mapping from node indices to node names
            output_dir: Directory to save the file
            motif_name: Name of the motif
            network_name: Name of the network
            set_label: Label for the set (e.g., "S" or "T")
            set_type: Type of the set (e.g., "fiedler_negative", "min_conductance")
        """
        # Create the filename
        edges_file = os.path.join(output_dir, f"motif_{motif_name}_{network_name}_{set_type}_edges.txt")

        try:
            # Convert node indices to actual node names if mapping is available
            node_set_names = [node_mapping[idx] if node_mapping else idx for idx in node_set]

            # Create a subgraph with only the nodes in the set
            subgraph = G.subgraph(node_set_names)

            # Save the edges to a file
            with open(edges_file, 'w') as f:
                f.write(f"# Edges within set {set_label} ({set_type})\n")
                f.write(f"# Motif: {motif_name}\n")
                f.write(f"# Network: {network_name}\n")
                f.write(f"# Number of nodes: {len(node_set)}\n")
                f.write(f"# Number of edges: {subgraph.number_of_edges()}\n\n")

                # Write each edge on a separate line
                for u, v in subgraph.edges():
                    f.write(f"{u} {v}\n")

            print(f"Edges within set {set_label} saved to {edges_file}")

        except Exception as e:
            print(f"Error saving edges within set {set_label}: {e}")

    def compute_fiedler(self, adj_matrix, idx_to_node, network_file, motif_name):
        """
        Compute the Fiedler vector of the normalized Laplacian of W,
        with isolates removed to avoid divide-by-zero.
        Returns a full-length vector with zeros for removed nodes.
        """
        # Compute degree
        deg = np.array(adj_matrix.sum(axis=1)).ravel()
        # Identify non-isolated nodes
        mask = deg > 0
        idx_noniso = np.where(mask)[0]
        if idx_noniso.size == 0:
            raise ValueError("All nodes are isolated; cannot compute Fiedler vector.")
        # Submatrix for non-isolated
        W_sub = adj_matrix[idx_noniso][:, idx_noniso]
        deg_sub = deg[idx_noniso]
        # Build D^{-1/2}
        inv_s = 1.0 / np.sqrt(deg_sub)
        D_inv_s = np.diag(inv_s)
        # Compute normalized Laplacian on subgraph
        with np.errstate(divide='ignore', invalid='ignore'):
            L_sub = np.eye(len(inv_s)) - D_inv_s @ (
                W_sub.toarray() if isinstance(W_sub, csr_matrix) else W_sub) @ D_inv_s
        # Eigen-decomposition
        eigvals, eigvecs = eigh(L_sub)
        fiedler_sub = eigvecs[:, 1]
        # Build full-length vector (zeros for isolated)
        fiedler = np.zeros_like(deg, dtype=float)
        fiedler[idx_noniso] = fiedler_sub
        self.save_fiedler_vector(fiedler, motif_name, idx_to_node)
        # Create and save sets based on Fiedler vector sign
        self.create_fiedler_sets(fiedler, idx_to_node, network_file, motif_name)
        return fiedler

    def compute_eigenvectors(self, adj_matrix, k):
        """
        Compute the first k eigenvectors of the normalized Laplacian of W,
        with isolates removed to avoid divide-by-zero.

        Args:
            adj_matrix: Adjacency matrix
            k: Number of eigenvectors to compute (including the trivial one)

        Returns:
            eigvals: Eigenvalues
            eigvecs: Eigenvectors
            idx_noniso: Indices of non-isolated nodes
            mask: Boolean mask for non-isolated nodes
        """
        # Compute degree
        deg = np.array(adj_matrix.sum(axis=1)).ravel()
        # Identify non-isolated nodes
        mask = deg > 0
        idx_noniso = np.where(mask)[0]
        if idx_noniso.size == 0:
            raise ValueError("All nodes are isolated; cannot compute eigenvectors.")
        # Submatrix for non-isolated
        W_sub = adj_matrix[idx_noniso][:, idx_noniso]
        deg_sub = deg[idx_noniso]
        # Build D^{-1/2}
        inv_s = 1.0 / np.sqrt(deg_sub)
        D_inv_s = np.diag(inv_s)
        # Compute normalized Laplacian on subgraph
        with np.errstate(divide='ignore', invalid='ignore'):
            L_sub = np.eye(len(inv_s)) - D_inv_s @ (
                W_sub.toarray() if isinstance(W_sub, csr_matrix) else W_sub) @ D_inv_s
        # Eigen-decomposition
        eigvals, eigvecs = eigh(L_sub)

        # Make sure we don't request more eigenvectors than available
        k = min(k, eigvecs.shape[1])

        return eigvals[:k], eigvecs[:, :k], idx_noniso, mask

    def compute_optimal_k_eigengap(self, adj_matrix, max_k=20):
        """
        Compute the optimal number of clusters using the eigengap heuristic.

        The eigengap heuristic suggests that the number of clusters should be
        determined by the largest gap in the eigenvalue sequence of the normalized
        Laplacian matrix.

        Args:
            adj_matrix: Adjacency matrix
            max_k: Maximum number of clusters to consider

        Returns:
            optimal_k: Optimal number of clusters
            eigvals: Eigenvalues
            eigvecs: Eigenvectors
            idx_noniso: Indices of non-isolated nodes
            mask: Boolean mask for non-isolated nodes
        """
        # Compute eigenvectors for max_k clusters
        eigvals, eigvecs, idx_noniso, mask = self.compute_eigenvectors(adj_matrix, max_k + 1)

        # Compute the gaps between consecutive eigenvalues
        # Skip the first eigenvalue (which is always 0 for connected graphs)
        gaps = np.diff(eigvals)

        # Find the index of the largest gap
        # Add 1 because we're looking at gaps between eigenvalues
        # Add another 1 because we skipped the first eigenvalue
        optimal_k = np.argmax(gaps) + 1

        print(f"Eigenvalues: {eigvals}")
        print(f"Eigengaps: {gaps}")
        print(f"Optimal number of clusters (eigengap heuristic): {optimal_k}")

        return optimal_k, eigvals, eigvecs, idx_noniso, mask

    def compute_kmeans_clusters(self, adj_matrix, idx_to_node, network_file, motif_name, k):
        """
        Perform k-means clustering on the first k eigenvectors of the normalized Laplacian.

        Args:
            adj_matrix: Adjacency matrix
            idx_to_node: Mapping from indices to node names
            network_file: Path to the network file
            motif_name: Name of the motif
            k: Number of clusters

        Returns:
            clusters: List of lists, where each inner list contains the node indices for a cluster
        """
        from sklearn.cluster import KMeans, MiniBatchKMeans

        # Compute eigenvectors
        eigvals, eigvecs, idx_noniso, mask = self.compute_eigenvectors(adj_matrix, k)

        # Use eigenvectors starting from the second one (skip the trivial eigenvector)
        # The first non-trivial eigenvector is the Fiedler vector
        X = eigvecs[:, 1:k]

        # Perform k-means clustering
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)

        # Create full-length cluster labels (with -1 for isolated nodes)
        full_labels = np.full(adj_matrix.shape[0], -1)
        full_labels[idx_noniso] = cluster_labels

        # Group nodes by cluster
        clusters = [[] for _ in range(k)]
        for i, label in enumerate(full_labels):
            if label != -1:  # Skip isolated nodes
                clusters[label].append(i)

        # Save cluster assignments to files
        self.save_kmeans_clusters(clusters, idx_to_node, network_file, motif_name)

        return clusters

    def compute_minibatch_kmeans_clusters(self, adj_matrix, idx_to_node, network_file, motif_name, k, batch_size=100, max_iter=100, reassignment_ratio=0.01):
        """
        Perform mini-batch k-means clustering on the first k eigenvectors of the normalized Laplacian.
        Mini-batch k-means is more efficient for large datasets than standard k-means.

        Args:
            adj_matrix: Adjacency matrix
            idx_to_node: Mapping from indices to node names
            network_file: Path to the network file
            motif_name: Name of the motif
            k: Number of clusters
            batch_size: Size of the mini-batches
            max_iter: Maximum number of iterations
            reassignment_ratio: Control the fraction of the maximum number of counts for a center to be reassigned

        Returns:
            clusters: List of lists, where each inner list contains the node indices for a cluster
        """
        from sklearn.cluster import KMeans, MiniBatchKMeans

        # Compute eigenvectors
        eigvals, eigvecs, idx_noniso, mask = self.compute_eigenvectors(adj_matrix, k)

        # Use eigenvectors starting from the second one (skip the trivial eigenvector)
        # The first non-trivial eigenvector is the Fiedler vector
        X = eigvecs[:, 1:k]

        # Perform mini-batch k-means clustering
        minibatch_kmeans = MiniBatchKMeans(
            n_clusters=k, 
            batch_size=batch_size,
            max_iter=max_iter,
            random_state=42,
            reassignment_ratio=reassignment_ratio
        )
        cluster_labels = minibatch_kmeans.fit_predict(X)

        # Create full-length cluster labels (with -1 for isolated nodes)
        full_labels = np.full(adj_matrix.shape[0], -1)
        full_labels[idx_noniso] = cluster_labels

        # Group nodes by cluster
        clusters = [[] for _ in range(k)]
        for i, label in enumerate(full_labels):
            if label != -1:  # Skip isolated nodes
                clusters[label].append(i)

        # Save cluster assignments to files
        self.save_kmeans_clusters(clusters, idx_to_node, network_file, motif_name)

        return clusters

    def compute_gmm_clusters(self, adj_matrix, idx_to_node, network_file, motif_name, k, covariance_type='full', n_init=10, max_iter=100):
        """
        Perform Gaussian Mixture Model clustering on the first k eigenvectors of the normalized Laplacian.

        Args:
            adj_matrix: Adjacency matrix
            idx_to_node: Mapping from indices to node names
            network_file: Path to the network file
            motif_name: Name of the motif
            k: Number of clusters
            covariance_type: Type of covariance parameters to use ('full', 'tied', 'diag', 'spherical')
            n_init: Number of initializations to perform
            max_iter: Maximum number of iterations

        Returns:
            clusters: List of lists, where each inner list contains the node indices for a cluster
        """
        # Compute eigenvectors
        eigvals, eigvecs, idx_noniso, mask = self.compute_eigenvectors(adj_matrix, k)

        # Use eigenvectors starting from the second one (skip the trivial eigenvector)
        # The first non-trivial eigenvector is the Fiedler vector
        X = eigvecs[:, 1:k]

        # Perform GMM clustering
        gmm = GaussianMixture(
            n_components=k,
            covariance_type=covariance_type,
            n_init=n_init,
            max_iter=max_iter,
            random_state=42
        )
        cluster_labels = gmm.fit_predict(X)

        # Create full-length cluster labels (with -1 for isolated nodes)
        full_labels = np.full(adj_matrix.shape[0], -1)
        full_labels[idx_noniso] = cluster_labels

        # Group nodes by cluster
        clusters = [[] for _ in range(k)]
        for i, label in enumerate(full_labels):
            if label != -1:  # Skip isolated nodes
                clusters[label].append(i)

        # Save cluster assignments to files
        self.save_gmm_clusters(clusters, idx_to_node, network_file, motif_name)

        return clusters

    def compute_mappr_cluster(self, adj_matrix, idx_to_node, network_file, motif_name, seed_nodes, alpha=0.15, epsilon=1e-6):
        """
        Perform Motif-based Approximate Personalized PageRank (MAPPR) for local clustering.

        Args:
            adj_matrix: Adjacency matrix
            idx_to_node: Mapping from indices to node names
            network_file: Path to the network file
            motif_name: Name of the motif
            seed_nodes: Index or list of indices of the seed node(s)
            alpha: Teleportation parameter (default: 0.15)
            epsilon: Approximation parameter (default: 1e-6)

        Returns:
            clusters: Dictionary mapping seed nodes to their respective clusters
        """
        # Convert single seed node to list if necessary
        if not isinstance(seed_nodes, list):
            seed_nodes = [seed_nodes]

        print(f"Performing MAPPR local clustering with seed nodes {seed_nodes}...")
        print(f"Parameters: alpha={alpha}, epsilon={epsilon}")

        # Create a reverse mapping from node names to indices
        node_to_idx = {node: idx for idx, node in idx_to_node.items()} if idx_to_node is not None else {}

        # Convert seed nodes from names to indices if necessary
        seed_node_indices = []
        for seed_node in seed_nodes:
            if isinstance(seed_node, str) and idx_to_node is not None:
                if seed_node in node_to_idx:
                    seed_node_idx = node_to_idx[seed_node]
                    seed_node_indices.append(seed_node_idx)
                    print(f"Converted seed node name '{seed_node}' to index {seed_node_idx}")
                else:
                    raise ValueError(f"Seed node '{seed_node}' not found in the graph")
            else:
                seed_node_indices.append(seed_node)

        # Perform MAPPR for each seed node individually
        clusters = {}
        pageranks = {}
        conductances = {}

        for i, seed_node_idx in enumerate(seed_node_indices):
            print(f"Computing cluster for seed node {seed_node_idx} ({i+1}/{len(seed_node_indices)})...")
            cluster, conductance, p = mappr(adj_matrix, [seed_node_idx], alpha, epsilon)

            # Store the results
            seed_node_name = seed_nodes[i]
            clusters[seed_node_name] = cluster
            conductances[seed_node_name] = conductance
            pageranks[seed_node_name] = p

            print(f"Found cluster for seed node {seed_node_name} with {len(cluster)} nodes and conductance {conductance}")

        # Save clusters to files
        self.save_mappr_cluster(clusters, conductances, pageranks, idx_to_node, network_file, motif_name, seed_nodes)

        return clusters

    def save_mappr_cluster(self, clusters, conductances, pageranks, node_mapping, network_file, motif_name, seed_nodes):
        """
        Save MAPPR clusters to files.

        Args:
            clusters: Dictionary mapping seed nodes to their respective clusters
            conductances: Dictionary mapping seed nodes to their respective conductance values
            pageranks: Dictionary mapping seed nodes to their respective PageRank vectors
            node_mapping: Mapping from indices to node names
            network_file: Path to the network file
            motif_name: Name of the motif
            seed_nodes: List of seed nodes
        """
        # Extract the base filename without directory and extension
        network_basename = os.path.basename(network_file)
        network_name, _ = os.path.splitext(network_basename)

        # Create output directory
        output_dir = os.path.join("data/results", network_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Load the original graph for saving edges
        G = load_network_from_file(network_file)

        # Save each cluster separately
        for seed_node, cluster in clusters.items():
            # Format seed node for display and filenames
            seed_node_str = str(seed_node)
            conductance = conductances[seed_node]
            pagerank = pageranks[seed_node]

            # Create a file for the cluster
            cluster_file = os.path.join(output_dir, f"motif_{motif_name}_{network_name}_mappr_cluster_{seed_node_str}_nodes.txt")

            try:
                # Save the cluster to a file
                with open(cluster_file, 'w') as f:
                    f.write(f"# MAPPR local clustering results\n")
                    f.write(f"# Motif: {motif_name}\n")
                    f.write(f"# Network: {network_name}\n")
                    f.write(f"# Seed node: {seed_node_str}\n")
                    f.write(f"# Conductance: {conductance}\n")
                    f.write(f"# Number of nodes: {len(cluster)}\n\n")

                    # Write each node on a separate line with node name if mapping is available
                    for node_idx in sorted(cluster):
                        node_name = node_mapping[node_idx] if node_mapping else node_idx
                        f.write(f"{node_name}\n")

                print(f"MAPPR cluster for seed node {seed_node_str} saved to {cluster_file}")

                # Save PageRank vector to a file
                pagerank_file = os.path.join(output_dir, f"motif_{motif_name}_{network_name}_mappr_pagerank_{seed_node_str}.txt")

                with open(pagerank_file, 'w') as f:
                    f.write(f"# MAPPR PageRank vector\n")
                    f.write(f"# Motif: {motif_name}\n")
                    f.write(f"# Network: {network_name}\n")
                    f.write(f"# Seed node: {seed_node_str}\n")
                    f.write(f"# Format: node_name pagerank_value\n\n")

                    # Sort nodes by PageRank value (descending)
                    sorted_nodes = sorted(range(len(pagerank)), key=lambda x: -pagerank[x])

                    for node_idx in sorted_nodes:
                        node_name = node_mapping[node_idx] if node_mapping else node_idx
                        f.write(f"{node_name} {pagerank[node_idx]}\n")

                print(f"MAPPR PageRank vector for seed node {seed_node_str} saved to {pagerank_file}")

                # Save edges within the cluster
                self.save_edges_within_set(G, cluster, node_mapping, output_dir, motif_name, network_name, 
                                          f"MAPPR Cluster for {seed_node_str}", f"mappr_cluster_{seed_node_str}")

            except Exception as e:
                print(f"Error saving MAPPR cluster for seed node {seed_node_str}: {e}")

        # Also save a summary file with all clusters
        summary_file = os.path.join(output_dir, f"motif_{motif_name}_{network_name}_mappr_clusters_summary.txt")
        try:
            with open(summary_file, 'w') as f:
                f.write(f"# MAPPR local clustering summary\n")
                f.write(f"# Motif: {motif_name}\n")
                f.write(f"# Network: {network_name}\n")
                f.write(f"# Number of seed nodes: {len(seed_nodes)}\n\n")

                f.write("# Seed node, Cluster size, Conductance\n")
                for seed_node in seed_nodes:
                    cluster = clusters[seed_node]
                    conductance = conductances[seed_node]
                    f.write(f"{seed_node}, {len(cluster)}, {conductance}\n")

            print(f"MAPPR clusters summary saved to {summary_file}")

        except Exception as e:
            print(f"Error saving MAPPR clusters summary: {e}")

    def save_kmeans_clusters(self, clusters, node_mapping, network_file, motif_name):
        """
        Save k-means cluster assignments to files.

        Args:
            clusters: List of lists, where each inner list contains the node indices for a cluster
            node_mapping: Mapping from indices to node names
            network_file: Path to the network file
            motif_name: Name of the motif
        """
        # Extract the base filename without directory and extension
        network_basename = os.path.basename(network_file)
        network_name, _ = os.path.splitext(network_basename)

        # Create output directory
        output_dir = os.path.join("data/results", network_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Create a file for all clusters
        all_clusters_file = os.path.join(output_dir, f"motif_{motif_name}_{network_name}_kmeans_clusters.txt")

        try:
            # Save all clusters to a single file
            with open(all_clusters_file, 'w') as f:
                f.write(f"# K-means clustering results\n")
                f.write(f"# Motif: {motif_name}\n")
                f.write(f"# Network: {network_name}\n")
                f.write(f"# Number of clusters: {len(clusters)}\n\n")

                for i, cluster in enumerate(clusters):
                    f.write(f"# Cluster {i+1} ({len(cluster)} nodes):\n")
                    for node_idx in sorted(cluster):
                        node_name = node_mapping[node_idx] if node_mapping else node_idx
                        f.write(f"{node_name}\n")
                    f.write("\n")

            print(f"K-means clusters saved to {all_clusters_file}")

            # Save each cluster to a separate file and save edges within each cluster
            G = load_network_from_file(network_file)

            for i, cluster in enumerate(clusters):
                cluster_file = os.path.join(output_dir, f"motif_{motif_name}_{network_name}_cluster_{i+1}_nodes.txt")

                with open(cluster_file, 'w') as f:
                    f.write(f"# Nodes in cluster {i+1}\n")
                    f.write(f"# Motif: {motif_name}\n")
                    f.write(f"# Network: {network_name}\n")
                    f.write(f"# Number of nodes: {len(cluster)}\n\n")

                    for node_idx in sorted(cluster):
                        node_name = node_mapping[node_idx] if node_mapping else node_idx
                        f.write(f"{node_name}\n")

                print(f"Nodes in cluster {i+1} saved to {cluster_file}")

                # Save edges within the cluster
                self.save_edges_within_set(G, cluster, node_mapping, output_dir, motif_name, network_name, 
                                          f"Cluster {i+1}", f"cluster_{i+1}")

        except Exception as e:
            print(f"Error saving k-means clusters: {e}")

    def save_gmm_clusters(self, clusters, node_mapping, network_file, motif_name):
        """
        Save Gaussian Mixture Model cluster assignments to files.

        Args:
            clusters: List of lists, where each inner list contains the node indices for a cluster
            node_mapping: Mapping from indices to node names
            network_file: Path to the network file
            motif_name: Name of the motif
        """
        # Extract the base filename without directory and extension
        network_basename = os.path.basename(network_file)
        network_name, _ = os.path.splitext(network_basename)

        # Create output directory
        output_dir = os.path.join("data/results", network_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Create a file for all clusters
        all_clusters_file = os.path.join(output_dir, f"motif_{motif_name}_{network_name}_gmm_clusters.txt")

        try:
            # Save all clusters to a single file
            with open(all_clusters_file, 'w') as f:
                f.write(f"# Gaussian Mixture Model clustering results\n")
                f.write(f"# Motif: {motif_name}\n")
                f.write(f"# Network: {network_name}\n")
                f.write(f"# Number of clusters: {len(clusters)}\n\n")

                for i, cluster in enumerate(clusters):
                    f.write(f"# Cluster {i+1} ({len(cluster)} nodes):\n")
                    for node_idx in sorted(cluster):
                        node_name = node_mapping[node_idx] if node_mapping else node_idx
                        f.write(f"{node_name}\n")
                    f.write("\n")

            print(f"GMM clusters saved to {all_clusters_file}")

            # Save each cluster to a separate file and save edges within each cluster
            G = load_network_from_file(network_file)

            for i, cluster in enumerate(clusters):
                cluster_file = os.path.join(output_dir, f"motif_{motif_name}_{network_name}_gmm_cluster_{i+1}_nodes.txt")

                with open(cluster_file, 'w') as f:
                    f.write(f"# Nodes in GMM cluster {i+1}\n")
                    f.write(f"# Motif: {motif_name}\n")
                    f.write(f"# Network: {network_name}\n")
                    f.write(f"# Number of nodes: {len(cluster)}\n\n")

                    for node_idx in sorted(cluster):
                        node_name = node_mapping[node_idx] if node_mapping else node_idx
                        f.write(f"{node_name}\n")

                print(f"Nodes in GMM cluster {i+1} saved to {cluster_file}")

                # Save edges within the cluster
                self.save_edges_within_set(G, cluster, node_mapping, output_dir, motif_name, network_name, 
                                          f"GMM Cluster {i+1}", f"gmm_cluster_{i+1}")

        except Exception as e:
            print(f"Error saving GMM clusters: {e}")

    def create_fiedler_sets(self, fiedler_vector: np.ndarray, node_mapping: Optional[dict], 
                           network_file: str, motif_name: str) -> Tuple[List, List]:
        """
        Create sets S and T based on positive and negative values in the Fiedler vector.

        Args:
            fiedler_vector: The Fiedler vector (second eigenvector of the normalized Laplacian)
            node_mapping: Optional mapping from node indices to node names
            network_file: Path to the network file
            motif_name: Name of the motif

        Returns:
            Tuple containing two lists: (set S with negative Fiedler values, set T with positive or zero Fiedler values)
        """
        # Extract the base filename without directory and extension
        network_basename = os.path.basename(args.network)
        network_name, _ = os.path.splitext(network_basename)

        # Create output directory
        output_dir = os.path.join("data/results", network_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Partition nodes based on Fiedler vector sign
        negative_nodes = []  # Set S (nodes with negative Fiedler values)
        positive_nodes = []  # Set T (nodes with positive Fiedler values)

        # Check if all values in the Fiedler vector are non-negative
        all_non_negative = all(value >= 0 for value in fiedler_vector)

        for i, value in enumerate(fiedler_vector):
            if value < 0:
                negative_nodes.append(i)
            elif value > 0:
                positive_nodes.append(i)
            elif value == 0:
                # If all values are non-negative, put zeros in negative_nodes (set S)
                # Otherwise, keep zeros in positive_nodes (set T)
                if all_non_negative:
                    negative_nodes.append(i)
                else:
                    positive_nodes.append(i)

        # Sort the nodes for consistent output
        negative_nodes.sort()
        positive_nodes.sort()

        # Create filenames for the sets
        negative_file = os.path.join(output_dir, f"motif_{motif_name}_{network_name}_fiedler_negative_nodes.txt")
        positive_file = os.path.join(output_dir, f"motif_{motif_name}_{network_name}_fiedler_positive_nodes.txt")

        try:
            # Save the negative nodes (set S) to a file
            with open(negative_file, 'w') as f:
                if all(value >= 0 for value in fiedler_vector):
                    f.write(f"# Nodes with zero values in the Fiedler vector (set S)\n")
                else:
                    f.write(f"# Nodes with negative values in the Fiedler vector (set S)\n")
                f.write(f"# Motif: {motif_name}\n")
                f.write(f"# Network: {network_name}\n")
                f.write(f"# Number of nodes: {len(negative_nodes)}\n\n")

                # Write each node on a separate line with node name if mapping is available
                for node_idx in negative_nodes:
                    node_name = node_mapping[node_idx] if node_mapping else node_idx
                    f.write(f"{node_name}\n")

            if all(value >= 0 for value in fiedler_vector):
                print(f"Nodes with zero Fiedler values (set S) saved to {negative_file}")
            else:
                print(f"Nodes with negative Fiedler values (set S) saved to {negative_file}")

            # Save the positive nodes (set T) to a file
            with open(positive_file, 'w') as f:
                if all(value >= 0 for value in fiedler_vector):
                    f.write(f"# Nodes with positive values in the Fiedler vector (set T)\n")
                else:
                    f.write(f"# Nodes with positive or zero values in the Fiedler vector (set T)\n")
                f.write(f"# Motif: {motif_name}\n")
                f.write(f"# Network: {network_name}\n")
                f.write(f"# Number of nodes: {len(positive_nodes)}\n\n")

                # Write each node on a separate line with node name if mapping is available
                for node_idx in positive_nodes:
                    node_name = node_mapping[node_idx] if node_mapping else node_idx
                    f.write(f"{node_name}\n")

            if all(value >= 0 for value in fiedler_vector):
                print(f"Nodes with positive Fiedler values (set T) saved to {positive_file}")
            else:
                print(f"Nodes with positive or zero Fiedler values (set T) saved to {positive_file}")

            # Save edges within set S and set T
            G = load_network_from_file(network_file)
            self.save_edges_within_set(G, negative_nodes, node_mapping, output_dir, motif_name, network_name, "S", "fiedler_negative")
            self.save_edges_within_set(G, positive_nodes, node_mapping, output_dir, motif_name, network_name, "T", "fiedler_positive")

        except Exception as e:
            print(f"Error saving Fiedler-based node sets: {e}")

        return negative_nodes, positive_nodes

    def _get_next_file_number(self, output_dir: str, motif_name: str, network_name: str) -> int:
        """
        Get the next available file number for a given motif and network.

        This function checks existing files in the output directory and finds the highest
        number used for the given motif and network, then returns the next number in sequence.

        Args:
            output_dir: Directory to check for existing files
            motif_name: Name of the motif
            network_name: Name of the network

        Returns:
            The next available file number
        """
        # Create the directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            return 1  # If directory doesn't exist, start with 1

        # Get all files in the output directory
        files = os.listdir(output_dir)

        # Filter files that match the pattern: motif_{motif_name}_{network_name}_*.csv
        pattern = f"motif_{motif_name}_{network_name}_"
        matching_files = [f for f in files if f.startswith(pattern)]

        # If no matching files, start with 1
        if not matching_files:
            return 1

        # Extract the numbers from the filenames
        numbers = []
        for filename in matching_files:
            # Remove the prefix and any suffix
            number_part = filename.replace(pattern, "").split("_")[0].split(".")[0]
            try:
                numbers.append(int(number_part))
            except ValueError:
                # Skip files with non-integer numbers
                continue

        # If no valid numbers found, start with 1
        if not numbers:
            return 1

        # Return the next number in sequence
        return max(numbers) + 1

    def _find_motif_occurrences(self, G: nx.Graph, motif: nx.Graph) -> List[Set]:
        # Get the number of nodes in the motif
        motif_size = motif.number_of_nodes()
        motif_edges = motif.number_of_edges()
        occurrences = []

        # Optimization for triangle motifs (3-cliques)
        if motif_size == 3 and motif_edges == 3:
            print("Using optimized triangle finding algorithm...")
            # Use a dictionary to store unique triangles
            triangle_dict = {}
            # Process the graph in batches to avoid memory issues with large graphs
            batch_size = 1000
            nodes = list(G.nodes())
            total_nodes = len(nodes)

            for i in range(0, total_nodes, batch_size):
                batch_nodes = nodes[i:min(i+batch_size, total_nodes)]

                # Print progress
                print(f"Processing nodes {i+1}-{min(i+batch_size, total_nodes)} of {total_nodes}...")

                # Find triangles in this batch
                for node in batch_nodes:
                    neighbors = set(G.neighbors(node))
                    # For each pair of neighbors, check if they are connected
                    for u, v in itertools.combinations(neighbors, 2):
                        if G.has_edge(u, v):
                            # Create a sorted tuple of the triangle nodes to use as a unique key
                            sorted_triangle = tuple(sorted([node, u, v]))
                            # Store the triangle in the dictionary if it's not already there
                            if sorted_triangle not in triangle_dict:
                                triangle_dict[sorted_triangle] = {node, u, v}

                # Print the number of triangles found so far
                print(f"Found {len(triangle_dict)} unique triangles so far...")

            # Convert the dictionary values to a list
            occurrences = list(triangle_dict.values())

        # Optimization for square motifs (4-node cycles)
        elif motif_size == 4 and motif_edges == 4 and nx.is_isomorphic(motif, self.motifs["square"]):
            print("Using optimized square finding algorithm...")
            # Use a dictionary to store unique squares
            square_dict = {}
            # Process the graph in batches to avoid memory issues with large graphs
            batch_size = 1000
            nodes = list(G.nodes())
            total_nodes = len(nodes)

            for i in range(0, total_nodes, batch_size):
                batch_nodes = nodes[i:min(i+batch_size, total_nodes)]

                # Print progress
                print(f"Processing nodes {i+1}-{min(i+batch_size, total_nodes)} of {total_nodes}...")

                # Find squares in this batch
                for node in batch_nodes:
                    # Get neighbors of the node
                    neighbors = set(G.neighbors(node))

                    # For each neighbor, get its neighbors (excluding the original node)
                    for neighbor in neighbors:
                        neighbor_neighbors = set(G.neighbors(neighbor)) - {node}

                        # For each pair of nodes that are 2 steps away from the original node
                        for u in neighbor_neighbors:
                            u_neighbors = set(G.neighbors(u)) - {neighbor}

                            # Check if any of these nodes are also neighbors of the original node
                            for v in neighbors:
                                if v != neighbor and v in u_neighbors:
                                    # We found a square: node -> neighbor -> u -> v -> node
                                    # Create a sorted tuple of the square nodes to use as a unique key
                                    sorted_square = tuple(sorted([node, neighbor, u, v]))
                                    # Store the square in the dictionary if it's not already there
                                    if sorted_square not in square_dict:
                                        square_dict[sorted_square] = {node, neighbor, u, v}

                # Print the number of squares found so far
                print(f"Found {len(square_dict)} unique squares so far...")

            # Convert the dictionary values to a list
            occurrences = list(square_dict.values())

        # Optimization for diamond motifs
        elif motif_size == 4 and motif_edges == 4 and nx.is_isomorphic(motif, self.motifs["diamond"]):
            print("Using optimized diamond finding algorithm...")
            # Use a dictionary to store unique diamonds
            diamond_dict = {}
            # Process the graph in batches to avoid memory issues with large graphs
            batch_size = 1000
            nodes = list(G.nodes())
            total_nodes = len(nodes)

            for i in range(0, total_nodes, batch_size):
                batch_nodes = nodes[i:min(i+batch_size, total_nodes)]

                # Print progress
                print(f"Processing nodes {i+1}-{min(i+batch_size, total_nodes)} of {total_nodes}...")

                # Find diamonds in this batch
                for node in batch_nodes:
                    # Get neighbors of the node
                    neighbors = list(G.neighbors(node))

                    # Check if this node could be the central node of a diamond
                    # For each pair of neighbors, check if they are connected
                    for j, u in enumerate(neighbors):
                        for v in neighbors[j+1:]:
                            if G.has_edge(u, v):
                                # For each other neighbor, check if it forms a diamond
                                for w in neighbors:
                                    if w != u and w != v and not G.has_edge(w, u) and not G.has_edge(w, v):
                                        # We found a diamond: node (central), u, v, w
                                        # Create a sorted tuple of the diamond nodes to use as a unique key
                                        sorted_diamond = tuple(sorted([node, u, v, w]))
                                        # Store the diamond in the dictionary if it's not already there
                                        if sorted_diamond not in diamond_dict:
                                            diamond_dict[sorted_diamond] = {node, u, v, w}

                    # Check if this node could be a non-central node of a diamond
                    # For each neighbor, check if it could be the central node
                    for central in neighbors:
                        central_neighbors = set(G.neighbors(central)) - {node}
                        # For each pair of central's neighbors, check if they form a diamond
                        for u, v in itertools.combinations(central_neighbors, 2):
                            if G.has_edge(u, v) and not G.has_edge(node, u) and not G.has_edge(node, v):
                                # We found a diamond: central (central), node, u, v
                                # Create a sorted tuple of the diamond nodes to use as a unique key
                                sorted_diamond = tuple(sorted([central, node, u, v]))
                                # Store the diamond in the dictionary if it's not already there
                                if sorted_diamond not in diamond_dict:
                                    diamond_dict[sorted_diamond] = {central, node, u, v}

                # Print the number of diamonds found so far
                print(f"Found {len(diamond_dict)} unique diamonds so far...")

            # Convert the dictionary values to a list
            occurrences = list(diamond_dict.values())

        # Optimization for complete4 motifs (4-cliques)
        elif motif_size == 4 and motif_edges == 6 and nx.is_isomorphic(motif, self.motifs["complete4"]):
            print("Using optimized complete4 finding algorithm...")
            # Use a dictionary to store unique complete4s
            complete4_dict = {}
            # Process the graph in batches to avoid memory issues with large graphs
            batch_size = 1000
            nodes = list(G.nodes())
            total_nodes = len(nodes)

            for i in range(0, total_nodes, batch_size):
                batch_nodes = nodes[i:min(i+batch_size, total_nodes)]

                # Print progress
                print(f"Processing nodes {i+1}-{min(i+batch_size, total_nodes)} of {total_nodes}...")

                # Find complete4s in this batch
                for node in batch_nodes:
                    # Get neighbors of the node
                    neighbors = list(G.neighbors(node))

                    # For each triple of neighbors, check if they form a triangle
                    for u, v, w in itertools.combinations(neighbors, 3):
                        if G.has_edge(u, v) and G.has_edge(v, w) and G.has_edge(u, w):
                            # We found a complete4: node, u, v, w
                            # Create a sorted tuple of the complete4 nodes to use as a unique key
                            sorted_complete4 = tuple(sorted([node, u, v, w]))
                            # Store the complete4 in the dictionary if it's not already there
                            if sorted_complete4 not in complete4_dict:
                                complete4_dict[sorted_complete4] = {node, u, v, w}

                # Print the number of complete4s found so far
                print(f"Found {len(complete4_dict)} unique complete4s so far...")

            # Convert the dictionary values to a list
            occurrences = list(complete4_dict.values())

        # Optimization for cycle with chord motifs (4-node cycle with one chord)
        elif motif_size == 4 and motif_edges == 5 and nx.is_isomorphic(motif, self.motifs["cycle_chord"]):
            print("Using optimized cycle with chord finding algorithm...")
            # Use a dictionary to store unique cycle with chords
            cycle_chord_dict = {}
            # Process the graph in batches to avoid memory issues with large graphs
            batch_size = 1000
            nodes = list(G.nodes())
            total_nodes = len(nodes)

            for i in range(0, total_nodes, batch_size):
                batch_nodes = nodes[i:min(i+batch_size, total_nodes)]

                # Print progress
                print(f"Processing nodes {i+1}-{min(i+batch_size, total_nodes)} of {total_nodes}...")

                # Find cycle with chords in this batch
                for node in batch_nodes:
                    # Get neighbors of the node
                    neighbors = list(G.neighbors(node))

                    # For each pair of neighbors
                    for u, v in itertools.combinations(neighbors, 2):
                        # If the neighbors are connected, they form a triangle with the node
                        if G.has_edge(u, v):
                            # Look for a fourth node that connects to exactly two of the three nodes
                            u_neighbors = set(G.neighbors(u)) - {node, v}
                            v_neighbors = set(G.neighbors(v)) - {node, u}
                            node_neighbors = set(neighbors) - {u, v}

                            # Check u's neighbors
                            for w in u_neighbors:
                                if (w in node_neighbors and not G.has_edge(w, v)) or (w in v_neighbors and not G.has_edge(w, node)):
                                    # We found a cycle with chord: node, u, v, w
                                    sorted_cycle_chord = tuple(sorted([node, u, v, w]))
                                    if sorted_cycle_chord not in cycle_chord_dict:
                                        cycle_chord_dict[sorted_cycle_chord] = {node, u, v, w}

                            # Check v's neighbors
                            for w in v_neighbors:
                                if (w in node_neighbors and not G.has_edge(w, u)) or (w in u_neighbors and not G.has_edge(w, node)):
                                    # We found a cycle with chord: node, u, v, w
                                    sorted_cycle_chord = tuple(sorted([node, u, v, w]))
                                    if sorted_cycle_chord not in cycle_chord_dict:
                                        cycle_chord_dict[sorted_cycle_chord] = {node, u, v, w}

                            # Check node's neighbors
                            for w in node_neighbors:
                                if (w in u_neighbors and not G.has_edge(w, v)) or (w in v_neighbors and not G.has_edge(w, u)):
                                    # We found a cycle with chord: node, u, v, w
                                    sorted_cycle_chord = tuple(sorted([node, u, v, w]))
                                    if sorted_cycle_chord not in cycle_chord_dict:
                                        cycle_chord_dict[sorted_cycle_chord] = {node, u, v, w}

                # Print the number of cycle with chords found so far
                print(f"Found {len(cycle_chord_dict)} unique cycle with chords so far...")

            # Convert the dictionary values to a list
            occurrences = list(cycle_chord_dict.values())

        # Optimization for path3 motifs (path of length 3)
        elif motif_size == 4 and motif_edges == 3 and nx.is_isomorphic(motif, self.motifs["path3"]):
            print("Using optimized path3 finding algorithm...")
            # Use a dictionary to store unique path3s
            path3_dict = {}
            # Process the graph in batches to avoid memory issues with large graphs
            batch_size = 1000
            nodes = list(G.nodes())
            total_nodes = len(nodes)

            for i in range(0, total_nodes, batch_size):
                batch_nodes = nodes[i:min(i+batch_size, total_nodes)]

                # Print progress
                print(f"Processing nodes {i+1}-{min(i+batch_size, total_nodes)} of {total_nodes}...")

                # Find path3s in this batch
                for node in batch_nodes:
                    # Get neighbors of the node
                    neighbors = list(G.neighbors(node))

                    # For each neighbor, get its neighbors (excluding the original node)
                    for neighbor in neighbors:
                        neighbor_neighbors = set(G.neighbors(neighbor)) - {node}

                        # For each neighbor of neighbor, get its neighbors (excluding the neighbor)
                        for nn in neighbor_neighbors:
                            nn_neighbors = set(G.neighbors(nn)) - {neighbor}

                            # Check if nn has neighbors that are not connected to node or neighbor
                            for nnn in nn_neighbors:
                                if nnn != node and nnn not in neighbors and nnn not in neighbor_neighbors:
                                    # We found a path3: node -> neighbor -> nn -> nnn
                                    sorted_path3 = tuple(sorted([node, neighbor, nn, nnn]))
                                    if sorted_path3 not in path3_dict:
                                        path3_dict[sorted_path3] = {node, neighbor, nn, nnn}

                # Print the number of path3s found so far
                print(f"Found {len(path3_dict)} unique path3s so far...")

            # Convert the dictionary values to a list
            occurrences = list(path3_dict.values())

        # Optimization for star motifs (central node with 3 neighbors)
        elif motif_size == 4 and motif_edges == 3 and nx.is_isomorphic(motif, self.motifs["star"]):
            print("Using optimized star finding algorithm...")
            # Use a dictionary to store unique stars
            star_dict = {}
            # Process the graph in batches to avoid memory issues with large graphs
            batch_size = 1000
            nodes = list(G.nodes())
            total_nodes = len(nodes)

            for i in range(0, total_nodes, batch_size):
                batch_nodes = nodes[i:min(i+batch_size, total_nodes)]

                # Print progress
                print(f"Processing nodes {i+1}-{min(i+batch_size, total_nodes)} of {total_nodes}...")

                # Find stars in this batch
                for node in batch_nodes:
                    # Get neighbors of the node
                    neighbors = list(G.neighbors(node))

                    # If the node has at least 3 neighbors, it could be the center of a star
                    if len(neighbors) >= 3:
                        # For each combination of 3 neighbors
                        for u, v, w in itertools.combinations(neighbors, 3):
                            # Check that the neighbors are not connected to each other
                            if not G.has_edge(u, v) and not G.has_edge(v, w) and not G.has_edge(u, w):
                                # We found a star: node (center), u, v, w
                                sorted_star = tuple(sorted([node, u, v, w]))
                                if sorted_star not in star_dict:
                                    star_dict[sorted_star] = {node, u, v, w}

                # Print the number of stars found so far
                print(f"Found {len(star_dict)} unique stars so far...")

            # Convert the dictionary values to a list
            occurrences = list(star_dict.values())

        else:
            print("Using general motif finding algorithm...")
            # For other motifs, use a more efficient approach
            # Process in batches to avoid memory issues
            batch_size = 1000
            node_list = list(G.nodes())
            total_nodes = len(node_list)
            processed = 0

            for i in range(0, total_nodes, batch_size):
                batch_nodes = node_list[i:min(i+batch_size, total_nodes)]
                batch_size_actual = len(batch_nodes)
                processed += batch_size_actual

                # Print progress
                print(f"Processing nodes {i+1}-{min(i+batch_size, total_nodes)} of {total_nodes} ({processed/total_nodes*100:.1f}%)...")

                # For each combination of nodes in the batch
                for nodes in itertools.combinations(batch_nodes, motif_size):
                    # Get the subgraph induced by these nodes
                    subgraph = G.subgraph(nodes)

                    # Only check if the subgraph has the right number of edges
                    if subgraph.number_of_edges() == motif_edges:
                        # Check if this subgraph is isomorphic to the motif
                        if nx.is_isomorphic(subgraph, motif):
                            occurrences.append(set(nodes))
        return occurrences

    def save_adjacency_matrix(self, adj_matrix, motif_name, motif_occurrences, node_to_idx, batch_size=1000, ):
        # Create a subfolder named after the input file if network_file is provided
        # Extract the base filename without directory and extension
        network_basename = os.path.basename(args.network)
        network_name, _ = os.path.splitext(network_basename)

        # Create a subfolder path
        output_dir = os.path.join("data/results", network_name)

        # Create directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Generate a unique number for the filename (using incremental numbering)
        unique_number = self._get_next_file_number(output_dir, motif_name, network_name)

        # Create the filename with the new naming convention: motif_name_network_name_number.s
        matrix_file = os.path.join(output_dir, f"motif_{motif_name}_{unique_number}.csv")
        idx_to_node = {i: node for node, i in node_to_idx.items()}

        # Create a pandas DataFrame with real node names as row and column labels
        # Get the list of node names in the correct order
        node_names = [idx_to_node[i] for i in range(len(node_to_idx))]

        # Create the DataFrame
        df = pd.DataFrame(adj_matrix, index=node_names, columns=node_names)
        df.to_csv(matrix_file)

        print(f"Adjacency matrix saved to {matrix_file}")

        occurrences_file = os.path.join(output_dir, f"motif_{motif_name}_{unique_number}_occurrences.txt")
        total_occurrences = len(motif_occurrences)
        with open(occurrences_file, 'w') as f:
            f.write(f"# Occurrences of motif '{motif_name}' in the graph\n")
            f.write(f"# Each line represents one occurrence, listing the nodes involved\n\n")
            # Write occurrences in batches to avoid memory issues
            for i in range(0, total_occurrences, batch_size):
                batch_end = min(i + batch_size, total_occurrences)
                batch = motif_occurrences[i:batch_end]
                # Print progress
                print(f"Writing occurrences {i+1}-{batch_end} of {total_occurrences} to file...")
                for occurrence in batch:
                    f.write(' '.join(str(node) for node in occurrence) + '\n')
        print(f"List of occurrences saved to {occurrences_file}")

    def save_fiedler_vector(self, fiedler_vector: np.ndarray, motif_name,
                             node_mapping: Optional[dict] = None) -> None:
        # Create a subfolder named after the input file if network_file is provided
        # Extract the base filename without directory and extension
        network_basename = os.path.basename(args.network)
        network_name, _ = os.path.splitext(network_basename)

        # Create a subfolder path
        output_dir = os.path.join("data/results", network_name)

        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)


        # Generate a unique number for the filename (using incremental numbering)
        unique_number = self._get_next_file_number(output_dir, motif_name, network_name)

        # Create the filename with the naming convention: motif_name_network_name_number_fiedler_vector.txt
        fiedler_file = os.path.join(output_dir, f"motif_{motif_name}_{unique_number}_fiedler_vector.txt")

        # Use the provided node_mapping if available, otherwise create a default mapping


        with open(fiedler_file, 'w') as f:
            f.write(f"# Fiedler vector (eigenvector corresponding to the second smallest eigenvalue)\n")
            f.write(f"# Format: node_name fiedler_value (sorted by fiedler_value)\n\n")

            # Create a list of (index, value) pairs
            index_value_pairs = [(i, value) for i, value in enumerate(fiedler_vector)]

            # Sort the pairs by Fiedler vector value
            sorted_pairs = sorted(index_value_pairs, key=lambda x: x[1])

            # Write the sorted nodes and values to the file (rounded to 5 decimal places)
            for i, value in sorted_pairs:
                node_id = node_mapping[i] if node_mapping else i
                f.write(f"{node_id} {str(value)}\n") # :.11f
        print(f"Fiedler vector saved to {fiedler_file}")

    def save_conductance(self, df, network_name, motif_name, node_mapping=None):
        # Create a subfolder named after the input file if network_file is provided
        # Extract the base filename without directory and extension
        network_basename = os.path.basename(args.network)
        network_name, _ = os.path.splitext(network_basename)

        # Create a subfolder path
        output_dir = os.path.join("data/results", network_name)

        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Generate a unique number for the filename (using incremental numbering)
        unique_number = self._get_next_file_number(output_dir, motif_name, network_name)
        # Save the nodes in the prefix with the lowest conductance to a file
        if output_dir and motif_name and network_name and unique_number:
            # Find the prefix with the lowest conductance
            min_conductance = df['conductance'].min()
            min_idx = df['conductance'].idxmin()
            min_prefix_nodes = df.loc[min_idx, 'prefix_nodes']

            # Load the original graph to get all nodes
            G = load_network_from_file(args.network)
            # Get all nodes from the original graph
            all_nodes = set(range(len(node_mapping)) if node_mapping else G.nodes())

            # Calculate the complement set T (all nodes not in min_prefix_nodes)
            complement_nodes = list(all_nodes - set(min_prefix_nodes))
            complement_nodes.sort()  # Sort for consistent output

            # Create the filenames
            min_conductance_file = os.path.join(output_dir, f"motif_{motif_name}_{network_name}_min_conductance_nodes.txt")
            complement_file = os.path.join(output_dir, f"motif_{motif_name}_{network_name}_complement_nodes.txt")

            try:
                # Create directory if it doesn't exist
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                # Save the nodes in set S to the file
                with open(min_conductance_file, 'w') as f:
                    f.write(f"# Nodes in the prefix with the lowest motif conductance (set S)\n")
                    f.write(f"# Motif: {motif_name}\n")
                    f.write(f"# Network: {network_name}\n")
                    f.write(f"# Minimum conductance: {min_conductance}\n")
                    f.write(f"# Number of nodes in prefix: {len(min_prefix_nodes)}\n\n")

                    # Write each node on a separate line with node name if mapping is available
                    for node_idx in min_prefix_nodes:
                        node_name = node_mapping[node_idx] if node_mapping else node_idx
                        f.write(f"{node_name}\n")

                print(f"Nodes in the prefix with the lowest conductance (set S) saved to {min_conductance_file}")

                # Save the complement nodes (set T) to a separate file
                with open(complement_file, 'w') as f:
                    f.write(f"# Complement nodes (set T) for the prefix with the lowest motif conductance\n")
                    f.write(f"# Motif: {motif_name}\n")
                    f.write(f"# Network: {network_name}\n")
                    f.write(f"# Minimum conductance: {min_conductance}\n")
                    f.write(f"# Number of nodes in complement: {len(complement_nodes)}\n\n")

                    # Write each node on a separate line with node name if mapping is available
                    for node_idx in complement_nodes:
                        node_name = node_mapping[node_idx] if node_mapping else node_idx
                        f.write(f"{node_name}\n")

                print(f"Complement nodes (set T) saved to {complement_file}")

                # Save edges within set S and set T
                G = load_network_from_file(args.network)
                self.save_edges_within_set(G, min_prefix_nodes, node_mapping, output_dir, motif_name, network_name, "S", "min_conductance")
                self.save_edges_within_set(G, complement_nodes, node_mapping, output_dir, motif_name, network_name, "T", "complement")

            except Exception as e:
                print(f"Error saving nodes: {e}")

    def find_good_seed_nodes(self, G: nx.Graph, adj_matrix, motif_name: str, num_nodes: int = 10, method: str = 'combined') -> List[int]:
        """
        Find good seed nodes for local clustering.

        This method identifies nodes that are likely to be good seeds for local clustering
        based on various centrality metrics and motif participation.

        Args:
            G: The original graph
            adj_matrix: The motif adjacency matrix
            motif_name: Name of the motif
            num_nodes: Number of seed nodes to return (default: 10)
            method: Method to use for ranking nodes ('degree', 'pagerank', 'motif', 'combined')

        Returns:
            List of node indices that are likely to be good seeds for local clustering
        """
        # Create a mapping from node identifiers to indices
        node_to_idx = {node: i for i, node in enumerate(G.nodes())}
        print(f"Finding {num_nodes} good seed nodes for local clustering using method '{method}'...")

        # Get the motif
        motif = self.get_motif(motif_name)

        # Find all occurrences of the motif in the graph
        print("Finding motif occurrences...")
        motif_occurrences = self._find_motif_occurrences(G, motif)
        print(f"Found {len(motif_occurrences)} occurrences of motif '{motif_name}'")

        # Count motif participation for each node
        motif_participation = {}
        for occurrence in motif_occurrences:
            for node in occurrence:
                if node in motif_participation:
                    motif_participation[node] += 1
                else:
                    motif_participation[node] = 1

        # Convert to a list of (node_idx, count) tuples and sort by count (descending)
        motif_participation_list = [(node_to_idx[node], count) for node, count in motif_participation.items()]
        motif_participation_list.sort(key=lambda x: x[1], reverse=True)

        # Compute degree centrality
        print("Computing degree centrality...")
        if sp.issparse(adj_matrix):
            degrees = np.array(adj_matrix.sum(axis=1)).flatten()
        else:
            degrees = adj_matrix.sum(axis=1)

        # Convert to a list of (node, degree) tuples and sort by degree (descending)
        degree_list = [(i, degrees[i]) for i in range(len(degrees))]
        degree_list.sort(key=lambda x: x[1], reverse=True)

        # Compute PageRank
        print("Computing PageRank...")
        pagerank = nx.pagerank(G)

        # Convert to a list of (node_idx, pagerank) tuples and sort by PageRank (descending)
        pagerank_list = [(node_to_idx[node], rank) for node, rank in pagerank.items()]
        pagerank_list.sort(key=lambda x: x[1], reverse=True)

        # Select seed nodes based on the specified method
        if method == 'degree':
            # Use nodes with highest degree centrality
            seed_nodes = [node for node, _ in degree_list[:num_nodes]]
        elif method == 'pagerank':
            # Use nodes with highest PageRank
            seed_nodes = [node for node, _ in pagerank_list[:num_nodes]]
        elif method == 'motif':
            # Use nodes with highest motif participation
            seed_nodes = [node for node, _ in motif_participation_list[:num_nodes]]
        elif method == 'combined':
            # Use a combination of metrics
            # Normalize scores to [0, 1] range
            max_degree = max(degrees)
            max_pagerank = max(pagerank.values())
            max_motif = max(motif_participation.values()) if motif_participation else 1

            # Compute combined score for each node
            combined_scores = {}
            for node in G.nodes():
                # Get normalized scores (default to 0 if not available)
                node_idx = node_to_idx[node]
                degree_score = degrees[node_idx] / max_degree if node_idx < len(degrees) else 0
                pagerank_score = pagerank.get(node, 0) / max_pagerank
                motif_score = motif_participation.get(node, 0) / max_motif

                # Compute combined score (weighted average)
                combined_scores[node_idx] = 0.3 * degree_score + 0.3 * pagerank_score + 0.4 * motif_score

            # Convert to a list of (node, score) tuples and sort by score (descending)
            combined_list = [(node, score) for node, score in combined_scores.items()]
            combined_list.sort(key=lambda x: x[1], reverse=True)

            # Use nodes with highest combined score
            seed_nodes = [node for node, _ in combined_list[:num_nodes]]
        else:
            raise ValueError(f"Unknown method: {method}")

        print(f"Found {len(seed_nodes)} good seed nodes: {seed_nodes}")
        return seed_nodes

    def save_multiway_conductance(self, df, network_name, motif_name, node_mapping, clusters):
        """
        Save multi-way conductance results to files.

        Args:
            df: DataFrame with conductance results for each cluster
            network_name: Name of the network
            motif_name: Name of the motif
            node_mapping: Mapping from indices to node names
            clusters: List of lists, where each inner list contains the node indices for a cluster
        """
        # Create a subfolder named after the input file if network_file is provided
        # Extract the base filename without directory and extension
        network_basename = os.path.basename(args.network)
        network_name, _ = os.path.splitext(network_basename)

        # Create a subfolder path
        output_dir = os.path.join("data/results", network_name)

        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Generate a unique number for the filename (using incremental numbering)
        unique_number = self._get_next_file_number(output_dir, motif_name, network_name)

        # Save the conductance results to a file
        if output_dir and motif_name and network_name and unique_number:
            # Create the filename
            conductance_file = os.path.join(output_dir, f"motif_{motif_name}_{network_name}_multiway_conductance.txt")

            try:
                # Create directory if it doesn't exist
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                # Save the conductance results to the file
                with open(conductance_file, 'w') as f:
                    f.write(f"# Multi-way motif conductance results\n")
                    f.write(f"# Motif: {motif_name}\n")
                    f.write(f"# Network: {network_name}\n")
                    f.write(f"# Number of clusters: {len(clusters)}\n\n")

                    # Write the DataFrame to the file
                    f.write(df.to_string(index=False))

                    # Write summary statistics
                    if not df.empty:
                        f.write("\n\n# Summary statistics\n")
                        max_conductance = df['conductance'].max()
                        max_idx = df['conductance'].idxmax()
                        max_cluster = df.loc[max_idx, 'cluster']
                        f.write(f"Maximum conductance: {max_conductance} for cluster {max_cluster}\n")

                        min_conductance = df['conductance'].min()
                        min_idx = df['conductance'].idxmin()
                        min_cluster = df.loc[min_idx, 'cluster']
                        f.write(f"Minimum conductance: {min_conductance} for cluster {min_cluster}\n")

                        avg_conductance = df['conductance'].mean()
                        f.write(f"Average conductance: {avg_conductance}\n")

                print(f"Multi-way conductance results saved to {conductance_file}")

                # Load the original graph
                G = load_network_from_file(args.network)

                # Save the nodes in the cluster with the lowest conductance to a file
                if not df.empty:
                    min_idx = df['conductance'].idxmin()
                    min_cluster_idx = df.loc[min_idx, 'cluster'] - 1  # Adjust for 0-based indexing
                    min_cluster_nodes = clusters[min_cluster_idx]

                    min_conductance_file = os.path.join(output_dir, f"motif_{motif_name}_{network_name}_min_conductance_cluster_nodes.txt")

                    with open(min_conductance_file, 'w') as f:
                        f.write(f"# Nodes in the cluster with the lowest motif conductance\n")
                        f.write(f"# Motif: {motif_name}\n")
                        f.write(f"# Network: {network_name}\n")
                        f.write(f"# Cluster: {min_cluster_idx + 1}\n")
                        f.write(f"# Conductance: {df.loc[min_idx, 'conductance']}\n")
                        f.write(f"# Number of nodes: {len(min_cluster_nodes)}\n\n")

                        # Write each node on a separate line with node name if mapping is available
                        for node_idx in min_cluster_nodes:
                            node_name = node_mapping[node_idx] if node_mapping else node_idx
                            f.write(f"{node_name}\n")

                    print(f"Nodes in the cluster with the lowest conductance saved to {min_conductance_file}")

                    # Save edges within the cluster with the lowest conductance
                    self.save_edges_within_set(G, min_cluster_nodes, node_mapping, output_dir, motif_name, network_name, 
                                              f"Cluster {min_cluster_idx + 1}", f"min_conductance_cluster")

            except Exception as e:
                print(f"Error saving multi-way conductance results: {e}")

# Example usage
if __name__ == "__main__":
    import argparse
    from network_loader import load_network_from_file

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Motif analysis and spectral clustering for undirected graphs.')
    parser.add_argument('--network', type=str, help='Path to the network file')
    parser.add_argument('--motif', type=str, help='Name of the motif to analyze')
    parser.add_argument('--input', type=str, help='Path to the input adjacency matrix file (can be .txt or .csv)')
    parser.add_argument('--clusters', type=int, help='Number of clusters for k-means clustering (if not specified, bisection is used unless --use-eigengap is specified)')
    parser.add_argument('--algorithm', type=str, choices=['kmeans', 'minibatch', 'gmm', 'mappr'], default='kmeans', 
                        help='Clustering algorithm to use: kmeans (default), minibatch (mini-batch k-means), gmm (Gaussian Mixture Models), or mappr (Motif-based Approximate Personalized PageRank)')
    parser.add_argument('--use-eigengap', action='store_true', 
                        help='Use eigengap heuristic to automatically determine the optimal number of clusters')
    parser.add_argument('--max-k', type=int, default=20, 
                        help='Maximum number of clusters to consider when using eigengap heuristic (default: 20)')
    parser.add_argument('--covariance-type', type=str, choices=['full', 'tied', 'diag', 'spherical'], default='full',
                        help='Type of covariance parameters to use for GMM (default: full)')
    parser.add_argument('--n-init', type=int, default=10,
                        help='Number of initializations to perform for GMM (default: 10)')
    parser.add_argument('--batch-size', type=int, default=100, help='Batch size for mini-batch k-means (default: 100)')
    parser.add_argument('--max-iter', type=int, default=100, help='Maximum number of iterations for mini-batch k-means (default: 100)')
    parser.add_argument('--reassignment-ratio', type=float, default=0.01, 
                        help='Control the fraction of the maximum number of counts for a center to be reassigned in mini-batch k-means (default: 0.01)')
    parser.add_argument('--seed-nodes', type=str, nargs='+', help='Seed node(s) for MAPPR local clustering. Multiple seed nodes can be provided separated by spaces. If not provided, good seed nodes will be automatically found.')
    parser.add_argument('--alpha', type=float, default=0.15, help='Teleportation parameter for MAPPR (default: 0.15)')
    parser.add_argument('--epsilon', type=float, default=1e-6, help='Approximation parameter for MAPPR (default: 1e-6)')
    parser.add_argument('--auto-seed', action='store_true', help='Automatically find good seed nodes for MAPPR local clustering')
    parser.add_argument('--seed-method', type=str, choices=['degree', 'pagerank', 'motif', 'combined'], default='combined', help='Method to use for ranking nodes when finding good seed nodes (default: combined)')
    parser.add_argument('--num-seeds', type=int, default=10, help='Number of seed nodes to find when using --auto-seed (default: 10)')
    args = parser.parse_args()

    # Initialize the motifs handler
    motifs = UndirectedMotifs()

    # Print available motifs
    print(f"Available motifs: {motifs.list_motifs()}")

    # If a network file and motif are provided, analyze the network
    if args.network and args.motif:
        # Load the network
        G = load_network_from_file(args.network)
        # Create the motif adjacency matrix
        adjacency_matrix, idx_to_node = motifs.create_motif_adjacency_matrix(G, args.motif, args.network)

        # If eigengap heuristic is requested, compute optimal k
        if args.use_eigengap:
            print(f"Using eigengap heuristic to determine optimal number of clusters (max_k={args.max_k})...")
            optimal_k, eigvals, eigvecs, idx_noniso, mask = motifs.compute_optimal_k_eigengap(adjacency_matrix, args.max_k)

            # Use the optimal k for clustering
            if optimal_k > 1:
                if args.algorithm == 'kmeans':
                    print(f"Performing k-means clustering with {optimal_k} clusters (determined by eigengap heuristic)...")
                    clusters = motifs.compute_kmeans_clusters(adjacency_matrix, idx_to_node, args.network, args.motif, optimal_k)
                elif args.algorithm == 'minibatch':
                    print(f"Performing mini-batch k-means clustering with {optimal_k} clusters (determined by eigengap heuristic)...")
                    print(f"Parameters: batch_size={args.batch_size}, max_iter={args.max_iter}, reassignment_ratio={args.reassignment_ratio}")
                    clusters = motifs.compute_minibatch_kmeans_clusters(
                        adjacency_matrix, 
                        idx_to_node, 
                        args.network, 
                        args.motif, 
                        optimal_k,
                        batch_size=args.batch_size,
                        max_iter=args.max_iter,
                        reassignment_ratio=args.reassignment_ratio
                    )
                elif args.algorithm == 'gmm':
                    print(f"Performing Gaussian Mixture Model clustering with {optimal_k} clusters (determined by eigengap heuristic)...")
                    print(f"Parameters: covariance_type={args.covariance_type}, n_init={args.n_init}, max_iter={args.max_iter}")
                    clusters = motifs.compute_gmm_clusters(
                        adjacency_matrix,
                        idx_to_node,
                        args.network,
                        args.motif,
                        optimal_k,
                        covariance_type=args.covariance_type,
                        n_init=args.n_init,
                        max_iter=args.max_iter
                    )
                print(f"Found {len(clusters)} clusters with sizes: {[len(c) for c in clusters]}")
                # Compute multi-way conductance for the clusters
                print("Computing multi-way conductance for the clusters...")
                motifs.compute_multiway_conductance(adjacency_matrix, clusters, idx_to_node, args.network, args.motif)
            else:
                # If optimal_k is 1, use traditional bisection
                print("Eigengap heuristic suggests 1 cluster. Performing traditional bisection with Fiedler vector...")
                fiedler = motifs.compute_fiedler(adjacency_matrix, idx_to_node, args.network, args.motif)
                motifs.compute_conductance(adjacency_matrix, fiedler, idx_to_node, args.network, args.motif)
        # If MAPPR algorithm is selected
        elif args.algorithm == 'mappr':
            # Check if we need to automatically find good seed nodes
            if not args.seed_nodes or args.auto_seed:
                print("Automatically finding good seed nodes for local clustering...")
                seed_nodes = motifs.find_good_seed_nodes(
                    G,
                    adjacency_matrix,
                    args.motif,
                    num_nodes=args.num_seeds,
                    method=args.seed_method
                )

                # Convert node indices to node names if mapping is available
                if idx_to_node:
                    seed_node_names = [idx_to_node[idx] for idx in seed_nodes]
                else:
                    seed_node_names = [str(idx) for idx in seed_nodes]

                print(f"Using automatically found seed nodes: {seed_node_names}")
            else:
                seed_node_names = args.seed_nodes
                print(f"Using provided seed nodes: {seed_node_names}")

            print(f"Performing MAPPR local clustering...")
            print(f"Parameters: alpha={args.alpha}, epsilon={args.epsilon}")

            clusters = motifs.compute_mappr_cluster(
                adjacency_matrix,
                idx_to_node,
                args.network,
                args.motif,
                seed_node_names,
                alpha=args.alpha,
                epsilon=args.epsilon
            )

            # Print summary of all clusters
            print("\nSummary of all clusters:")
            for seed_node, cluster in clusters.items():
                print(f"  Seed node {seed_node}: {len(cluster)} nodes in cluster")

        # If number of clusters is specified, use clustering
        elif args.clusters and args.clusters > 1:
            if args.algorithm == 'kmeans':
                print(f"Performing k-means clustering with {args.clusters} clusters...")
                clusters = motifs.compute_kmeans_clusters(adjacency_matrix, idx_to_node, args.network, args.motif, args.clusters)
            elif args.algorithm == 'minibatch':
                print(f"Performing mini-batch k-means clustering with {args.clusters} clusters...")
                print(f"Parameters: batch_size={args.batch_size}, max_iter={args.max_iter}, reassignment_ratio={args.reassignment_ratio}")
                clusters = motifs.compute_minibatch_kmeans_clusters(
                    adjacency_matrix, 
                    idx_to_node, 
                    args.network, 
                    args.motif, 
                    args.clusters,
                    batch_size=args.batch_size,
                    max_iter=args.max_iter,
                    reassignment_ratio=args.reassignment_ratio
                )
            elif args.algorithm == 'gmm':
                print(f"Performing Gaussian Mixture Model clustering with {args.clusters} clusters...")
                print(f"Parameters: covariance_type={args.covariance_type}, n_init={args.n_init}, max_iter={args.max_iter}")
                clusters = motifs.compute_gmm_clusters(
                    adjacency_matrix,
                    idx_to_node,
                    args.network,
                    args.motif,
                    args.clusters,
                    covariance_type=args.covariance_type,
                    n_init=args.n_init,
                    max_iter=args.max_iter
                )
            print(f"Found {len(clusters)} clusters with sizes: {[len(c) for c in clusters]}")
            # Compute multi-way conductance for the clusters
            print("Computing multi-way conductance for the clusters...")
            motifs.compute_multiway_conductance(adjacency_matrix, clusters, idx_to_node, args.network, args.motif)
        else:
            # Otherwise, use traditional bisection with Fiedler vector
            print("Performing traditional bisection with Fiedler vector...")
            fiedler = motifs.compute_fiedler(adjacency_matrix, idx_to_node, args.network, args.motif)
            motifs.compute_conductance(adjacency_matrix, fiedler, idx_to_node, args.network, args.motif)
