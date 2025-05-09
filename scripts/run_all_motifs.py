#!/usr/bin/env python3
"""
Script to run all available motifs on a given network.

This script takes a network file as input and counts the occurrences of all available motifs.
"""

import argparse
import os
import time
import pandas as pd
import numpy as np
import itertools
from typing import List, Dict, Tuple, Optional
from sklearn.mixture import GaussianMixture

import networkx as nx

from motif_spectral_analysis import UndirectedMotifs
from network_loader import load_network_from_file
from BonusFunctions import mappr

class MotifWrapper(UndirectedMotifs):
    """
    A wrapper around UndirectedMotifs that provides simplified clustering methods.
    """
    def __init__(self, network_file):
        super().__init__()
        self.network_file = network_file

    def create_motif_adjacency_matrix_simple(self, G, motif_name):
        """
        Create a motif adjacency matrix without saving it to a file.
        """
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

        # Convert adjacency matrix to float for better numerical stability
        adj_matrix_float = adj_matrix.astype(float)

        return adj_matrix_float, idx_to_node

    def compute_kmeans_clusters_simple(self, adj_matrix, k):
        """
        Perform k-means clustering without saving results to files.
        """
        from sklearn.cluster import KMeans

        # Compute eigenvectors
        eigvals, eigvecs, idx_noniso, mask = self.compute_eigenvectors(adj_matrix, k)

        # Use eigenvectors starting from the second one (skip the trivial eigenvector)
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

        return clusters

    def compute_minibatch_kmeans_clusters_simple(self, adj_matrix, k, batch_size=100, max_iter=100, reassignment_ratio=0.01):
        """
        Perform mini-batch k-means clustering without saving results to files.
        """
        from sklearn.cluster import MiniBatchKMeans

        # Compute eigenvectors
        eigvals, eigvecs, idx_noniso, mask = self.compute_eigenvectors(adj_matrix, k)

        # Use eigenvectors starting from the second one (skip the trivial eigenvector)
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

        return clusters

    def compute_gmm_clusters_simple(self, adj_matrix, k, covariance_type='full', n_init=10, max_iter=100):
        """
        Perform Gaussian Mixture Model clustering without saving results to files.
        """
        # Compute eigenvectors
        eigvals, eigvecs, idx_noniso, mask = self.compute_eigenvectors(adj_matrix, k)

        # Use eigenvectors starting from the second one (skip the trivial eigenvector)
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

        return clusters

    def compute_mappr_cluster_simple(self, adj_matrix, seed_nodes, alpha=0.15, epsilon=1e-6):
        """
        Perform Motif-based Approximate Personalized PageRank (MAPPR) for local clustering without saving results to files.
        """
        # Convert single seed node to list if necessary
        if not isinstance(seed_nodes, list):
            seed_nodes = [seed_nodes]

        print(f"Performing MAPPR local clustering with seed nodes {seed_nodes}...")
        print(f"Parameters: alpha={alpha}, epsilon={epsilon}")

        # Perform MAPPR for each seed node individually
        clusters = {}

        for i, seed_node_idx in enumerate(seed_nodes):
            print(f"Computing cluster for seed node {seed_node_idx} ({i+1}/{len(seed_nodes)})...")
            cluster, conductance, p = mappr(adj_matrix, [seed_node_idx], alpha, epsilon)

            # Store the results
            seed_node_name = seed_nodes[i]
            clusters[seed_node_name] = cluster

            print(f"Found cluster for seed node {seed_node_name} with {len(cluster)} nodes and conductance {conductance}")

        return clusters


def run_all_motifs(network_file: str, seed_methods: Optional[List[str]] = None, subfolder_name: Optional[str] = None, optimal_k: Optional[int] = None) -> None:
    """
    Run analysis for all available motifs on the given network.

    Args:
        network_file: Path to the network file
        seed_methods: List of seed methods to use for MAPPR clustering. If None, uses "combined" and "fiedler" methods to find good seed nodes.
                     The number of seed nodes is determined by the highest optimal k value from eigen-gap, elbow, and silhouette methods,
                     or by the provided optimal_k if specified.
        subfolder_name: Optional name of the subfolder containing the network file, used to generate unique summary filenames.
                       If None, the network name is extracted from the filename.
        optimal_k: Optional value to use for k instead of calculating optimal k. This value will be used for k-means, 
                  mini-batch k-means, GMM, and as the number of seed nodes for MAPPR.
    """
    # Initialize the motifs handler with the network file
    motifs = MotifWrapper(network_file)

    # Get all available motifs
    all_motifs = motifs.list_motifs()
    print(f"Available motifs: {all_motifs}")

    # Load the network
    print(f"Loading network from {network_file}...")
    G = load_network_from_file(network_file)
    print(f"Network loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Get the network name (filename without extension)
    network_name = os.path.splitext(os.path.basename(network_file))[0]

    # Use the subfolder name if provided, otherwise use the network name
    output_name = subfolder_name if subfolder_name else network_name

    # Create a summary file with a unique name based on the subfolder name
    summary_file = os.path.join("../data", "results", "all_motifs", f"{output_name}_summary.txt")
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)

    # Create a CSV file for detailed results with a unique name based on the subfolder name
    csv_file = os.path.join("../data", "results", "all_motifs", f"{output_name}_motifs.csv")

    print(f"Summary will be saved to: {summary_file}")
    print(f"Detailed results will be saved to: {csv_file}")

    # Initialize results dictionary
    results = []

    # Process each motif
    for motif_name in all_motifs:
        print(f"\n{'='*80}")
        print(f"Processing motif: {motif_name}")
        print(f"{'='*80}")

        start_time = time.time()

        try:
            # Get the motif graph
            motif = motifs.get_motif(motif_name)

            # Find all occurrences of the motif in the graph
            print(f"Finding occurrences of motif '{motif_name}' in the graph...")
            motif_occurrences = motifs._find_motif_occurrences(G, motif)
            occurrences_count = len(motif_occurrences)
            motif_time = time.time() - start_time

            print(f"Found {occurrences_count} occurrences in {motif_time:.2f} seconds")

            # Store basic results
            result = {
                'Motif': motif_name,
                'Occurrences': occurrences_count,
                'Time(s)': motif_time
            }

            # If there are occurrences, perform clustering
            if occurrences_count > 0:
                # Create motif adjacency matrix
                print(f"Creating motif adjacency matrix for '{motif_name}'...")
                adj_matrix, idx_to_node = motifs.create_motif_adjacency_matrix_simple(G, motif_name)

                # Determine optimal number of clusters (k) using different methods or use provided optimal_k
                if optimal_k is not None:
                    print(f"Using provided optimal k={optimal_k} for all clustering methods")
                    k_eigengap = optimal_k
                    k_elbow = optimal_k
                    k_silhouette = optimal_k

                    # Set times to 0 since we're not computing
                    eigengap_time = 0
                    elbow_time = 0
                    silhouette_time = 0

                    # Store in result dictionary
                    result['EigenGap_Time'] = eigengap_time
                    result['Elbow_Time'] = elbow_time
                    result['Silhouette_Time'] = silhouette_time
                else:
                    print(f"Determining optimal number of clusters for '{motif_name}'...")
                try:
                    # Skip optimal k calculations if optimal_k is provided
                    if optimal_k is None:
                        # 1. Eigen-gap method for k-means
                        print(f"Computing optimal k using eigen-gap method for '{motif_name}'...")
                        eigengap_start_time = time.time()
                        k_eigengap = motifs.compute_optimal_k_eigengap(adj_matrix)
                        eigengap_time = time.time() - eigengap_start_time
                        print(f"Optimal number of clusters (eigen-gap) for '{motif_name}': {k_eigengap}")
                        result['EigenGap_Time'] = eigengap_time

                        # If k is too small, use a default value
                        if k_eigengap < 2:
                            k_eigengap = min(5, adj_matrix.shape[0] // 10)
                            print(f"Using default k={k_eigengap} as optimal k (eigen-gap) was too small")

                        # 2. Elbow method for mini-batch k-means
                        print(f"Computing optimal k using elbow method for '{motif_name}'...")
                        elbow_start_time = time.time()
                        k_elbow = motifs.compute_optimal_k_elbow(adj_matrix)
                        elbow_time = time.time() - elbow_start_time
                        print(f"Optimal number of clusters (elbow) for '{motif_name}': {k_elbow}")
                        result['Elbow_Time'] = elbow_time

                        # If k is too small, use a default value
                        if k_elbow < 2:
                            k_elbow = min(5, adj_matrix.shape[0] // 10)
                            print(f"Using default k={k_elbow} as optimal k (elbow) was too small")

                        # 3. Silhouette method for GMM
                        print(f"Computing optimal k using silhouette method for '{motif_name}'...")
                        silhouette_start_time = time.time()
                        k_silhouette = motifs.compute_optimal_k_silhouette(adj_matrix)
                        silhouette_time = time.time() - silhouette_start_time
                        print(f"Optimal number of clusters (silhouette) for '{motif_name}': {k_silhouette}")
                        result['Silhouette_Time'] = silhouette_time

                        # If k is too small, use a default value
                        if k_silhouette < 2:
                            k_silhouette = min(5, adj_matrix.shape[0] // 10)
                            print(f"Using default k={k_silhouette} as optimal k (silhouette) was too small")

                    # Apply all clustering algorithms with all optimal k values
                    print(f"Applying clustering algorithms for '{motif_name}'...")

                    # Store all optimal k values
                    optimal_ks = {
                        'eigen-gap': k_eigengap,
                        'elbow': k_elbow,
                        'silhouette': k_silhouette
                    }

                    # Group k methods by their k values to avoid redundant computations
                    k_value_to_methods = {}
                    for k_method, k_value in optimal_ks.items():
                        if k_value not in k_value_to_methods:
                            k_value_to_methods[k_value] = []
                        k_value_to_methods[k_value].append(k_method)

                    # For each unique k value
                    for k_value, k_methods in k_value_to_methods.items():
                        print(f"Running clustering algorithms for '{motif_name}' with k={k_value} (methods: {', '.join(k_methods)})...")

                        # 1. K-means clustering (run once per unique k value)
                        print(f"Running K-means clustering for '{motif_name}' with k={k_value}...")
                        kmeans_start_time = time.time()
                        kmeans_clusters = motifs.compute_kmeans_clusters_simple(adj_matrix, k_value)
                        kmeans_time = time.time() - kmeans_start_time
                        kmeans_cluster_sizes = [len(cluster) for cluster in kmeans_clusters]

                        # Store results for all methods that use this k value
                        for k_method in k_methods:
                            result[f'KMeans_{k_method}_Clusters'] = len(kmeans_clusters)
                            result[f'KMeans_{k_method}_Cluster_Sizes'] = kmeans_cluster_sizes
                            result[f'KMeans_{k_method}_Time'] = kmeans_time

                        # 2. Mini-batch K-means clustering (run once per unique k value)
                        print(f"Running Mini-batch K-means clustering for '{motif_name}' with k={k_value}...")
                        minibatch_start_time = time.time()
                        minibatch_clusters = motifs.compute_minibatch_kmeans_clusters_simple(adj_matrix, k_value)
                        minibatch_time = time.time() - minibatch_start_time
                        minibatch_cluster_sizes = [len(cluster) for cluster in minibatch_clusters]

                        # Store results for all methods that use this k value
                        for k_method in k_methods:
                            result[f'MiniBatch_{k_method}_Clusters'] = len(minibatch_clusters)
                            result[f'MiniBatch_{k_method}_Cluster_Sizes'] = minibatch_cluster_sizes
                            result[f'MiniBatch_{k_method}_Time'] = minibatch_time

                        # 3. GMM clustering (run once per unique k value)
                        print(f"Running GMM clustering for '{motif_name}' with k={k_value}...")
                        gmm_start_time = time.time()
                        gmm_clusters = motifs.compute_gmm_clusters_simple(adj_matrix, k_value)
                        gmm_time = time.time() - gmm_start_time
                        gmm_cluster_sizes = [len(cluster) for cluster in gmm_clusters]

                        # Store results for all methods that use this k value
                        for k_method in k_methods:
                            result[f'GMM_{k_method}_Clusters'] = len(gmm_clusters)
                            result[f'GMM_{k_method}_Cluster_Sizes'] = gmm_cluster_sizes
                            result[f'GMM_{k_method}_Time'] = gmm_time

                    # Store the original optimal k values for reference
                    if optimal_k is not None:
                        result['KMeans_OptimalK_Method'] = 'user-specified'
                        result['KMeans_OptimalK'] = optimal_k
                        result['MiniBatch_OptimalK_Method'] = 'user-specified'
                        result['MiniBatch_OptimalK'] = optimal_k
                        result['GMM_OptimalK_Method'] = 'user-specified'
                        result['GMM_OptimalK'] = optimal_k
                    else:
                        result['KMeans_OptimalK_Method'] = 'eigen-gap'
                        result['KMeans_OptimalK'] = k_eigengap
                        result['MiniBatch_OptimalK_Method'] = 'elbow'
                        result['MiniBatch_OptimalK'] = k_elbow
                        result['GMM_OptimalK_Method'] = 'silhouette'
                        result['GMM_OptimalK'] = k_silhouette

                    # 4. MAPPR clustering
                    print(f"Running MAPPR clustering for '{motif_name}'...")

                    # Find good seed nodes using combined and fiedler methods
                    print(f"Finding good seed nodes for MAPPR clustering using combined and fiedler methods...")

                    # Use the provided optimal_k or find the highest optimal k value among the three methods
                    if optimal_k is not None:
                        highest_k = optimal_k
                        print(f"Using provided optimal k value ({highest_k}) as the number of seed nodes")
                    else:
                        highest_k = max(k_eigengap, k_elbow, k_silhouette)
                        print(f"Using highest optimal k value ({highest_k}) as the number of seed nodes")

                    # Number of seed nodes to find
                    num_seed_nodes = highest_k

                    # Find good seed nodes using combined method
                    combined_seed_nodes = motifs.find_good_seed_nodes(G, adj_matrix, motif_name, num_seed_nodes, method='combined')
                    print(f"Found {len(combined_seed_nodes)} good seed nodes using combined method: {combined_seed_nodes}")

                    # Find good seed nodes using fiedler method
                    fiedler_seed_nodes = motifs.find_good_seed_nodes_fiedler(adj_matrix, num_seed_nodes)
                    print(f"Found {len(fiedler_seed_nodes)} good seed nodes using fiedler method: {fiedler_seed_nodes}")

                    # If seed_methods is provided, use those as seed node indices
                    # Otherwise, use the combined and fiedler seed nodes
                    if seed_methods:
                        print(f"Using specified seed methods: {seed_methods}")
                        mappr_start_time = time.time()

                        # Store results for each seed method
                        all_mappr_clusters = {}
                        all_mappr_cluster_sizes = {}

                        for method_name in seed_methods:
                            print(f"Running MAPPR with seed method: {method_name}")

                            # Use the method name as a simple way to select different seed nodes
                            if method_name == "first":
                                # Use the first few nodes as seeds
                                seed_nodes = list(range(min(5, adj_matrix.shape[0])))
                            elif method_name == "last":
                                # Use the last few nodes as seeds
                                seed_nodes = list(range(adj_matrix.shape[0] - min(5, adj_matrix.shape[0]), adj_matrix.shape[0]))
                            elif method_name == "middle":
                                # Use nodes from the middle of the matrix as seeds
                                middle = adj_matrix.shape[0] // 2
                                seed_nodes = list(range(middle - min(2, middle), middle + min(3, adj_matrix.shape[0] - middle)))
                            else:
                                # Default to using the first few nodes as seeds
                                seed_nodes = list(range(min(5, adj_matrix.shape[0])))

                            # Track time for this specific method
                            method_start_time = time.time()

                            # Compute MAPPR clusters
                            method_clusters = motifs.compute_mappr_cluster_simple(adj_matrix, seed_nodes)

                            # Calculate and store the time for this method
                            method_time = time.time() - method_start_time
                            result[f'MAPPR_{method_name}_Time'] = method_time

                            # Store results
                            all_mappr_clusters[method_name] = method_clusters
                            all_mappr_cluster_sizes[method_name] = [len(cluster) for cluster in method_clusters.values()]

                            # Store in result dictionary
                            result[f'MAPPR_{method_name}_Clusters'] = len(method_clusters)
                            result[f'MAPPR_{method_name}_Cluster_Sizes'] = [len(cluster) for cluster in method_clusters.values()]
                            result[f'MAPPR_{method_name}_Seeds'] = seed_nodes

                        mappr_time = time.time() - mappr_start_time
                        result['MAPPR_Time'] = mappr_time
                        result['MAPPR_Methods'] = seed_methods
                    else:
                        print("Using combined and fiedler seed nodes for MAPPR clustering")

                        # Use the combined and fiedler seed nodes
                        mappr_start_time = time.time()

                        # Store results for each seed method
                        all_mappr_clusters = {}
                        all_mappr_cluster_sizes = {}

                        # Run MAPPR with combined seed nodes
                        print(f"Running MAPPR with combined seed nodes: {combined_seed_nodes}")

                        # Track time for combined method
                        combined_start_time = time.time()
                        combined_clusters = motifs.compute_mappr_cluster_simple(adj_matrix, combined_seed_nodes)
                        combined_time = time.time() - combined_start_time

                        # Store time for combined method
                        result['MAPPR_combined_Time'] = combined_time

                        # Store results for combined method
                        all_mappr_clusters['combined'] = combined_clusters
                        all_mappr_cluster_sizes['combined'] = [len(cluster) for cluster in combined_clusters.values()]

                        # Store in result dictionary
                        result['MAPPR_combined_Clusters'] = len(combined_clusters)
                        result['MAPPR_combined_Cluster_Sizes'] = [len(cluster) for cluster in combined_clusters.values()]
                        result['MAPPR_combined_Seeds'] = combined_seed_nodes

                        # Run MAPPR with fiedler seed nodes
                        print(f"Running MAPPR with fiedler seed nodes: {fiedler_seed_nodes}")

                        # Track time for fiedler method
                        fiedler_start_time = time.time()
                        fiedler_clusters = motifs.compute_mappr_cluster_simple(adj_matrix, fiedler_seed_nodes)
                        fiedler_time = time.time() - fiedler_start_time

                        # Store time for fiedler method
                        result['MAPPR_fiedler_Time'] = fiedler_time

                        # Store results for fiedler method
                        all_mappr_clusters['fiedler'] = fiedler_clusters
                        all_mappr_cluster_sizes['fiedler'] = [len(cluster) for cluster in fiedler_clusters.values()]

                        # Store in result dictionary
                        result['MAPPR_fiedler_Clusters'] = len(fiedler_clusters)
                        result['MAPPR_fiedler_Cluster_Sizes'] = [len(cluster) for cluster in fiedler_clusters.values()]
                        result['MAPPR_fiedler_Seeds'] = fiedler_seed_nodes

                        mappr_time = time.time() - mappr_start_time
                        result['MAPPR_Time'] = mappr_time
                        result['MAPPR_Methods'] = ['combined', 'fiedler']

                except Exception as e:
                    print(f"Error during clustering for motif {motif_name}: {e}")
                    result['Clustering_Error'] = str(e)

            # Add the result to the results list
            results.append(result)

        except Exception as e:
            print(f"Error processing motif {motif_name}: {e}")
            results.append({
                'Motif': motif_name,
                'Occurrences': 'ERROR',
                'Time(s)': time.time() - start_time,
                'Error': str(e)
            })

    # Process cluster sizes to make them more readable
    for result in results:
        # Process cluster sizes for each clustering method and optimal k method
        for method in ['KMeans', 'MiniBatch', 'GMM', 'MAPPR']:
            # For MAPPR, handle both default and seed method versions
            if method == 'MAPPR':
                # Check if we're using seed methods
                if 'MAPPR_Methods' in result:
                    # Process each seed method
                    for seed_method in result['MAPPR_Methods']:
                        sizes_key = f'MAPPR_{seed_method}_Cluster_Sizes'
                        if sizes_key in result and result[sizes_key] != 'N/A':
                            # Convert string representation of list to actual list if needed
                            if isinstance(result[sizes_key], str) and result[sizes_key].startswith('['):
                                try:
                                    sizes = eval(result[sizes_key])
                                except:
                                    sizes = []
                            else:
                                sizes = result[sizes_key]

                            if sizes:
                                # Calculate statistics
                                min_size = min(sizes)
                                max_size = max(sizes)
                                avg_size = sum(sizes) / len(sizes)
                                total_nodes = sum(sizes)

                                # Create a more readable format with individual cluster sizes
                                result[sizes_key] = f"Total: {total_nodes}, Clusters: {len(sizes)}, Min: {min_size}, Max: {max_size}, Avg: {avg_size:.1f}, Sizes: {sizes}"
                else:
                    # Process the default MAPPR
                    sizes_key = f'{method}_Cluster_Sizes'
                    if sizes_key in result and result[sizes_key] != 'N/A':
                        # Convert string representation of list to actual list if needed
                        if isinstance(result[sizes_key], str) and result[sizes_key].startswith('['):
                            try:
                                sizes = eval(result[sizes_key])
                            except:
                                sizes = []
                        else:
                            sizes = result[sizes_key]

                        if sizes:
                            # Calculate statistics
                            min_size = min(sizes)
                            max_size = max(sizes)
                            avg_size = sum(sizes) / len(sizes)
                            total_nodes = sum(sizes)

                            # Create a more readable format with individual cluster sizes
                            result[sizes_key] = f"Total: {total_nodes}, Clusters: {len(sizes)}, Min: {min_size}, Max: {max_size}, Avg: {avg_size:.1f}, Sizes: {sizes}"
            else:
                # For other methods, process each optimal k method
                for k_method in ['eigen-gap', 'elbow', 'silhouette']:
                    sizes_key = f'{method}_{k_method}_Cluster_Sizes'
                    if sizes_key in result and result[sizes_key] != 'N/A':
                        # Convert string representation of list to actual list if needed
                        if isinstance(result[sizes_key], str) and result[sizes_key].startswith('['):
                            try:
                                sizes = eval(result[sizes_key])
                            except:
                                sizes = []
                        else:
                            sizes = result[sizes_key]

                        if sizes:
                            # Calculate statistics
                            min_size = min(sizes)
                            max_size = max(sizes)
                            avg_size = sum(sizes) / len(sizes)
                            total_nodes = sum(sizes)

                            # Create a more readable format with individual cluster sizes
                            result[sizes_key] = f"Total: {total_nodes}, Clusters: {len(sizes)}, Min: {min_size}, Max: {max_size}, Avg: {avg_size:.1f}, Sizes: {sizes}"

    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv(csv_file, index=False)
    print(f"\nAll motifs processed. Results saved to {csv_file}")

    # Also save a summary to the text file in a more readable format
    with open(summary_file, 'w') as f:
        # Write header information
        f.write(f"===============================================================================\n")
        f.write(f"                  MOTIF ANALYSIS SUMMARY FOR NETWORK: {output_name}\n")
        f.write(f"===============================================================================\n")
        f.write(f"Network file: {network_file}\n")
        if subfolder_name:
            f.write(f"Subfolder: {subfolder_name}\n")
        f.write(f"Number of nodes: {G.number_of_nodes()}\n")
        f.write(f"Number of edges: {G.number_of_edges()}\n")
        f.write(f"===============================================================================\n\n")

        # For each motif, write a detailed section
        for result in results:
            motif_name = result['Motif']
            f.write(f"-------------------------------------------------------------------------------\n")
            f.write(f"MOTIF: {motif_name.upper()}\n")
            f.write(f"-------------------------------------------------------------------------------\n")
            f.write(f"Occurrences: {result.get('Occurrences', 'N/A')}\n")
            f.write(f"Processing time: {result.get('Time(s)', 0):.2f} seconds\n\n")

            # Write clustering results if available
            f.write("CLUSTERING RESULTS:\n")
            f.write("-------------------------------------------------------------------------------\n")

            # Write optimal k values found by each method or user-specified value
            f.write("OPTIMAL K VALUES:\n")
            if result.get('KMeans_OptimalK_Method') == 'user-specified':
                f.write(f"   User-specified: {result.get('KMeans_OptimalK', 'N/A')}\n\n")
            else:
                f.write(f"   Eigen-gap: {result.get('KMeans_OptimalK', 'N/A')} (time: {result.get('EigenGap_Time', 0):.2f} seconds)\n")
                f.write(f"   Elbow: {result.get('MiniBatch_OptimalK', 'N/A')} (time: {result.get('Elbow_Time', 0):.2f} seconds)\n")
                f.write(f"   Silhouette: {result.get('GMM_OptimalK', 'N/A')} (time: {result.get('Silhouette_Time', 0):.2f} seconds)\n\n")

            # Define the methods and their display names
            methods = [
                ('KMeans', 'K-MEANS CLUSTERING'),
                ('MiniBatch', 'MINI-BATCH K-MEANS CLUSTERING'),
                ('GMM', 'GAUSSIAN MIXTURE MODEL (GMM) CLUSTERING')
            ]

            # Define the optimal k methods
            k_methods = ['eigen-gap', 'elbow', 'silhouette']

            # Counter for numbering the sections
            section_num = 1

            # For each clustering method
            for method_key, method_name in methods:
                for k_method in k_methods:
                    # Get the cluster count and sizes
                    clusters_key = f'{method_key}_{k_method}_Clusters'
                    sizes_key = f'{method_key}_{k_method}_Cluster_Sizes'

                    if clusters_key in result and sizes_key in result:
                        # Get the k value based on the method
                        k_value = None
                        if result.get('KMeans_OptimalK_Method') == 'user-specified':
                            # For user-specified k, use the same value for all methods
                            k_value = result.get('KMeans_OptimalK', 'N/A')
                            # Change the k_method to 'user-specified' for display
                            k_method = 'user-specified'
                        else:
                            # For calculated k values, use the appropriate value based on the method
                            if k_method == 'eigen-gap':
                                k_value = result.get('KMeans_OptimalK', 'N/A')
                            elif k_method == 'elbow':
                                k_value = result.get('MiniBatch_OptimalK', 'N/A')
                            elif k_method == 'silhouette':
                                k_value = result.get('GMM_OptimalK', 'N/A')

                        f.write(f"{section_num}. {method_name} (k method: {k_method})\n")
                        f.write(f"   Optimal K: {k_value}\n")
                        f.write(f"   Clusters: {result.get(clusters_key, 'N/A')}\n")
                        f.write(f"   Cluster sizes: {result.get(sizes_key, 'N/A')}\n")
                        f.write(f"   Processing time: {result.get(f'{method_key}_{k_method}_Time', 0):.2f} seconds\n\n")
                        section_num += 1

            # MAPPR
            # Check if we're using seed methods
            if 'MAPPR_Methods' in result:
                # Write a header for MAPPR
                f.write(f"{section_num}. MAPPR CLUSTERING\n")
                f.write(f"   Processing time: {result.get('MAPPR_Time', 0):.2f} seconds\n")
                f.write(f"   Seed methods: {', '.join(result['MAPPR_Methods'])}\n\n")

                # For each seed method, write a subsection
                for i, seed_method in enumerate(result['MAPPR_Methods']):
                    f.write(f"   {section_num}.{i+1} MAPPR with {seed_method} seeds\n")
                    f.write(f"      Clusters: {result.get(f'MAPPR_{seed_method}_Clusters', 'N/A')}\n")
                    f.write(f"      Cluster sizes: {result.get(f'MAPPR_{seed_method}_Cluster_Sizes', 'N/A')}\n")
                    f.write(f"      Seed nodes: {result.get(f'MAPPR_{seed_method}_Seeds', 'N/A')}\n")
                    f.write(f"      Processing time: {result.get(f'MAPPR_{seed_method}_Time', 0):.2f} seconds\n\n")

                section_num += 1
            else:
                # Write the default MAPPR section
                f.write(f"{section_num}. MAPPR CLUSTERING\n")
                f.write(f"   Processing time: {result.get('MAPPR_Time', 0):.2f} seconds\n\n")

                # Write subsection for combined method
                if 'MAPPR_combined_Clusters' in result:
                    f.write(f"   {section_num}.1 MAPPR with combined seeds\n")
                    f.write(f"      Clusters: {result.get('MAPPR_combined_Clusters', 'N/A')}\n")
                    f.write(f"      Cluster sizes: {result.get('MAPPR_combined_Cluster_Sizes', 'N/A')}\n")
                    f.write(f"      Seed nodes: {result.get('MAPPR_combined_Seeds', 'N/A')}\n")
                    f.write(f"      Processing time: {result.get('MAPPR_combined_Time', 0):.2f} seconds\n\n")

                # Write subsection for fiedler method
                if 'MAPPR_fiedler_Clusters' in result:
                    f.write(f"   {section_num}.2 MAPPR with fiedler seeds\n")
                    f.write(f"      Clusters: {result.get('MAPPR_fiedler_Clusters', 'N/A')}\n")
                    f.write(f"      Cluster sizes: {result.get('MAPPR_fiedler_Cluster_Sizes', 'N/A')}\n")
                    f.write(f"      Seed nodes: {result.get('MAPPR_fiedler_Seeds', 'N/A')}\n")
                    f.write(f"      Processing time: {result.get('MAPPR_fiedler_Time', 0):.2f} seconds\n\n")

                section_num += 1

            # Add error information if available
            if 'Clustering_Error' in result:
                f.write(f"ERROR: {result['Clustering_Error']}\n\n")

            f.write("\n")


def find_edge_files(root_dir):
    """
    Find all .edge files in the given directory and its subdirectories.

    Args:
        root_dir: Path to the root directory to search in

    Returns:
        A list of tuples, each containing (file_path, subfolder_name)
    """
    edge_files = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.edge'):
                # Get the full path to the file
                file_path = os.path.join(dirpath, filename)

                # Extract the subfolder name (e.g., "aged_with_dementia")
                # The structure is data/granger/100.0/aged_with_dementia/2-2000-1118W-01/graph_300_650_100.0.edge
                # We want to extract "aged_with_dementia" and "100.0" to create a unique name
                path_parts = dirpath.split(os.sep)

                # Find the index of "granger" in the path
                try:
                    granger_index = path_parts.index("granger")
                    # The subfolder name is two levels below "granger"
                    if len(path_parts) > granger_index + 2:
                        threshold = path_parts[granger_index + 1]  # e.g., "100.0"
                        category = path_parts[granger_index + 2]   # e.g., "aged_with_dementia"
                        subject = path_parts[granger_index + 3] if len(path_parts) > granger_index + 3 else ""  # e.g., "2-2000-1118W-01"

                        # Create a unique name based on the subfolder structure
                        subfolder_name = f"{threshold}_{category}_{subject}"
                    else:
                        # If the path doesn't have the expected structure, use the parent directory name
                        subfolder_name = os.path.basename(dirpath)
                except ValueError:
                    # If "granger" is not in the path, use the parent directory name
                    subfolder_name = os.path.basename(dirpath)

                edge_files.append((file_path, subfolder_name))

    return edge_files

def main():
    """Parse command line arguments and run the analysis."""
    parser = argparse.ArgumentParser(description='Run all motifs on a given network.')
    parser.add_argument('--network', type=str, help='Path to the network file')
    parser.add_argument('--seed-method', type=str, nargs='+',
                        help='Seed methods to use for MAPPR clustering. Available methods: "first", "last", "middle". If not specified, uses "combined" and "fiedler" methods to find good seed nodes.')
    parser.add_argument('--optimal-k', type=int, 
                        help='Use the specified value for k instead of calculating optimal k. This value will be used for k-means, mini-batch k-means, GMM, and as the number of seed nodes for MAPPR.')
    parser.add_argument('--granger-dir', action='store_true',
                        help='Process all .edge files in the data/granger directory and its subdirectories. Summary files will be saved with unique names based on the subfolder structure.')

    args = parser.parse_args()

    if args.granger_dir:
        # Process all .edge files in the data/granger directory and its subdirectories
        granger_dir = os.path.join("../data", "granger")
        edge_files = find_edge_files(granger_dir)

        if not edge_files:
            print(f"No .edge files found in {granger_dir}")
            return

        print(f"Found {len(edge_files)} .edge files in {granger_dir}")

        for file_path, subfolder_name in edge_files:
            print(f"\nProcessing {file_path} (subfolder: {subfolder_name})...")
            run_all_motifs(network_file=file_path, seed_methods=args.seed_method, subfolder_name=subfolder_name, optimal_k=args.optimal_k)
    elif args.network:
        # Process a single network file
        run_all_motifs(network_file=args.network, seed_methods=args.seed_method, optimal_k=args.optimal_k)
    else:
        parser.error("Either --network or --granger-dir must be specified")


if __name__ == "__main__":
    main()
