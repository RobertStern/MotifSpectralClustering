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
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh
from numpy.linalg import eigh
from scipy.sparse import csr_matrix, lil_matrix


def load_adjacency_matrix_from_csv(file_path: str) -> Tuple[np.ndarray, Optional[dict]]:
    """
    Load an adjacency matrix from a CSV file.

    Args:
        file_path: Path to the CSV file containing the adjacency matrix

    Returns:
        A tuple containing:
        - The adjacency matrix as a numpy array
        - A dictionary mapping node indices to node IDs
    """
    try:
        # Load the CSV file into a pandas DataFrame
        print(f"Loading CSV file {file_path}...")
        df = pd.read_csv(file_path, index_col=0)

        # Extract the node names from the DataFrame
        node_names = list(df.index)

        # Create a mapping from indices to node names
        node_mapping = {i: node for i, node in enumerate(node_names)}

        # Convert the DataFrame to a numpy array
        adjacency_matrix = df.values

        # Ensure the adjacency matrix is symmetric (undirected graph)
        if adjacency_matrix.shape[0] != adjacency_matrix.shape[1]:
            print(f"Warning: Adjacency matrix is not square ({adjacency_matrix.shape})")

        # Convert to float for better numerical stability
        adjacency_matrix = adjacency_matrix.astype(float)

        # Check if the matrix is connected (has non-zero values)
        if np.sum(adjacency_matrix) == 0:
            print("Warning: Adjacency matrix has all zero values. The graph may be disconnected.")

        print(f"Loaded adjacency matrix with shape {adjacency_matrix.shape}")
        print(f"Node mapping: {len(node_mapping)} nodes")

        return adjacency_matrix, node_mapping
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        raise


def compute_laplacian(adjacency_matrix: np.ndarray, normalized: bool = True) -> Union[np.ndarray, sp.spmatrix]:
    """
    Compute the Laplacian matrix from an adjacency matrix.

    Args:
        adjacency_matrix: The adjacency matrix of the graph
        normalized: Whether to compute the normalized Laplacian (default: True)

    Returns:
        The Laplacian matrix (normalized or unnormalized)
    """
    # Check if the input is a sparse matrix
    is_sparse = sp.issparse(adjacency_matrix)

    # Compute the degree matrix
    if is_sparse:
        degrees = np.array(adjacency_matrix.sum(axis=1)).flatten()
        D = sp.diags(degrees)
    else:
        degrees = np.sum(adjacency_matrix, axis=1)
        D = np.diag(degrees)

    # Compute the Laplacian matrix
    if is_sparse:
        L = D - adjacency_matrix
    else:
        L = D - adjacency_matrix

    # Compute the normalized Laplacian if requested
    if normalized:
        if is_sparse:
            # Avoid division by zero
            inv_sqrt_degrees = np.zeros_like(degrees)
            non_zero = degrees > 0
            inv_sqrt_degrees[non_zero] = 1.0 / np.sqrt(degrees[non_zero])
            D_inv_sqrt = sp.diags(inv_sqrt_degrees)
            L_norm = D_inv_sqrt @ L @ D_inv_sqrt
        else:
            # Avoid division by zero
            inv_sqrt_degrees = np.zeros_like(degrees)
            non_zero = degrees > 0
            inv_sqrt_degrees[non_zero] = 1.0 / np.sqrt(degrees[non_zero])
            D_inv_sqrt = np.diag(inv_sqrt_degrees)
            L_norm = D_inv_sqrt @ L @ D_inv_sqrt
        return L_norm
    else:
        return L


def compute_fiedler_vector(adjacency_matrix: np.ndarray, normalized: bool = True) -> Tuple[np.ndarray, float]:
    """
    Compute the Fiedler vector from an adjacency matrix.

    The Fiedler vector is the eigenvector corresponding to the second smallest
    eigenvalue of the Laplacian matrix. It can be used for spectral clustering
    and graph partitioning.

    Args:
        adjacency_matrix: The adjacency matrix of the graph
        normalized: Whether to use the normalized Laplacian (default: True)

    Returns:
        A tuple containing:
        - The Fiedler vector (eigenvector corresponding to the second smallest eigenvalue)
        - The second smallest eigenvalue (algebraic connectivity)
    """
    # Compute the Laplacian matrix
    L = compute_laplacian(adjacency_matrix, normalized)
    print("Laplacian")
    print(L)
    # Check if the Laplacian is sparse
    is_sparse = sp.issparse(L)

    # Compute the eigenvalues and eigenvectors
    if is_sparse:
        # For sparse matrices, use scipy.sparse.linalg.eigsh to compute a few eigenvalues/vectors
        eigenvalues, eigenvectors = spla.eigsh(L, k=2, which='SM')
    else:
        # For dense matrices, use numpy.linalg.eigh
        eigenvalues, eigenvectors = np.linalg.eigh(L)

    # Sort eigenvalues and corresponding eigenvectors
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # The Fiedler vector is the eigenvector corresponding to the second smallest eigenvalue
    # The smallest eigenvalue should be close to zero for a connected graph
    fiedler_value = eigenvalues[1]
    fiedler_vector = eigenvectors[:, 1]
    print(fiedler_vector)
    return fiedler_vector, fiedler_value

def approximate_pagerank(adj_matrix, alpha, epsilon, seed_vector):
    """
    Compute the approximate personalized PageRank vector using the push method.

    Args:
        adj_matrix: Adjacency matrix (scipy sparse matrix or numpy array)
        alpha: Teleportation parameter (typically 0.15)
        epsilon: Approximation parameter
        seed_vector: Personalization vector (typically a one-hot vector for a single node)

    Returns:
        p: Approximate PageRank vector
        r: Residual vector
    """
    n = adj_matrix.shape[0]

    # Initialize the PageRank vector p and residual vector r
    p = np.zeros(n)
    r = seed_vector.copy()

    # Compute the degree of each node
    if sp.issparse(adj_matrix):
        degrees = np.array(adj_matrix.sum(axis=1)).flatten()
    else:
        degrees = adj_matrix.sum(axis=1)

    # Create a queue of nodes with residual above the threshold
    queue = [i for i in range(n) if r[i] > epsilon * degrees[i]]

    # Process nodes until the queue is empty
    while queue:
        u = queue.pop(0)

        # Skip isolated nodes
        if degrees[u] == 0:
            continue

        # Push operation
        push_value = r[u] - 0.5 * epsilon * degrees[u]
        p[u] += push_value
        r[u] = 0.5 * epsilon * degrees[u]

        # Distribute residual to neighbors
        if sp.issparse(adj_matrix):
            neighbors = adj_matrix[u].nonzero()[1]
            for v in neighbors:
                old_residual = r[v]
                r[v] += (1 - alpha) * push_value * adj_matrix[u, v] / degrees[u]

                # Add to queue if residual exceeds threshold
                if old_residual <= epsilon * degrees[v] and r[v] > epsilon * degrees[v] and v not in queue:
                    queue.append(v)
        else:
            neighbors = np.nonzero(adj_matrix[u])[0]
            for v in neighbors:
                old_residual = r[v]
                r[v] += (1 - alpha) * push_value * adj_matrix[u, v] / degrees[u]

                # Add to queue if residual exceeds threshold
                if old_residual <= epsilon * degrees[v] and r[v] > epsilon * degrees[v] and v not in queue:
                    queue.append(v)

    return p, r

def sweep_cut(adj_matrix, p, degrees=None):
    """
    Perform a sweep cut to find the best conductance.

    Args:
        adj_matrix: Adjacency matrix (scipy sparse matrix or numpy array)
        p: PageRank vector
        degrees: Node degrees (if None, computed from adj_matrix)

    Returns:
        best_set: Set of nodes with the best conductance
        best_conductance: The conductance value of the best set
    """
    n = len(p)

    # Compute degrees if not provided
    if degrees is None:
        if sp.issparse(adj_matrix):
            degrees = np.array(adj_matrix.sum(axis=1)).flatten()
        else:
            degrees = adj_matrix.sum(axis=1)
    # Sort nodes by PageRank divided by degree (in descending order)
    # For isolated nodes (degree=0), use -inf to place them at the end
    sorted_nodes = sorted(range(n), key=lambda x: -p[x]/degrees[x] if degrees[x] > 0 else -float('inf'))

    # Initialize variables for tracking the best cut
    best_conductance = float('inf')
    best_set = []
    vol_S = 0
    cut = 0
    vol_total = sum(degrees)

    # Sweep through the sorted nodes
    for i, node in enumerate(sorted_nodes):
        # Skip isolated nodes
        if degrees[node] == 0:
            continue

        # Update volume of set S
        vol_S += degrees[node]

        # Update cut value
        if sp.issparse(adj_matrix):
            # For sparse matrices
            row = adj_matrix[node].toarray().flatten()
            for j in range(i+1, n):
                next_node = sorted_nodes[j]
                if row[next_node] > 0:
                    cut += row[next_node]
        else:
            # For dense matrices
            for j in range(i+1, n):
                next_node = sorted_nodes[j]
                cut += adj_matrix[node, next_node]

        # Skip the first few nodes to avoid tiny clusters
        if i < 5:
            continue

        # Skip if volume is too small or too large
        vol_complement = vol_total - vol_S
        if vol_S == 0 or vol_complement == 0:
            continue

        # Compute conductance
        conductance = cut / min(vol_S, vol_complement)

        # Update best cut if this one is better
        if conductance < best_conductance:
            best_conductance = conductance
            best_set = sorted_nodes[:i+1]

    return best_set, best_conductance

def mappr(adj_matrix, seed_nodes, alpha=0.15, epsilon=1e-6):
    """
    Perform Motif-based Approximate Personalized PageRank for local clustering.

    Args:
        adj_matrix: Adjacency matrix (scipy sparse matrix or numpy array)
        seed_nodes: Index or list of indices of the seed node(s)
        alpha: Teleportation parameter (default: 0.15)
        epsilon: Approximation parameter (default: 1e-6)

    Returns:
        cluster: Set of nodes in the local cluster
        conductance: Conductance of the cluster
        p: PageRank vector
    """
    n = adj_matrix.shape[0]

    # Convert single seed node to list if necessary
    if not isinstance(seed_nodes, list):
        seed_nodes = [seed_nodes]

    # Create seed vector (personalization vector)
    seed_vector = np.zeros(n)
    for seed_node in seed_nodes:
        seed_vector[seed_node] = 1.0 / len(seed_nodes)  # Equal weight for each seed node

    # Compute approximate PageRank
    p, r = approximate_pagerank(adj_matrix, alpha, epsilon, seed_vector)
    # Compute degrees
    if sp.issparse(adj_matrix):
        degrees = np.array(adj_matrix.sum(axis=1)).flatten()
    else:
        degrees = adj_matrix.sum(axis=1)

    # Perform sweep cut to find the best cluster
    cluster, conductance = sweep_cut(adj_matrix, p, degrees)

    return cluster, conductance, p
