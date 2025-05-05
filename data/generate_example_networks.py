#!/usr/bin/env python3
"""
Generate Example Networks

This script generates example network files of different sizes for testing the network_loader.py script.
It creates three files:
1. small_network.txt - A small network with about 10 nodes and 15 edges
2. medium_network.txt - A medium-sized network with about 100 nodes and 300 edges
3. large_network.txt - A larger network with about 1000 nodes and 5000 edges
"""

import random
import os

def generate_network_file(filename, num_nodes, num_edges, add_header=True):
    """
    Generate a network file with the specified number of nodes and edges.

    Args:
        filename: Name of the output file
        num_nodes: Number of nodes in the network
        num_edges: Number of edges in the network
        add_header: Whether to add a header comment to the file
    """
    # Create a set to track edges (to avoid duplicates)
    edges = set()

    # Generate random edges until we have the desired number
    while len(edges) < num_edges:
        # Generate a random edge
        source = random.randint(0, num_nodes - 1)
        target = random.randint(0, num_nodes - 1)

        # Skip self-loops
        if source == target:
            continue

        # Add the edge (in canonical order to avoid duplicates)
        edge = tuple(sorted([source, target]))
        edges.add(edge)

    # Write the edges to the file
    with open(filename, 'w') as f:
        if add_header:
            f.write(f"# Network with {num_nodes} nodes and {num_edges} edges\n")
            f.write("# Format: source_node target_node\n")

        for source, target in edges:
            f.write(f"{source} {target}\n")

    print(f"Generated {filename} with {num_nodes} nodes and {num_edges} edges")




def main():
    """Generate example network files of different sizes."""
    # Create the data/examples directory
    examples_dir = os.path.join("examples")
    os.makedirs(examples_dir, exist_ok=True)

    # Also create the examples directory for backward compatibility
    os.makedirs("examples", exist_ok=True)

    # Generate a small network (10 nodes, 15 edges)
    generate_network_file(os.path.join(examples_dir, "small_network.txt"), 10, 15)

    # Generate a medium network (100 nodes, 300 edges)
    generate_network_file(os.path.join(examples_dir, "medium_network.txt"), 100, 300)

    # Generate a large network (1000 nodes, 5000 edges)
    generate_network_file(os.path.join(examples_dir, "large_network.txt"), 1000, 5000)

    generate_network_file(os.path.join(examples_dir, "very_large_network.txt"), 10000, 50000)

    print("All example networks generated successfully in the data/examples directory.")


if __name__ == "__main__":
    main()
