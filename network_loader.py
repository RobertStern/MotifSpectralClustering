#!/usr/bin/env python3
"""
Network Loader

This script provides functionality to load networks from a two-column format file,
where each row represents an edge between two nodes. The script can save
the network as a PNG or GEXF file.

Usage:
    python network_loader.py input_file [--save-png [output_file] | --save-gexf [output_file]]

If no output_file is provided, the input filename will be used (with appropriate extension).
For GEXF files, the .gexf extension is automatically added if not present.

Example:
    python network_loader.py data/examples/small_network.txt --save-png
    python network_loader.py data/examples/medium_network.txt --save-png custom_name.png
    python network_loader.py data/examples/large_network.txt --save-gexf
    python network_loader.py data/examples/large_network.txt --save-gexf custom_name
"""

import networkx as nx
import matplotlib.pyplot as plt
import argparse
import os
import time
from typing import Optional, Tuple


def load_network_from_file(file_path: str) -> nx.Graph:
    """
    Load a network from a file with two columns representing edges.

    Args:
        file_path: Path to the input file

    Returns:
        A NetworkX graph object

    Raises:
        FileNotFoundError: If the input file does not exist
        ValueError: If the file format is invalid
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file '{file_path}' not found")

    # Create an empty undirected graph
    G = nx.Graph()

    # Read the file and add edges
    print(f"Loading network from {file_path}...")
    start_time = time.time()

    try:
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                # Skip empty lines and comments
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                # Parse the line to get the two nodes
                parts = line.split()
                if len(parts) < 2:
                    raise ValueError(f"Line {line_num} does not contain two nodes: {line}")

                # Extract the nodes and add the edge
                source, target = parts[0], parts[1]
                G.add_edge(source, target)

                # Print progress for large files
                if line_num % 100000 == 0:
                    print(f"Processed {line_num} lines...")
    except Exception as e:
        raise ValueError(f"Error reading file: {str(e)}")

    elapsed_time = time.time() - start_time
    print(f"Network loaded in {elapsed_time:.2f} seconds")
    print(f"Network has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    return G


def save_as_png(G: nx.Graph, output_file: str, title: str = "Network Visualization", 
                figsize: Tuple[int, int] = (12, 10),
                node_size: int = 500, node_color: str = 'skyblue',
                edge_color: str = 'gray', alpha: float = 0.7,
                input_file: Optional[str] = None) -> None:
    """
    Save a network as a PNG file.

    Args:
        G: NetworkX graph to save
        output_file: Path to the output PNG file
        title: Title for the visualization
        figsize: Figure size as (width, height) in inches
        node_size: Size of nodes
        node_color: Color of nodes
        edge_color: Color of edges
        alpha: Transparency of nodes and edges
        input_file: Path to the input file (used for creating subfolder)
    """
    # If no directory is specified, save to the data/results directory
    if os.path.dirname(output_file) == '':
        # If input_file is provided, create a subfolder named after it
        if input_file:
            # Extract the base filename without directory and extension
            input_basename = os.path.basename(input_file)
            input_name, _ = os.path.splitext(input_basename)
            # Create a subfolder path
            output_file = os.path.join('data', 'results', input_name, output_file)
        else:
            output_file = os.path.join('data', 'results', output_file)

    print(f"Saving network to {output_file}...")
    start_time = time.time()

    # Check if the network is too large for visualization
    if G.number_of_nodes() > 1000:
        print("Warning: Network is large. Visualization may be slow and cluttered.")
        response = input("Do you want to continue? (y/n): ")
        if response.lower() != 'y':
            print("Visualization cancelled.")
            return

    # Create the figure
    plt.figure(figsize=figsize)
    plt.title(title)

    # Use a layout algorithm appropriate for the network size
    if G.number_of_nodes() <= 100:
        print("Using spring layout...")
        pos = nx.spring_layout(G, seed=42)
    else:
        print("Using kamada_kawai layout for better visualization of larger networks...")
        pos = nx.kamada_kawai_layout(G)

    # Draw the network with node labels
    nx.draw(G, pos, with_labels=True, node_size=node_size,
            node_color=node_color, edge_color=edge_color, alpha=alpha,
            font_weight='bold')

    elapsed_time = time.time() - start_time
    print(f"Visualization prepared in {elapsed_time:.2f} seconds")

    # Adjust layout
    plt.tight_layout()

    # Create the directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the figure
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()

    elapsed_time = time.time() - start_time
    print(f"Network saved in {elapsed_time:.2f} seconds")
    print(f"Network visualization saved to {output_file}")


def save_as_gexf(G: nx.Graph, output_file: str, input_file: Optional[str] = None) -> None:
    """
    Save the network as a GEXF file.

    Args:
        G: NetworkX graph to save
        output_file: Path to the output GEXF file
        input_file: Path to the input file (used for creating subfolder)
    """
    # If no directory is specified, save to the data/results directory
    if os.path.dirname(output_file) == '':
        # If input_file is provided, create a subfolder named after it
        if input_file:
            # Extract the base filename without directory and extension
            input_basename = os.path.basename(input_file)
            input_name, _ = os.path.splitext(input_basename)
            # Create a subfolder path
            output_file = os.path.join('data', 'results', input_name, output_file)
        else:
            output_file = os.path.join('data', 'results', output_file)

    # Ensure the file has a .gexf extension
    if not output_file.lower().endswith('.gexf'):
        output_file += '.gexf'

    print(f"Saving network to {output_file}...")
    start_time = time.time()

    # Create the directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the network
    nx.write_gexf(G, output_file)

    elapsed_time = time.time() - start_time
    print(f"Network saved in {elapsed_time:.2f} seconds")
    print(f"Network saved to {output_file}")


def main():
    """Main function to parse arguments and execute the script."""
    parser = argparse.ArgumentParser(description='Load a network from a two-column file and save as PNG or GEXF.')
    parser.add_argument('input_file', help='Path to the input file with two columns representing edges')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--save-png', nargs='?', const='', metavar='OUTPUT_FILE', 
                       help='Save the network as a PNG file. If no filename is provided, uses the input filename.')
    group.add_argument('--save-gexf', nargs='?', const='', metavar='OUTPUT_FILE', 
                       help='Save the network as a GEXF file. If no filename is provided, uses the input filename.')

    args = parser.parse_args()

    try:
        # Load the network
        G = load_network_from_file(args.input_file)

        # Get the base filename from the input file (without directory and extension)
        input_basename = os.path.basename(args.input_file)
        input_name, _ = os.path.splitext(input_basename)

        # Process according to the chosen action
        if args.save_png is not None:
            # If no output file is specified, use the input filename
            output_file = args.save_png if args.save_png else f"{input_name}.png"
            save_as_png(G, output_file, title=f"Network from {args.input_file}", input_file=args.input_file)
        elif args.save_gexf is not None:
            # If no output file is specified, use the input filename
            output_file = args.save_gexf if args.save_gexf else input_name
            save_as_gexf(G, output_file, input_file=args.input_file)

    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
