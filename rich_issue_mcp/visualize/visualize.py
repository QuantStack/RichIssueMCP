"""Visualization module for Rich Issue MCP."""

import gzip
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors


def load_enriched_issues(file_path: Path) -> List[Dict[str, Any]]:
    """Load enriched issues from gzipped JSON file."""
    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        return json.load(f)


def extract_embeddings(issues: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[str]]:
    """Extract embeddings and issue IDs from enriched issues."""
    embeddings = []
    issue_ids = []
    
    for issue in issues:
        if "embedding" in issue and issue["embedding"]:
            embeddings.append(issue["embedding"])
            issue_ids.append(str(issue["number"]))
    
    return np.array(embeddings), issue_ids


def compute_tsne(embeddings: np.ndarray, random_state: int = 42) -> np.ndarray:
    """Compute T-SNE projection of embeddings to 2D."""
    tsne = TSNE(n_components=2, random_state=random_state, perplexity=min(30, len(embeddings) - 1))
    return tsne.fit_transform(embeddings)


def find_nearest_neighbors(embeddings: np.ndarray, k: int = 4) -> List[List[int]]:
    """Find k nearest neighbors for each embedding."""
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)
    
    # Remove self (first neighbor) and return indices
    return [list(neighbors[1:]) for neighbors in indices]


def extract_cross_references(issues: List[Dict[str, Any]]) -> Dict[str, Set[str]]:
    """Extract cross-reference relationships between issues."""
    cross_refs = {}
    issue_numbers = {str(issue["number"]) for issue in issues}
    
    for issue in issues:
        issue_id = str(issue["number"])
        cross_refs[issue_id] = set()
        
        # Check cross_references field
        if "cross_references" in issue and issue["cross_references"]:
            for ref in issue["cross_references"]:
                if ref.get("type") == "issue" and "number" in ref:
                    ref_id = str(ref["number"])
                    if ref_id in issue_numbers:
                        cross_refs[issue_id].add(ref_id)
    
    return cross_refs


def write_graphml(
    issue_ids: List[str],
    tsne_coords: np.ndarray,
    nearest_neighbors: List[List[int]],
    cross_references: Dict[str, Set[str]],
    output_path: Path,
    issues: List[Dict[str, Any]],
    scale: float = 1.0
) -> None:
    """Write GraphML file with issue network."""
    # Create root element
    root = ET.Element("graphml")
    root.set("xmlns", "http://graphml.graphdrawing.org/xmlns")
    root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
    root.set("xsi:schemaLocation", "http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd")
    
    # Define node attributes
    node_attrs = [
        ("d0", "x", "double"),
        ("d1", "y", "double"),
        ("d2", "label", "string"),
        ("d3", "title", "string"),
        ("d4", "state", "string"),
        ("d5", "reactions", "int"),
        ("d6", "comments", "int"),
    ]
    
    for attr_id, attr_name, attr_type in node_attrs:
        key = ET.SubElement(root, "key")
        key.set("id", attr_id)
        key.set("for", "node")
        key.set("attr.name", attr_name)
        key.set("attr.type", attr_type)
    
    # Define edge attributes
    edge_attrs = [
        ("d7", "edge_type", "string"),
        ("d8", "weight", "double"),
    ]
    
    for attr_id, attr_name, attr_type in edge_attrs:
        key = ET.SubElement(root, "key")
        key.set("id", attr_id)
        key.set("for", "edge")
        key.set("attr.name", attr_name)
        key.set("attr.type", attr_type)
    
    # Create graph
    graph = ET.SubElement(root, "graph")
    graph.set("id", "IssueNetwork")
    graph.set("edgedefault", "undirected")
    
    # Create issue lookup
    issue_lookup = {str(issue["number"]): issue for issue in issues}
    
    # Add nodes
    for i, issue_id in enumerate(issue_ids):
        node = ET.SubElement(graph, "node")
        node.set("id", issue_id)
        
        # Add coordinates (scaled)
        x_data = ET.SubElement(node, "data")
        x_data.set("key", "d0")
        x_data.text = str(tsne_coords[i, 0] * scale)
        
        y_data = ET.SubElement(node, "data")
        y_data.set("key", "d1")
        y_data.text = str(tsne_coords[i, 1] * scale)
        
        # Add issue metadata
        issue = issue_lookup.get(issue_id, {})
        
        label_data = ET.SubElement(node, "data")
        label_data.set("key", "d2")
        label_data.text = f"{issue_id}: {issue.get('title', '')}"
        
        title_data = ET.SubElement(node, "data")
        title_data.set("key", "d3")
        title_data.text = issue.get("title", "")
        
        state_data = ET.SubElement(node, "data")
        state_data.set("key", "d4")
        state_data.text = issue.get("state", "unknown")
        
        reactions_data = ET.SubElement(node, "data")
        reactions_data.set("key", "d5")
        reactions_data.text = str(issue.get("reactions", {}).get("total_count", 0))
        
        comments_data = ET.SubElement(node, "data")
        comments_data.set("key", "d6")
        comments_data.text = str(issue.get("number_of_comments", 0))
    
    # Add nearest neighbor edges
    edge_id = 0
    for i, neighbors in enumerate(nearest_neighbors):
        source_id = issue_ids[i]
        for neighbor_idx in neighbors:
            target_id = issue_ids[neighbor_idx]
            
            edge = ET.SubElement(graph, "edge")
            edge.set("id", f"e{edge_id}")
            edge.set("source", source_id)
            edge.set("target", target_id)
            
            type_data = ET.SubElement(edge, "data")
            type_data.set("key", "d7")
            type_data.text = "nearest_neighbor"
            
            weight_data = ET.SubElement(edge, "data")
            weight_data.set("key", "d8")
            weight_data.text = "1.0"
            
            edge_id += 1
    
    # Add cross-reference edges
    for source_id, refs in cross_references.items():
        if source_id in issue_ids:
            for target_id in refs:
                if target_id in issue_ids:
                    edge = ET.SubElement(graph, "edge")
                    edge.set("id", f"e{edge_id}")
                    edge.set("source", source_id)
                    edge.set("target", target_id)
                    
                    type_data = ET.SubElement(edge, "data")
                    type_data.set("key", "d7")
                    type_data.text = "cross_reference"
                    
                    weight_data = ET.SubElement(edge, "data")
                    weight_data.set("key", "d8")
                    weight_data.text = "2.0"
                    
                    edge_id += 1
    
    # Write to file
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="utf-8", xml_declaration=True)


def visualize_issues(input_path: Path, output_dir: Path, scale: float = 1.0) -> None:
    """Create T-SNE visualization and GraphML network from enriched issues."""
    print(f"🔍 Loading enriched issues from {input_path}...")
    issues = load_enriched_issues(input_path)
    
    print(f"📊 Extracting embeddings from {len(issues)} issues...")
    embeddings, issue_ids = extract_embeddings(issues)
    
    if len(embeddings) == 0:
        raise ValueError("No embeddings found in input file")
    
    print(f"🧮 Computing T-SNE projection for {len(embeddings)} embeddings...")
    tsne_coords = compute_tsne(embeddings)
    
    print("🔗 Finding 4 nearest neighbors for each issue...")
    nearest_neighbors = find_nearest_neighbors(embeddings, k=4)
    
    print("🔍 Extracting cross-reference relationships...")
    cross_references = extract_cross_references(issues)
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Write GraphML file
    graphml_path = output_dir / "issue_network.graphml"
    print(f"💾 Writing GraphML network to {graphml_path}...")
    write_graphml(issue_ids, tsne_coords, nearest_neighbors, cross_references, graphml_path, issues, scale)
    
    # Write T-SNE coordinates as JSON
    tsne_path = output_dir / "tsne_coordinates.json"
    print(f"💾 Writing T-SNE coordinates to {tsne_path}...")
    
    tsne_data = {
        "coordinates": [
            {"issue_id": issue_id, "x": float(coord[0]), "y": float(coord[1])}
            for issue_id, coord in zip(issue_ids, tsne_coords)
        ]
    }
    
    with open(tsne_path, "w") as f:
        json.dump(tsne_data, f, indent=2)
    
    print(f"✅ Visualization complete!")
    print(f"   - GraphML network: {graphml_path}")
    print(f"   - T-SNE coordinates: {tsne_path}")
    print(f"   - Issues processed: {len(issue_ids)}")
    
    # Print edge statistics
    nn_edges = len(issue_ids) * 4  # Each issue has 4 nearest neighbors
    cr_edges = sum(len(refs) for refs in cross_references.values())
    print(f"   - Nearest neighbor edges: {nn_edges}")
    print(f"   - Cross-reference edges: {cr_edges}")