#!/usr/bin/env python3
"""Main enrichment module for processing raw GitHub issues."""

import hashlib
import json
import re
from datetime import datetime
from typing import Any

import hdbscan
import numpy as np
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from rich_issue_mcp.config import get_cache


def _get_cache_key(content: str, model: str) -> str:
    """Generate cache key from content and model hash."""
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return f"{model}:{content_hash}"


def _sanitize_content(content: str) -> str:
    """Sanitize content for API requests using built-in methods."""
    # Remove null bytes and other problematic characters
    content = content.replace("\x00", "").replace("\ufeff", "")

    # Replace control characters (except common whitespace) with spaces
    sanitized = "".join(c if c in "\t\n\r" or ord(c) >= 32 else " " for c in content)

    # Clean up whitespace
    sanitized = " ".join(sanitized.split())

    # Truncate if too long
    if len(sanitized) > 50000:
        sanitized = sanitized[:50000] + "..."

    return sanitized


def get_mistral_embedding(
    content: str, api_key: str, model: str = "mistral-embed"
) -> list[float] | None:
    """Get embedding from Mistral API with caching."""
    if not content.strip():
        return None

    # Sanitize content for API
    sanitized_content = _sanitize_content(content)
    if not sanitized_content.strip():
        return None

    # Check cache first (use original content for cache key to avoid duplicates)
    cache_key = _get_cache_key(content, model)
    cache = get_cache()
    cached_embedding = cache.get(cache_key)
    if cached_embedding is not None:
        print("📦 Cache hit for embedding")
        return cached_embedding

    try:
        payload = {"model": model, "input": [sanitized_content]}

        response = requests.post(
            "https://api.mistral.ai/v1/embeddings",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        embedding = response.json()["data"][0]["embedding"]

        # Cache the result
        cache.set(cache_key, embedding)
        print("💾 Cached embedding for content")

        return embedding
    except Exception as e:
        print(f"❌ API call failed for content length {len(sanitized_content)}: {e}")
        # Try with just title if content is too long
        if len(sanitized_content) > 20000:
            print("🔄 Retrying with title only...")
            try:
                title_only = _sanitize_content(content.split("\n")[0])
                if title_only and len(title_only) < 1000:
                    payload = {"model": model, "input": [title_only]}
                    response = requests.post(
                        "https://api.mistral.ai/v1/embeddings",
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json",
                        },
                        json=payload,
                        timeout=30,
                    )
                    response.raise_for_status()
                    embedding = response.json()["data"][0]["embedding"]
                    cache.set(cache_key, embedding)
                    print("💾 Cached title-only embedding")
                    return embedding
            except Exception as retry_e:
                print(f"❌ Title-only retry also failed: {retry_e}")
        return None


def get_mistral_completion(
    prompt: str, api_key: str, model: str = "mistral-small"
) -> str | None:
    """Get completion from Mistral API with caching."""
    if not prompt.strip():
        return None

    # Check cache first
    cache_key = _get_cache_key(prompt, model)
    cache = get_cache()
    cached_completion = cache.get(cache_key)
    if cached_completion is not None:
        print("📦 Cache hit for completion")
        return cached_completion

    try:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 500,
            "temperature": 0.1,
        }

        response = requests.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        completion = response.json()["choices"][0]["message"]["content"]

        # Cache the result
        cache.set(cache_key, completion)
        print("💾 Cached completion")

        return completion
    except Exception as e:
        print(f"❌ API call failed: {e}")
        return None


def get_issue_embedding(
    issue: dict[str, Any], api_key: str | None, model: str
) -> list[float] | None:
    """Get embedding for issue content."""
    if not api_key:
        return None

    # Truncate body and comments to fit within API limits
    title = issue.get("title", "")
    body = issue.get("body", "") or ""
    if len(body) > 15000:  # Reduced from 30000
        body = body[:15000] + "... [truncated]"

    # Limit comments to prevent excessive length
    comments = issue.get("comments", [])
    comment_texts = []
    total_comment_length = 0
    for comment in comments:
        comment_body = comment.get("body", "") or ""
        if total_comment_length + len(comment_body) > 8000:  # Reduced from 15000
            comment_texts.append("[... more comments truncated]")
            break
        comment_texts.append(comment_body)
        total_comment_length += len(comment_body)

    content_parts = [
        title,
        body,
        "\n".join(comment_texts),
    ]

    content = "\n".join(content_parts).strip()

    if not content:
        return None

    return get_mistral_embedding(content, api_key, model)


def get_issue_summary(
    issue: dict[str, Any], api_key: str | None, model: str = "mistral-small"
) -> str | None:
    """Generate a summary for the issue including metrics analysis."""
    if not api_key:
        return None

    # Create a compact, LLM-friendly representation of the issue
    compact_issue = {
        "number": issue.get("number"),
        "url": issue.get("url"),
        "title": issue.get("title"),
        "conversation": issue.get("conversation"),
        "author": issue.get("author", {}).get("login"),
        "updatedAt": issue.get("updatedAt"),
        "labels": [label.get("name") for label in issue.get("labels", [])],
        "state": issue.get("state"),
        "age_days": issue.get("age_days"),
        "age_days_quartile": issue.get("age_days_quartile"),
        "comment_count": issue.get("comment_count"),
        "comment_count_quartile": issue.get("comment_count_quartile"),
        "engagements": issue.get("engagements"),
        "engagements_quartile": issue.get("engagements_quartile"),
        "engagements_per_day": issue.get("engagements_per_day"),
        "engagements_per_day_quartile": issue.get("engagements_per_day_quartile"),
        "body_emojis": issue.get("body_emojis"),
        "body_emojis_quartile": issue.get("body_emojis_quartile"),
        "comment_emojis": issue.get("comment_emojis"),
        "comment_emojis_quartile": issue.get("comment_emojis_quartile"),
        "total_emojis": issue.get("total_emojis"),
        "total_emojis_quartile": issue.get("total_emojis_quartile"),
    }

    issue_json = json.dumps(
        compact_issue, indent=None, separators=(",", ":"), default=str
    )

    prompt = f"""Analyze this GitHub issue and write a concise 3 sentence summary:
- The main problem or request, NOT restating issue number or url.
- Key discussion points and current status (open, fixed & closed, not fixed & closed, etc)
- Community engagement using engagement, age, comment and emoji metrics and their associated quartiles fields.

Issue:
{issue_json}
"""

    return get_mistral_completion(prompt, api_key, model)


def calc_comment_count(issue: dict[str, Any]) -> int:
    """Calculate number of comments on issue."""
    return len(issue.get("comments", []))


def calc_reaction_totals(reaction_groups: list[dict[str, Any]]) -> int:
    """Calculate total reactions from reaction groups."""
    return sum(r.get("users", {}).get("totalCount", 0) for r in reaction_groups)


def format_emoji_counts(reaction_groups: list[dict[str, Any]]) -> str:
    """Format emoji reaction counts for conversation column."""
    if not reaction_groups:
        return ""

    # Map GitHub reaction content to actual emoji symbols
    emoji_map = {
        "THUMBS_UP": "👍",
        "THUMBS_DOWN": "👎",
        "LAUGH": "😄",
        "HOORAY": "🎉",
        "CONFUSED": "😕",
        "HEART": "❤️",
        "ROCKET": "🚀",
        "EYES": "👀"
    }

    emoji_counts = []
    for reaction in reaction_groups:
        content = reaction.get("content", "")
        count = reaction.get("users", {}).get("totalCount", 0)
        if count > 0:
            emoji = emoji_map.get(content, content)
            emoji_counts.append(f"{emoji} {count}")

    return f" [{', '.join(emoji_counts)}]" if emoji_counts else ""


def create_conversation_column(issue: dict[str, Any]) -> str:
    """Create a conversation column that combines title, body, and comments with emoji counts."""
    parts = []

    # Add title
    title = issue.get("title", "").strip()
    if title:
        parts.append(title)

    # Add body with author login (truncate if too long)
    body = issue.get("body", "")
    author_login = issue.get("author", {}).get("login", "unknown")
    if body and body.strip():
        # Truncate body for conversation column
        if len(body) > 20000:
            body = body[:20000] + "... [truncated]"
        body_part = f"{author_login}: {body.strip()}"
        # Add issue-level emoji counts
        emoji_counts = format_emoji_counts(issue.get("reactionGroups", []))
        body_part += emoji_counts
        parts.append(body_part)

    # Add comments with author login and emoji counts (truncate individually)
    comments = issue.get("comments", [])
    total_comment_chars = 0
    for comment in comments:
        comment_body = comment.get("body", "")
        comment_author = comment.get("author", {}).get("login", "unknown")
        if comment_body and comment_body.strip():
            # Truncate individual comments if too long
            if len(comment_body) > 5000:
                comment_body = comment_body[:5000] + "... [truncated]"

            # Stop adding comments if total gets too long
            if total_comment_chars + len(comment_body) > 25000:
                parts.append("[... more comments truncated for conversation]")
                break

            comment_part = f"{comment_author}: {comment_body.strip()}"
            # Add comment-level emoji counts
            if comment.get("reactionGroups"):
                # Use GraphQL format with detailed emoji breakdown
                emoji_counts = format_emoji_counts(comment.get("reactionGroups", []))
                comment_part += emoji_counts
            else:
                # Fallback to REST API format (total count only)
                reaction_count = comment.get("reactions", {}).get("totalCount", 0)
                if reaction_count > 0:
                    comment_part += f" [👍 {reaction_count}]"
            parts.append(comment_part)
            total_comment_chars += len(comment_body)

    return "\n\n".join(parts)


def calc_reaction_metrics(issue: dict[str, Any]) -> dict[str, int]:
    """Calculate reaction metrics for issue and comments separately."""
    # Handle both GraphQL format (reactionGroups) and REST format (reactions.totalCount)
    issue_reactions = calc_reaction_totals(issue.get("reactionGroups", []))

    # If no reactionGroups (REST API), try to get total from reactions field
    if issue_reactions == 0 and "reactions" in issue:
        issue_reactions = issue.get("reactions", {}).get("totalCount", 0)

    # Calculate comment reactions separately
    comment_reactions_total = 0

    for comment in issue.get("comments", []):
        # Try GraphQL format first
        comment_reactions = calc_reaction_totals(comment.get("reactionGroups", []))

        # If no reactionGroups (REST API), use reactions.totalCount
        if comment_reactions == 0 and "reactions" in comment:
            comment_total = comment.get("reactions", {}).get("totalCount", 0)
            comment_reactions_total += comment_total
        else:
            comment_reactions_total += comment_reactions

    return {
        "body_emojis": issue_reactions,
        "comment_emojis": comment_reactions_total,
        "total_emojis": issue_reactions + comment_reactions_total,
    }


def calc_age_days(issue: dict[str, Any]) -> int:
    """Calculate age in days between creation and last update."""
    created = datetime.fromisoformat(issue["createdAt"].replace("Z", "+00:00"))
    updated = datetime.fromisoformat(issue["updatedAt"].replace("Z", "+00:00"))
    return int((updated - created).days)


def calc_engagements(comment_count: int, total_emojis: int) -> int:
    """Calculate total engagements (comments + reactions)."""
    return comment_count + total_emojis


def calc_engagements_per_day(engagements: int, age_days: int) -> float:
    """Calculate engagements per day from total engagements and age."""
    return engagements / age_days if age_days > 0 else 0.0


def enrich_issue(
    issue: dict[str, Any], api_key: str | None, model: str
) -> dict[str, Any]:
    """Enrich a single issue with additional metrics and embeddings."""
    enriched = issue.copy()

    print(f"🔧 Enriching issue #{issue['number']}: {issue['title'][:50]}...")

    # Initialize recommendations field as empty list if not present
    if "recommendations" not in enriched:
        enriched["recommendations"] = []

    # Add conversation column
    enriched["conversation"] = create_conversation_column(issue)

    # Add embeddings
    enriched["embedding"] = get_issue_embedding(issue, api_key, model)

    # Add metrics
    enriched["comment_count"] = calc_comment_count(issue)

    reactions = calc_reaction_metrics(issue)
    enriched.update(reactions)

    enriched["age_days"] = calc_age_days(issue)
    enriched["engagements"] = calc_engagements(
        enriched["comment_count"], enriched["total_emojis"]
    )
    enriched["engagements_per_day"] = calc_engagements_per_day(
        enriched["engagements"], enriched["age_days"]
    )

    return enriched


def add_quartile_columns(issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add quartile columns for key metrics using pandas qcut."""
    metrics = [
        "comment_count",
        "age_days",
        "engagements",
        "engagements_per_day",
        "body_emojis",
        "comment_emojis",
        "total_emojis",
    ]

    df = pd.DataFrame(issues)

    # Use pandas qcut for clean quartile assignment
    for metric in metrics:
        if metric in df.columns:
            try:
                quartile_col = f"{metric}_quartile"
                df[quartile_col] = pd.qcut(
                    df[metric],
                    q=4,
                    labels=["Bottom25%", "Bottom50%", "Top50%", "Top25%"],
                    duplicates="raise",
                )
            except ValueError:
                # Handle case where all values are the same
                df[f"{metric}_quartile"] = "Top25%"

    return df.to_dict("records")


def add_summaries(
    issues: list[dict[str, Any]], api_key: str | None, model: str = "mistral-small"
) -> list[dict[str, Any]]:
    """Add AI-generated summaries to issues (must run after quartiles are added)."""
    if not api_key:
        print("⚠️  No API key provided, skipping summaries")
        for issue in issues:
            issue["summary"] = None
        return issues

    print(f"📝 Generating summaries for {len(issues)} issues...")

    for issue in issues:
        print(f"🔧 Generating summary for issue #{issue['number']}")
        issue["summary"] = get_issue_summary(issue, api_key, model)

    return issues


def add_k4_distances(issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add k-4 nearest neighbor distances using embeddings."""
    # Filter issues with embeddings
    issues_with_embeddings = [
        issue for issue in issues if issue.get("embedding") is not None
    ]

    if len(issues_with_embeddings) < 5:  # Need at least 5 for k=4 neighbors
        # Add empty k4_distances for all issues
        for issue in issues:
            issue["k4_distances"] = []
        return issues

    # Extract embeddings, issue numbers, and titles
    embeddings = np.array([issue["embedding"] for issue in issues_with_embeddings])
    issue_numbers = [issue["number"] for issue in issues_with_embeddings]
    issue_titles = [issue.get("title", "") for issue in issues_with_embeddings]

    # Create mapping from issue number to title for quick lookup
    issue_title_map = dict(zip(issue_numbers, issue_titles, strict=False))

    # Fit k-nearest neighbors (k=5 to get 4 neighbors excluding self)
    nn = NearestNeighbors(n_neighbors=5, metric="cosine")
    nn.fit(embeddings)

    # Find distances and indices
    distances, indices = nn.kneighbors(embeddings)

    # Create mapping from issue number to k4_distances
    k4_distances_map = {}

    for i, issue_num in enumerate(issue_numbers):
        # Skip first neighbor (self at distance 0) and take next 4
        neighbor_distances = distances[i][1:5]
        neighbor_indices = indices[i][1:5]

        k4_distances = []
        for dist, idx in zip(neighbor_distances, neighbor_indices, strict=False):
            neighbor_issue_num = issue_numbers[idx]
            neighbor_title = issue_title_map[neighbor_issue_num]
            k4_distances.append(
                {
                    "issue_number": neighbor_issue_num,
                    "title": neighbor_title,
                    "distance": float(dist),
                }
            )

        k4_distances_map[issue_num] = k4_distances

    # Add k4_distances to all issues
    for issue in issues:
        issue_num = issue["number"]
        issue["k4_distances"] = k4_distances_map.get(issue_num, [])

    return issues




def extract_cluster_keywords_llm(issues_in_cluster: list[dict[str, Any]], api_key: str | None, model: str = "mistral-large-latest") -> list[str]:
    """Extract representative keywords for a cluster of issues using LLM."""
    if not api_key or len(issues_in_cluster) == 0:
        return []

    # Create a summary of the cluster using only titles, limit to 500 issues
    titles = []
    for issue in issues_in_cluster[:500]:
        title = issue.get("title", "")
        if title:
            titles.append(title)

    if not titles:
        return []

    titles_text = "\n".join(titles)

    prompt = f"""Analyze these GitHub issue titles and extract 5-8 representative keywords that capture the main themes and topics. Return only the keywords as a comma-separated list, no explanations.

Issue titles:
{titles_text}

Keywords:"""

    response = get_mistral_completion(prompt, api_key, model)
    if response:
        # Parse keywords from response
        keywords = [k.strip() for k in response.split(",") if k.strip()]
        return keywords[:8]  # Limit to 8 keywords

    return []


def extract_condensed_tree_taxonomy(clusterer: hdbscan.HDBSCAN, issues_with_embeddings: list[dict[str, Any]], min_cluster_size: int = 5) -> dict[str, Any]:
    """Extract hierarchical taxonomy from HDBSCAN condensed tree."""
    if not hasattr(clusterer, 'condensed_tree_'):
        return {}

    condensed_tree = clusterer.condensed_tree_

    # Get the condensed tree data as a pandas DataFrame for easier analysis
    tree_df = condensed_tree.to_pandas()

    # Separate split nodes from persistence nodes
    # Split nodes are where a cluster splits into multiple children
    # Persistence nodes are where cluster size reduces without splitting

    # Group by parent to identify splits vs persistence
    parent_groups = tree_df.groupby('parent')

    taxonomy = {}
    split_nodes = set()
    persistence_nodes = set()

    # Identify split vs persistence nodes
    for parent_id, group in parent_groups:
        children = group['child'].unique()
        if len(children) > 1:
            # This parent has multiple children - it's a split node
            split_nodes.add(parent_id)
        else:
            # Single child - persistence node
            persistence_nodes.add(parent_id)

    # Find leaf clusters (final clusters that appear in cluster_labels)
    cluster_labels = clusterer.labels_
    leaf_clusters = set(cluster_labels[cluster_labels >= 0])

    # Build taxonomy by traversing from root
    root_cluster = tree_df['parent'].max()  # Root is typically the largest cluster ID

    def get_descendant_leaves(node_id: int) -> set[int]:
        """Get all leaf cluster IDs that are descendants of this node."""
        descendants = set()

        def traverse_descendants(current_node: int):
            children_df = tree_df[tree_df['parent'] == current_node]
            if children_df.empty:
                # This is a leaf
                if current_node in leaf_clusters:
                    descendants.add(current_node)
                return

            for child_id in children_df['child'].unique():
                traverse_descendants(child_id)

        traverse_descendants(node_id)
        return descendants

    def get_cluster_issues(cluster_id: int) -> list[dict[str, Any]]:
        """Get issues belonging to a specific cluster."""
        if cluster_id in leaf_clusters:
            # For leaf clusters, use the final cluster assignments
            issue_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
        else:
            # For internal nodes, get all issues from descendant leaf clusters
            descendant_leaves = get_descendant_leaves(cluster_id)
            issue_indices = []
            for leaf_id in descendant_leaves:
                leaf_indices = [i for i, label in enumerate(cluster_labels) if label == leaf_id]
                issue_indices.extend(leaf_indices)

        return [issues_with_embeddings[i] for i in issue_indices if i < len(issues_with_embeddings)]

    def traverse_tree(node_id: int, path: list[str], level: int) -> None:
        """Recursively traverse the condensed tree to build taxonomy."""
        if level > 10:  # Prevent infinite recursion
            return

        # Get children of this node
        children_df = tree_df[tree_df['parent'] == node_id]

        if children_df.empty:
            # Leaf node
            issues = get_cluster_issues(node_id)
            if issues:
                features = extract_cluster_features_hdbscan(issues, node_id)
                taxonomy[node_id] = {
                    'name': features['name'],
                    'keywords': features['keywords'],
                    'issues': [issue['number'] for issue in issues],
                    'path': path + [features['name']],
                    'level': level,
                    'node_type': 'leaf',
                    'size': len(issues)
                }
            return

        # Check if this is a split node
        children = children_df['child'].unique()

        if node_id in split_nodes and len(children) > 1:
            # Split node - create taxonomy entry and recurse to children
            # Get representative sample for this level
            issues = get_cluster_issues(node_id)

            # For better representativeness, sample from the most stable part of the cluster
            # before the split (last persistence node before this split)
            if len(issues) > 50:  # If we have many issues, sample strategically
                # Sample proportionally from each child cluster to maintain diversity
                sampled_issues = []
                issues_per_child = min(20, len(issues) // len(children))

                for child_id in children:
                    child_issues = get_cluster_issues(child_id)
                    if child_issues:
                        # Take a representative sample from this child
                        sample_size = min(issues_per_child, len(child_issues))
                        sampled_issues.extend(child_issues[:sample_size])

                issues = sampled_issues[:60]  # Limit total sample size

            if issues:
                features = extract_cluster_features_hdbscan(issues, node_id)
                node_name = features['name']

                taxonomy[node_id] = {
                    'name': node_name,
                    'keywords': features['keywords'],
                    'issues': [issue['number'] for issue in issues],
                    'path': path + [node_name],
                    'level': level,
                    'node_type': 'split',
                    'children': list(children),
                    'size': len(issues)
                }

                # Recurse to children
                for child_id in children:
                    traverse_tree(child_id, path + [node_name], level + 1)

        elif len(children) == 1:
            # Persistence node - skip and continue to child
            child_id = children[0]
            traverse_tree(child_id, path, level)

    # Start traversal from root
    traverse_tree(root_cluster, ['all'], 0)

    return taxonomy


def print_taxonomy_tree(taxonomy: dict[str, Any]) -> None:
    """Print the full taxonomy as a tree structure."""
    if not taxonomy:
        print("📋 No taxonomy available")
        return

    print("🌳 Full Taxonomy Tree:")

    # Build parent-child relationships
    children_map = {}
    root_nodes = []

    for node_id, node_info in taxonomy.items():
        level = node_info.get('level', 0)
        children = node_info.get('children', [])

        # Track children relationships
        if children:
            children_map[node_id] = children

        # Root nodes are at level 0
        if level == 0:
            root_nodes.append(node_id)

    def print_node(node_id: int, prefix: str = "", is_last: bool = True) -> None:
        """Recursively print tree nodes with proper tree formatting."""
        if node_id not in taxonomy:
            return

        node_info = taxonomy[node_id]
        name = node_info.get('name', f'node_{node_id}')
        size = node_info.get('size', 0)
        node_type = node_info.get('node_type', 'unknown')
        keywords = node_info.get('keywords', [])[:3]  # Show top 3 keywords
        path = '.'.join(node_info.get('path', []))

        # Tree formatting
        connector = "└── " if is_last else "├── "

        # Format node information
        keywords_str = f" [{', '.join(keywords)}]" if keywords else ""
        type_indicator = {"split": "🔀", "leaf": "🍃", "unknown": "❓"}.get(node_type, "❓")

        print(f"{prefix}{connector}{type_indicator} {name} ({size} issues){keywords_str}")
        print(f"{prefix}{'    ' if is_last else '│   '}    └─ Path: {path}")

        # Print children
        children = children_map.get(node_id, [])
        if children:
            # Sort children by size (descending)
            children_with_info = []
            for child_id in children:
                if child_id in taxonomy:
                    child_size = taxonomy[child_id].get('size', 0)
                    children_with_info.append((child_id, child_size))

            children_with_info.sort(key=lambda x: x[1], reverse=True)
            sorted_children = [child_id for child_id, _ in children_with_info]

            for i, child_id in enumerate(sorted_children):
                is_last_child = (i == len(sorted_children) - 1)
                new_prefix = prefix + ("    " if is_last else "│   ")
                print_node(child_id, new_prefix, is_last_child)

    # Print from root nodes
    for i, root_id in enumerate(sorted(root_nodes)):
        is_last_root = (i == len(root_nodes) - 1)
        print_node(root_id, "", is_last_root)

    print()  # Add spacing after tree


def create_taxonomy_paths(taxonomy: dict[str, Any]) -> dict[int, str]:
    """Create dot-separated taxonomy paths for each issue."""
    issue_paths = {}

    for _node_id, node_info in taxonomy.items():
        path_components = node_info.get('path', [])
        # Clean up path components to create valid taxonomy names
        clean_components = []
        for component in path_components:
            # Clean the component name
            clean_name = re.sub(r'[^a-zA-Z0-9]', '_', component.lower())
            clean_name = re.sub(r'_+', '_', clean_name)  # Remove multiple underscores
            clean_name = clean_name.strip('_')  # Remove leading/trailing underscores
            if clean_name and clean_name != 'cluster':  # Skip empty or generic names
                clean_components.append(clean_name)

        if clean_components:
            taxonomy_path = '.'.join(clean_components)

            # Assign this path to all issues in this node
            for issue_number in node_info.get('issues', []):
                # Only assign if we don't have a path yet, or this is a deeper (more specific) path
                if issue_number not in issue_paths or len(clean_components) > len(issue_paths[issue_number].split('.')):
                    issue_paths[issue_number] = taxonomy_path

    return issue_paths


def extract_cluster_features_hdbscan(issues_in_cluster: list[dict[str, Any]], cluster_id: int) -> dict[str, Any]:
    """Extract representative features for an HDBSCAN cluster using TF-IDF."""
    if not issues_in_cluster:
        return {
            'name': f'cluster_{cluster_id}',
            'keywords': [],
            'size': 0
        }

    # Combine titles and bodies for text analysis
    def get_text(issue):
        title = issue.get('title', '') or ''
        body = issue.get('body', '') or ''
        return f"{title} {body}".strip()

    cluster_texts = [get_text(issue) for issue in issues_in_cluster]
    cluster_texts = [text for text in cluster_texts if text]

    if not cluster_texts:
        return {
            'name': f'cluster_{cluster_id}',
            'keywords': [],
            'size': len(issues_in_cluster)
        }

    try:
        # Simple keyword extraction using TF-IDF
        vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.8,
            lowercase=True,
            token_pattern=r'\b[a-zA-Z][a-zA-Z]+\b'
        )

        tfidf_matrix = vectorizer.fit_transform(cluster_texts)
        feature_names = vectorizer.get_feature_names_out()

        # Get average TF-IDF scores across all documents in cluster
        avg_scores = tfidf_matrix.mean(axis=0).A1

        # Get top keywords
        top_indices = avg_scores.argsort()[-10:][::-1]
        keywords = [feature_names[i] for i in top_indices if avg_scores[i] > 0]

        # Create cluster name from top keywords
        name = '_'.join(keywords[:3]) if keywords else f'cluster_{cluster_id}'
        name = re.sub(r'[^a-zA-Z0-9_]', '_', name)[:30]

        return {
            'name': name or f'cluster_{cluster_id}',
            'keywords': keywords[:8],
            'size': len(issues_in_cluster)
        }

    except Exception as e:
        print(f"⚠️  Feature extraction failed for cluster {cluster_id}: {e}")
        return {
            'name': f'cluster_{cluster_id}',
            'keywords': [],
            'size': len(issues_in_cluster)
        }


def perform_hdbscan_clustering(issues: list[dict[str, Any]], api_key: str | None = None, min_cluster_size: int = 5, min_samples: int = 3) -> list[dict[str, Any]]:
    """Perform HDBSCAN clustering on all issues using their embeddings."""
    # Filter issues with embeddings (cluster all of them)
    issues_with_embeddings = [
        issue for issue in issues if issue.get("embedding") is not None
    ]

    if len(issues_with_embeddings) < min_cluster_size:
        print(f"⚠️  Not enough issues with embeddings for clustering (need at least {min_cluster_size})")
        # Add empty cluster info to all issues
        for issue in issues:
            issue["cluster_id"] = -1  # HDBSCAN uses -1 for noise
            issue["cluster_size"] = None
            issue["cluster_keywords"] = []
        return issues

    print(f"🔍 Performing HDBSCAN clustering on {len(issues_with_embeddings)} issues with embeddings...")
    print(f"  Parameters: min_cluster_size={min_cluster_size}, min_samples={min_samples}")

    # Extract embeddings
    embeddings = np.array([issue["embedding"] for issue in issues_with_embeddings])

    # Normalize embeddings for cosine-like distance using euclidean metric
    # This is equivalent to cosine distance for unit vectors
    embeddings_normalized = normalize(embeddings, norm='l2')

    # Perform HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',  # Euclidean on normalized vectors = cosine distance
        cluster_selection_method='eom'  # Excess of Mass for better cluster selection
    )

    cluster_labels = clusterer.fit_predict(embeddings_normalized)

    # Plot the condensed tree for visualization
    try:
        print("📈 Plotting HDBSCAN condensed tree...")
        import matplotlib.pyplot as plt
        
        # Create the condensed tree plot
        fig, ax = plt.subplots(figsize=(12, 8))
        clusterer.condensed_tree_.plot(select_clusters=True, selection_palette=['red', 'blue', 'green', 'orange', 'purple'], ax=ax)
        plt.title('HDBSCAN Condensed Tree')
        plt.tight_layout()
        
        # Save the plot
        plot_path = 'hdbscan_condensed_tree.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Condensed tree plot saved to: {plot_path}")
        plt.close()
        
        # Also create a simplified tree plot
        fig, ax = plt.subplots(figsize=(10, 6))
        clusterer.condensed_tree_.plot(ax=ax)
        plt.title('HDBSCAN Condensed Tree (Full)')
        plt.tight_layout()
        
        plot_path_full = 'hdbscan_condensed_tree_full.png'
        plt.savefig(plot_path_full, dpi=300, bbox_inches='tight')
        print(f"📊 Full condensed tree plot saved to: {plot_path_full}")
        plt.close()
        
        # Print condensed tree statistics
        print(f"🔍 Condensed Tree Analysis:")
        print(f"  Tree size: {len(clusterer.condensed_tree_.to_pandas())}")
        print(f"  Cluster persistence: {clusterer.cluster_persistence_}")
        print(f"  Probabilities range: {clusterer.probabilities_.min():.3f} - {clusterer.probabilities_.max():.3f}")
        
        # Show the raw condensed tree data (first few rows)
        tree_df = clusterer.condensed_tree_.to_pandas()
        print(f"📋 Condensed Tree Data (first 10 rows):")
        print(tree_df.head(10).to_string())
        
    except ImportError:
        print("⚠️  matplotlib not available for plotting")
    except Exception as e:
        print(f"⚠️  Error creating condensed tree plot: {e}")

    # Group issues by cluster
    clusters = {}
    noise_issues = []

    for i, label in enumerate(cluster_labels):
        if label == -1:  # Noise points
            noise_issues.append(issues_with_embeddings[i])
        else:
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(issues_with_embeddings[i])

    # Extract features for each cluster
    cluster_features = {}
    for cluster_id, cluster_issues in clusters.items():
        features = extract_cluster_features_hdbscan(cluster_issues, cluster_id)
        cluster_features[cluster_id] = features

    # Extract hierarchical taxonomy from condensed tree
    print("🌳 Extracting hierarchical taxonomy from condensed tree...")
    taxonomy = extract_condensed_tree_taxonomy(clusterer, issues_with_embeddings, min_cluster_size)

    # Create taxonomy paths for each issue
    issue_taxonomy_paths = create_taxonomy_paths(taxonomy)

    # Create mapping from issue number to cluster info
    issue_cluster_map = {}

    # Add cluster information for clustered issues
    for cluster_id, cluster_issues in clusters.items():
        features = cluster_features[cluster_id]
        for issue in cluster_issues:
            issue_num = issue["number"]
            issue_cluster_map[issue_num] = {
                'cluster_id': cluster_id,
                'cluster_name': features['name'],
                'cluster_keywords': features['keywords'],
                'cluster_size': features['size'],
                'taxonomy_path': issue_taxonomy_paths.get(issue_num, 'all.uncategorized')
            }

    # Add noise information for noise issues
    for issue in noise_issues:
        issue_num = issue["number"]
        issue_cluster_map[issue_num] = {
            'cluster_id': -1,
            'cluster_name': 'noise',
            'cluster_keywords': [],
            'cluster_size': len(noise_issues),
            'taxonomy_path': 'all.noise'
        }

    # Add cluster information to all issues
    for issue in issues:
        issue_num = issue["number"]

        if issue_num in issue_cluster_map:
            cluster_info = issue_cluster_map[issue_num]
            issue["cluster_id"] = cluster_info['cluster_id']
            issue["cluster_name"] = cluster_info['cluster_name']
            issue["cluster_keywords"] = cluster_info['cluster_keywords']
            issue["cluster_size"] = cluster_info['cluster_size']
            issue["taxonomy_path"] = cluster_info['taxonomy_path']
        else:
            # Issues without embeddings
            issue["cluster_id"] = None
            issue["cluster_name"] = "no_embedding"
            issue["cluster_keywords"] = []
            issue["cluster_size"] = None
            issue["taxonomy_path"] = "all.no_embedding"

    # Print clustering statistics
    num_clusters = len(clusters)
    num_noise = len(noise_issues)
    num_clustered = sum(len(cluster_issues) for cluster_issues in clusters.values())

    print("📊 HDBSCAN Clustering Results:")
    print(f"  Number of clusters: {num_clusters}")
    print(f"  Clustered issues: {num_clustered}")
    print(f"  Noise issues: {num_noise}")
    print(f"  Issues without embeddings: {len(issues) - len(issues_with_embeddings)}")
    
    # Provide parameter tuning suggestions if clustering seems too conservative
    if num_clusters < 5 and num_noise > num_clustered:
        print("\n⚠️  Clustering seems conservative (few clusters, lots of noise).")
        print("💡 Consider adjusting parameters:")
        print(f"   - Reduce min_cluster_size (current: {min_cluster_size}) → try {max(3, min_cluster_size//2)}")
        print(f"   - Reduce min_samples (current: {min_samples}) → try {max(1, min_samples//2)}")
        print("   - Or use cluster_selection_method='leaf' for more granular clusters")

    # Print cluster details
    if clusters:
        print("🔍 Cluster Details:")
        sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)
        for cluster_id, cluster_issues in sorted_clusters[:10]:  # Top 10 clusters
            features = cluster_features[cluster_id]
            keywords_str = ', '.join(features['keywords'][:5]) if features['keywords'] else 'no keywords'
            print(f"  Cluster {cluster_id} ({len(cluster_issues)} issues): {features['name']}")
            print(f"    Keywords: {keywords_str}")

    # Print full taxonomy tree
    if taxonomy:
        print_taxonomy_tree(taxonomy)

        # Show some example taxonomy paths
        sample_paths = set()
        for issue in issues[:20]:  # Sample from first 20 issues
            if issue.get('taxonomy_path'):
                sample_paths.add(issue['taxonomy_path'])

        if sample_paths:
            print("📋 Sample Taxonomy Paths:")
            for path in sorted(sample_paths)[:10]:
                print(f"    {path}")

    return issues


def print_stats(enriched: list[dict[str, Any]]) -> None:
    """Print statistics about enriched issues."""
    total = len(enriched)
    with_embeddings = sum(1 for issue in enriched if issue.get("embedding") is not None)
    with_summaries = sum(1 for issue in enriched if issue.get("summary") is not None)
    with_clusters = sum(1 for issue in enriched if issue.get("cluster_id") is not None)

    print("📊 Statistics:")
    print(f"  Total issues: {total}")
    print(f"  With embeddings: {with_embeddings}")
    print(f"  With summaries: {with_summaries}")
    print(f"  With clusters: {with_clusters}")
