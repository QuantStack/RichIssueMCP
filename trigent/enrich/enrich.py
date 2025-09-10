#!/usr/bin/env python3
"""Main enrichment module for processing raw GitHub issues."""

import json
import gzip
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import requests
import pandas as pd
import numpy as np
import umap


def load_raw_issues(filepath: Path) -> List[Dict[str, Any]]:
    """Load raw issues from gzipped JSON file."""
    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        return json.load(f)


def save_enriched_issues(issues: List[Dict[str, Any]], filepath: Path) -> None:
    """Save enriched issues to gzipped JSON file."""
    with gzip.open(filepath, 'wt', encoding='utf-8') as f:
        json.dump(issues, f, indent=None, separators=(',', ':'))


def get_mistral_embedding(content: str, api_key: str, model: str = "mistral-embed") -> Optional[List[float]]:
    """Get embedding from Mistral API."""
    if not content.strip():
        return None
    
    try:
        payload = {
            "model": model,
            "input": [content]
        }
        
        response = requests.post(
            "https://api.mistral.ai/v1/embeddings",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]
    except Exception as e:
        print(f"❌ API call failed: {e}")
        return None


def get_issue_embedding(issue: Dict[str, Any], api_key: Optional[str], model: str) -> Optional[List[float]]:
    """Get embedding for issue content."""
    if not api_key:
        return None
    
    content_parts = [
        issue.get("title", ""),
        issue.get("body", "") or "",
        "\n".join(comment.get("body", "") or "" for comment in issue.get("comments", []))
    ]
    
    content = "\n".join(content_parts).strip()
    
    if not content:
        return None
    
    return get_mistral_embedding(content, api_key, model)


def calc_comment_count(issue: Dict[str, Any]) -> int:
    """Calculate number of comments on issue."""
    return len(issue.get("comments", []))


def calc_reaction_totals(reaction_groups: List[Dict[str, Any]]) -> Dict[str, int]:
    """Calculate reaction totals from reaction groups."""
    total = sum(r.get("totalCount", 0) for r in reaction_groups)
    positive = sum(r.get("totalCount", 0) for r in reaction_groups 
                   if r.get("content") in ["THUMBS_UP", "HEART", "HOORAY"])
    negative = sum(r.get("totalCount", 0) for r in reaction_groups 
                   if r.get("content") in ["THUMBS_DOWN", "CONFUSED"])
    
    return {"total": total, "positive": positive, "negative": negative}


def calc_reaction_metrics(issue: Dict[str, Any]) -> Dict[str, int]:
    """Calculate reaction metrics for issue and comments."""
    issue_reactions = calc_reaction_totals(issue.get("reactionGroups", []))
    
    comment_reactions_total = 0
    for comment in issue.get("comments", []):
        comment_reactions = calc_reaction_totals(comment.get("reactionGroups", []))
        comment_reactions_total += comment_reactions["total"]
    
    return {
        "issue_total_emojis": issue_reactions["total"] + comment_reactions_total,
        "issue_positive_emojis": issue_reactions["positive"],
        "issue_negative_emojis": issue_reactions["negative"],
        "conversation_total_emojis": issue_reactions["total"] + comment_reactions_total
    }


def calc_age_days(issue: Dict[str, Any]) -> int:
    """Calculate age in days between creation and last update."""
    created = datetime.fromisoformat(issue["createdAt"].replace('Z', '+00:00'))
    updated = datetime.fromisoformat(issue["updatedAt"].replace('Z', '+00:00'))
    return int((updated - created).days)


def calc_activity_score(comment_count: int, age_days: int) -> float:
    """Calculate activity score as comments per day."""
    return comment_count / age_days if age_days > 0 else 0.0


def enrich_issue(issue: Dict[str, Any], api_key: Optional[str], model: str) -> Dict[str, Any]:
    """Enrich a single issue with additional metrics and embeddings."""
    enriched = issue.copy()
    
    print(f"🔧 Enriching issue #{issue['number']}: {issue['title'][:50]}...")
    
    # Add embeddings
    enriched["embedding"] = get_issue_embedding(issue, api_key, model)
    
    # Add metrics
    enriched["comment_count"] = calc_comment_count(issue)
    
    reactions = calc_reaction_metrics(issue)
    enriched.update(reactions)
    
    enriched["age_days"] = calc_age_days(issue)
    enriched["activity_score"] = calc_activity_score(enriched["comment_count"], enriched["age_days"])
    
    return enriched


def add_quartile_columns(issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Add quartile columns for key metrics using pandas qcut."""
    metrics = ["comment_count", "age_days", "activity_score", "issue_positive_emojis", 
               "issue_negative_emojis", "conversation_total_emojis"]
    
    df = pd.DataFrame(issues)
    
    # Use pandas qcut for clean quartile assignment
    for metric in metrics:
        if metric in df.columns:
            try:
                quartile_col = f"{metric}_q"
                df[quartile_col] = pd.qcut(df[metric], q=4, labels=[0.25, 0.5, 0.75, 1.0], duplicates='drop')
            except ValueError:
                # Handle case where all values are the same
                df[f"{metric}_q"] = 1.0
    
    return df.to_dict('records')


def add_umap_projection(issues: List[Dict[str, Any]], n_components: int = 2) -> List[Dict[str, Any]]:
    """Add UMAP 2D projection of embeddings."""
    # Extract embeddings
    embeddings = []
    valid_indices = []
    
    for i, issue in enumerate(issues):
        if issue.get("embedding") is not None:
            embeddings.append(issue["embedding"])
            valid_indices.append(i)
    
    if len(embeddings) < 2:
        print("⚠️  Not enough embeddings for UMAP projection")
        return issues
    
    print(f"🗺️  Computing UMAP projection for {len(embeddings)} embeddings...")
    
    # Compute UMAP
    reducer = umap.UMAP(n_components=n_components, random_state=42)
    embedding_2d = reducer.fit_transform(np.array(embeddings))
    
    # Add projections back to issues
    enriched_issues = [issue.copy() for issue in issues]
    for idx, issue_idx in enumerate(valid_indices):
        enriched_issues[issue_idx]["umap_x"] = float(embedding_2d[idx, 0])
        enriched_issues[issue_idx]["umap_y"] = float(embedding_2d[idx, 1])
    
    # Set None for issues without embeddings
    for i, issue in enumerate(enriched_issues):
        if i not in valid_indices:
            issue["umap_x"] = None
            issue["umap_y"] = None
    
    return enriched_issues


def print_stats(enriched: List[Dict[str, Any]]) -> None:
    """Print statistics about enriched issues."""
    total = len(enriched)
    with_embeddings = sum(1 for issue in enriched if issue.get("embedding") is not None)
    with_umap = sum(1 for issue in enriched if issue.get("umap_x") is not None)
    
    print(f"📊 Statistics:")
    print(f"  Total issues: {total}")
    print(f"  With embeddings: {with_embeddings}")
    print(f"  With UMAP coordinates: {with_umap}")


def main():
    """Main enrichment entry point."""
    parser = argparse.ArgumentParser(description="Enrich raw GitHub issues with metrics and embeddings")
    parser.add_argument("input_file", help="Path to raw issues JSON.gz file")
    parser.add_argument("--api-key", help="Mistral API key for embeddings")
    parser.add_argument("--model", default="mistral-embed", help="Mistral embedding model")
    parser.add_argument("--output", help="Output file path (default: data/enriched-{input_basename})")
    parser.add_argument("--skip-umap", action="store_true", help="Skip UMAP projection")
    
    args = parser.parse_args()
    
    input_path = Path(args.input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        basename = input_path.name.replace("raw-", "enriched-")
        output_path = Path("data") / basename
    
    output_path.parent.mkdir(exist_ok=True)
    
    print(f"🔍 Loading raw issues from {input_path}...")
    raw_issues = load_raw_issues(input_path)
    print(f"📥 Retrieved {len(raw_issues)} issues")
    
    # Enrich issues
    enriched = [enrich_issue(issue, args.api_key, args.model) for issue in raw_issues]
    
    print("🔧 Computing quartile assignments...")
    enriched = add_quartile_columns(enriched)
    
    # Add UMAP projection if requested
    if not args.skip_umap:
        enriched = add_umap_projection(enriched)
    
    save_enriched_issues(enriched, output_path)
    print(f"✅ Enriched issue database saved to {output_path}")
    print_stats(enriched)


if __name__ == "__main__":
    main()