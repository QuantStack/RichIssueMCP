#!/usr/bin/env python3
"""Main enrichment module for processing raw GitHub issues."""

import argparse
import gzip
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from trigent.config import get_config


def load_raw_issues(filepath: Path) -> list[dict[str, Any]]:
    """Load raw issues from gzipped JSON file."""
    with gzip.open(filepath, "rt", encoding="utf-8") as f:
        return json.load(f)


def save_enriched_issues(issues: list[dict[str, Any]], filepath: Path) -> None:
    """Save enriched issues to gzipped JSON file."""
    with gzip.open(filepath, "wt", encoding="utf-8") as f:
        json.dump(issues, f, indent=None, separators=(",", ":"))


def get_mistral_embedding(
    content: str, api_key: str, model: str = "mistral-embed"
) -> list[float] | None:
    """Get embedding from Mistral API."""
    if not content.strip():
        return None

    try:
        payload = {"model": model, "input": [content]}

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
        return response.json()["data"][0]["embedding"]
    except Exception as e:
        print(f"❌ API call failed: {e}")
        return None


def get_issue_embedding(
    issue: dict[str, Any], api_key: str | None, model: str
) -> list[float] | None:
    """Get embedding for issue content."""
    if not api_key:
        return None

    content_parts = [
        issue.get("title", ""),
        issue.get("body", "") or "",
        "\n".join(
            comment.get("body", "") or "" for comment in issue.get("comments", [])
        ),
    ]

    content = "\n".join(content_parts).strip()

    if not content:
        return None

    return get_mistral_embedding(content, api_key, model)


def calc_comment_count(issue: dict[str, Any]) -> int:
    """Calculate number of comments on issue."""
    return len(issue.get("comments", []))


def calc_reaction_totals(reaction_groups: list[dict[str, Any]]) -> dict[str, int]:
    """Calculate reaction totals from reaction groups."""
    total = sum(r.get("totalCount", 0) for r in reaction_groups)
    positive = sum(
        r.get("totalCount", 0)
        for r in reaction_groups
        if r.get("content") in ["THUMBS_UP", "HEART", "HOORAY"]
    )
    negative = sum(
        r.get("totalCount", 0)
        for r in reaction_groups
        if r.get("content") in ["THUMBS_DOWN", "CONFUSED"]
    )

    return {"total": total, "positive": positive, "negative": negative}


def calc_reaction_metrics(issue: dict[str, Any]) -> dict[str, int]:
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
        "conversation_total_emojis": issue_reactions["total"] + comment_reactions_total,
    }


def calc_age_days(issue: dict[str, Any]) -> int:
    """Calculate age in days between creation and last update."""
    created = datetime.fromisoformat(issue["createdAt"].replace("Z", "+00:00"))
    updated = datetime.fromisoformat(issue["updatedAt"].replace("Z", "+00:00"))
    return int((updated - created).days)


def calc_activity_score(comment_count: int, age_days: int) -> float:
    """Calculate activity score as comments per day."""
    return comment_count / age_days if age_days > 0 else 0.0


def enrich_issue(
    issue: dict[str, Any], api_key: str | None, model: str
) -> dict[str, Any]:
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
    enriched["activity_score"] = calc_activity_score(
        enriched["comment_count"], enriched["age_days"]
    )

    return enriched


def add_quartile_columns(issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add quartile columns for key metrics using pandas qcut."""
    metrics = [
        "comment_count",
        "age_days",
        "activity_score",
        "issue_positive_emojis",
        "issue_negative_emojis",
        "conversation_total_emojis",
    ]

    df = pd.DataFrame(issues)

    # Use pandas qcut for clean quartile assignment
    for metric in metrics:
        if metric in df.columns:
            try:
                quartile_col = f"{metric}_q"
                df[quartile_col] = pd.qcut(
                    df[metric], q=4, labels=[0.25, 0.5, 0.75, 1.0], duplicates="drop"
                )
            except ValueError:
                # Handle case where all values are the same
                df[f"{metric}_q"] = 1.0

    return df.to_dict("records")


def print_stats(enriched: list[dict[str, Any]]) -> None:
    """Print statistics about enriched issues."""
    total = len(enriched)
    with_embeddings = sum(1 for issue in enriched if issue.get("embedding") is not None)

    print("📊 Statistics:")
    print(f"  Total issues: {total}")
    print(f"  With embeddings: {with_embeddings}")
