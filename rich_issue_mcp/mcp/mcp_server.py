"""FastMCP server for accessing Rich Issues database."""

import gzip
import json
import re
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from rich_issue_mcp.config import get_data_file_path

mcp = FastMCP("Rich Issues Server")
_issues_cache: dict[str, list[dict[str, Any]]] = {}


def _get_default_db_file() -> str:
    """Get default database file path from config."""
    try:
        return str(get_data_file_path("enriched-issues-jupyterlab-jupyterlab.json.gz"))
    except (FileNotFoundError, ValueError):
        # Fallback if config not available
        return "data/enriched-issues-jupyterlab-jupyterlab.json.gz"


def load_issues_db(db_file: str) -> list[dict[str, Any]]:
    """Load and cache issues from compressed JSON database."""
    if db_file in _issues_cache:
        return _issues_cache[db_file]

    db_path = Path(db_file)
    if not db_path.exists():
        raise FileNotFoundError(f"Database file not found: {db_file}")

    try:
        with gzip.open(db_path, "rt") as f:
            issues: list[dict[str, Any]] = json.load(f)
        _issues_cache[db_file] = issues
        return issues
    except Exception as e:
        raise RuntimeError(f"Failed to load database: {e}") from e


@mcp.tool()
def get_issue(issue_number: int, db_file: str | None = None) -> dict[str, Any] | None:
    """Get specific issue summary and number."""
    if db_file is None:
        db_file = _get_default_db_file()
    issues = load_issues_db(db_file)
    issue = next((i for i in issues if i["number"] == issue_number), None)
    if not issue:
        return None
    return {
        "number": issue["number"],
        "summary": issue.get("summary"),
        "conversation": issue.get("conversation")
    }


@mcp.tool()
def find_similar_issues(
    issue_number: int,
    threshold: float = 0.8,
    limit: int = 10,
    db_file: str | None = None,
) -> list[dict[str, Any]]:
    """Find issues similar to target issue using embeddings."""
    if db_file is None:
        db_file = _get_default_db_file()
    issues = load_issues_db(db_file)
    target = next((i for i in issues if i["number"] == issue_number), None)

    if not target or not target.get("embedding"):
        return []

    def cosine_similarity(a: list[float], b: list[float]) -> float:
        dot_product = sum(x * y for x, y in zip(a, b, strict=False))
        magnitude_a = sum(x * x for x in a) ** 0.5
        magnitude_b = sum(x * x for x in b) ** 0.5
        return (
            dot_product / (magnitude_a * magnitude_b)
            if magnitude_a and magnitude_b
            else 0
        )

    similar = []
    for issue in issues:
        if issue["number"] == issue_number or not issue.get("embedding"):
            continue

        similarity = cosine_similarity(target["embedding"], issue["embedding"])
        if similarity >= threshold:
            result = {
                "number": issue["number"],
                "title": issue.get("title"),
                "summary": issue.get("summary"),
                "url": issue.get("url"),
                "similarity": similarity,
            }
            similar.append(result)

    return sorted(similar, key=lambda x: x["similarity"], reverse=True)[:limit]


@mcp.tool()
def find_linked_issues(
    issue_number: int, db_file: str | None = None
) -> list[dict[str, Any]]:
    """Find issues referenced in the body/comments of target issue."""
    if db_file is None:
        db_file = _get_default_db_file()
    issues = load_issues_db(db_file)
    target = next((i for i in issues if i["number"] == issue_number), None)

    if not target:
        return []

    # Extract issue numbers from text (e.g., #1234, #5678)
    text = f"{target.get('title', '')} {target.get('body', '')}"
    for comment in target.get("comments", []):
        text += f" {comment.get('body', '')}"

    linked_numbers = {int(m.group(1)) for m in re.finditer(r"#(\d+)", text)}
    linked_numbers.discard(issue_number)  # Remove self-reference

    return [
        {
            "number": i["number"],
            "title": i.get("title"),
            "summary": i.get("summary"),
            "url": i.get("url")
        }
        for i in issues
        if i["number"] in linked_numbers
    ]


@mcp.tool()
def get_issue_metrics(
    issue_number: int, db_file: str | None = None
) -> dict[str, Any] | None:
    """Get enrichment metrics for a specific issue."""
    if db_file is None:
        db_file = _get_default_db_file()
    issues = load_issues_db(db_file)
    issue = next((i for i in issues if i["number"] == issue_number), None)

    if not issue:
        return None

    return {
        "priority_score": issue.get("priority_score", 0),
        "frequency_score": issue.get("frequency_score", 0),
        "severity_score": issue.get("severity_score", 0),
        "comment_count": issue.get("comment_count", 0),
        "total_reactions": issue.get("total_reactions", 0),
        "positive_reactions": issue.get("positive_reactions", 0),
        "negative_reactions": issue.get("negative_reactions", 0),
        "age_days": issue.get("age_days", 0),
        "has_embedding": issue.get("embedding") is not None,
    }


@mcp.tool()
def get_available_sort_columns(db_file: str | None = None) -> list[str]:
    """Get list of available columns that can be used for sorting issues."""
    if db_file is None:
        db_file = _get_default_db_file()
    issues = load_issues_db(db_file)

    if not issues:
        return []

    # Get all available columns from the first issue
    all_columns = list(issues[0].keys())

    # Filter to columns that are likely useful for sorting (numeric, string, not complex objects)
    sortable_columns = []
    sample_issue = issues[0]

    for column in all_columns:
        value = sample_issue.get(column)
        # Include columns with numeric, string, or None values
        # Exclude lists, dicts, and other complex types unless they're specific known ones
        if value is None or isinstance(value, int | float | str | bool):
            sortable_columns.append(column)
        elif column in ["k4_distances"]:  # Skip complex columns we know aren't sortable
            continue
        else:
            # For other types, check if they're consistently comparable across a few samples
            sample_values = [
                issue.get(column)
                for issue in issues[:5]
                if issue.get(column) is not None
            ]
            if sample_values and all(
                isinstance(v, type(sample_values[0])) for v in sample_values
            ):
                try:
                    # Test if values are sortable
                    sorted(sample_values)
                    sortable_columns.append(column)
                except (TypeError, ValueError):
                    continue

    return sorted(sortable_columns)


@mcp.tool()
def get_top_issues(
    sort_column: str,
    limit: int = 10,
    descending: bool = True,
    db_file: str | None = None,
) -> list[dict[str, Any]]:
    """Get top n issues sorted by a specific column from the enriched database."""
    if db_file is None:
        db_file = _get_default_db_file()
    issues = load_issues_db(db_file)

    if not issues:
        return []

    # Validate that the sort column exists
    available_columns = set(issues[0].keys()) if issues else set()
    if sort_column not in available_columns:
        raise ValueError(
            f"Column '{sort_column}' not found. Available columns: {sorted(available_columns)}"
        )

    # Filter out issues that don't have the sort column or have None values
    valid_issues = [issue for issue in issues if issue.get(sort_column) is not None]

    # Sort issues by the specified column
    try:
        sorted_issues = sorted(
            valid_issues, key=lambda x: x[sort_column], reverse=descending
        )
    except TypeError:
        # Handle case where values might not be comparable (mixed types)
        sorted_issues = sorted(
            valid_issues, key=lambda x: str(x[sort_column]), reverse=descending
        )

    return [
        {
            "number": issue["number"],
            "title": issue.get("title"),
            "summary": issue.get("summary"),
            "url": issue.get("url")
        }
        for issue in sorted_issues[:limit]
    ]


def run_mcp_server(
    host: str = "localhost", port: int = 8000, db_file: str | None = None
) -> None:
    """Run the MCP server with specified configuration."""

    # Set default db_file if not provided
    if db_file is None:
        db_file = _get_default_db_file()

    print("🚀 Starting MCP server")
    print(f"📂 Using database: {db_file}")

    # Run with stdio transport (default for MCP)
    mcp.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="FastMCP server for accessing Rich Issues database"
    )
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--db-file", help="Database file path")

    args = parser.parse_args()
    run_mcp_server(args.host, args.port, args.db_file)
