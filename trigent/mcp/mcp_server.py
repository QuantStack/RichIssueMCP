"""FastMCP server for accessing Rich Issues database."""

import gzip
import json
import re
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

mcp = FastMCP("Rich Issues Server")
_issues_cache: dict[str, list[dict]] = {}
_default_db_file = "data/enriched-issues-jupyterlab-jupyterlab.json.gz"


def load_issues_db(db_file: str) -> list[dict]:
    """Load and cache issues from compressed JSON database."""
    if db_file in _issues_cache:
        return _issues_cache[db_file]

    db_path = Path(db_file)
    if not db_path.exists():
        raise FileNotFoundError(f"Database file not found: {db_file}")

    try:
        with gzip.open(db_path, "rt") as f:
            issues = json.load(f)
        _issues_cache[db_file] = issues
        return issues
    except Exception as e:
        raise RuntimeError(f"Failed to load database: {e}") from e


@mcp.tool()
def get_issue(issue_number: int, db_file: str | None = None) -> dict[str, Any] | None:
    """Get specific issue with all metadata and enrichment data."""
    if db_file is None:
        db_file = _default_db_file
    issues = load_issues_db(db_file)
    return next((i for i in issues if i["number"] == issue_number), None)


@mcp.tool()
def find_similar_issues(
    issue_number: int,
    threshold: float = 0.8,
    limit: int = 10,
    db_file: str | None = None,
) -> list[dict[str, Any]]:
    """Find issues similar to target issue using embeddings."""
    if db_file is None:
        db_file = _default_db_file
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
            result = issue.copy()
            result["similarity"] = similarity
            similar.append(result)

    return sorted(similar, key=lambda x: x["similarity"], reverse=True)[:limit]


@mcp.tool()
def find_linked_issues(
    issue_number: int, db_file: str | None = None
) -> list[dict[str, Any]]:
    """Find issues referenced in the body/comments of target issue."""
    if db_file is None:
        db_file = _default_db_file
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

    return [i for i in issues if i["number"] in linked_numbers]


@mcp.tool()
def get_issue_metrics(
    issue_number: int, db_file: str | None = None
) -> dict[str, Any] | None:
    """Get enrichment metrics for a specific issue."""
    if db_file is None:
        db_file = _default_db_file
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


def run_mcp_server(
    host: str = "localhost", port: int = 8000, db_file: str | None = None
) -> None:
    """Run the MCP server with specified configuration."""

    # Set default db_file if not provided
    if db_file is None:
        db_file = "data/enriched-issues-jupyterlab-jupyterlab.json.gz"

    print(f"🚀 Starting MCP server on {host}:{port}")
    print(f"📂 Using database: {db_file}")

    # Update default db_file in all tool functions
    global _default_db_file
    _default_db_file = db_file

    mcp.run(host=host, port=port)


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
