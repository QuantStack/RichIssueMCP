"""FastMCP server for accessing Rich Issues database."""

import gzip
import json
import re
from datetime import datetime
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


def _get_agent_db_file() -> str:
    """Get agent database file path."""
    try:
        base_path = get_data_file_path("agent-issues-jupyterlab-jupyterlab.json.gz")
        return str(base_path)
    except (FileNotFoundError, ValueError):
        # Fallback if config not available
        return "data/agent-issues-jupyterlab-jupyterlab.json.gz"


def _get_preferred_db_file(db_file: str | None = None) -> str:
    """Get preferred database file - agent DB if exists, otherwise default."""
    if db_file is not None:
        return db_file

    agent_db = _get_agent_db_file()
    if Path(agent_db).exists():
        return agent_db

    return _get_default_db_file()


def _create_agent_database() -> str:
    """Create agent database as copy of enriched database if it doesn't exist."""
    agent_db_path = _get_agent_db_file()

    if Path(agent_db_path).exists():
        return agent_db_path

    # Copy from enriched database
    enriched_db_path = _get_default_db_file()
    if not Path(enriched_db_path).exists():
        raise FileNotFoundError(f"Source database not found: {enriched_db_path}")

    # Load enriched data
    with gzip.open(enriched_db_path, "rt") as f:
        issues = json.load(f)

    # Add empty recommendations field to each issue
    for issue in issues:
        issue["recommendations"] = []

    # Save as agent database
    Path(agent_db_path).parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(agent_db_path, "wt") as f:
        json.dump(issues, f)

    # Clear cache for the new file
    if agent_db_path in _issues_cache:
        del _issues_cache[agent_db_path]

    return agent_db_path


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
    db_file = _get_preferred_db_file(db_file)
    issues = load_issues_db(db_file)
    issue = next((i for i in issues if i["number"] == issue_number), None)
    if not issue:
        return None
    return {
        "number": issue["number"],
        "summary": issue.get("summary"),
        "conversation": issue.get("conversation"),
        "recommendations": issue.get("recommendations", []),
    }


@mcp.tool()
def find_similar_issues(
    issue_number: int,
    threshold: float = 0.8,
    limit: int = 10,
    db_file: str | None = None,
) -> list[dict[str, Any]]:
    """Find issues similar to target issue using embeddings."""
    db_file = _get_preferred_db_file(db_file)
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
    db_file = _get_preferred_db_file(db_file)
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
            "url": i.get("url"),
        }
        for i in issues
        if i["number"] in linked_numbers
    ]


@mcp.tool()
def get_issue_metrics(
    issue_number: int, db_file: str | None = None
) -> dict[str, Any] | None:
    """Get enrichment metrics for a specific issue."""
    db_file = _get_preferred_db_file(db_file)
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
    db_file = _get_preferred_db_file(db_file)
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
    db_file = _get_preferred_db_file(db_file)
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
            "url": issue.get("url"),
        }
        for issue in sorted_issues[:limit]
    ]


@mcp.tool()
def export_all_open_issues(
    output_path: str,
    db_file: str | None = None,
) -> dict[str, Any]:
    """Export all open issues to a JSON file with name, title, url, and summary."""
    db_file = _get_preferred_db_file(db_file)
    issues = load_issues_db(db_file)

    if not issues:
        return {"status": "error", "message": "No issues found in database"}

    # Filter for open issues (state == "OPEN")
    open_issues = [issue for issue in issues if issue.get("state") == "OPEN"]

    # Create the output data structure
    export_data = []
    for issue in open_issues:
        export_data.append(
            {
                "name": f"#{issue['number']}",
                "title": issue.get("title", ""),
                "url": issue.get("url", ""),
                "summary": issue.get("summary", ""),
            }
        )

    # Write to JSON file
    try:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(export_data, f, indent=2)

        return {
            "status": "success",
            "message": f"Exported {len(export_data)} open issues to {output_path}",
            "count": len(export_data),
            "file_path": str(output_file.absolute()),
        }
    except Exception as e:
        return {"status": "error", "message": f"Failed to write file: {e}"}


@mcp.tool()
def add_recommendation(
    issue_number: int,
    severity: str,
    frequency: str,
    prevalence: str,
    report: str,
    recommendation: str,
    solution_complexity: str,
    solution_risk: str,
    db_file: str | None = None,
) -> dict[str, Any]:
    """Add a recommendation to an issue in the agent database."""

    # Validate input parameters
    valid_levels = {"low", "medium", "high"}
    valid_recommendations = {"keep", "close", "deprioritize", "prioritize"}

    errors = []
    if severity not in valid_levels:
        errors.append(f"severity must be one of: {', '.join(sorted(valid_levels))}")
    if frequency not in valid_levels:
        errors.append(f"frequency must be one of: {', '.join(sorted(valid_levels))}")
    if prevalence not in valid_levels:
        errors.append(f"prevalence must be one of: {', '.join(sorted(valid_levels))}")
    if solution_complexity not in valid_levels:
        errors.append(f"solution_complexity must be one of: {', '.join(sorted(valid_levels))}")
    if solution_risk not in valid_levels:
        errors.append(f"solution_risk must be one of: {', '.join(sorted(valid_levels))}")
    if recommendation not in valid_recommendations:
        errors.append(
            f"recommendation must be one of: {', '.join(sorted(valid_recommendations))}"
        )
    if not isinstance(report, str) or not report.strip():
        errors.append("report must be a non-empty string")
    if not isinstance(issue_number, int):
        errors.append("issue_number must be an integer")

    if errors:
        return {"status": "error", "message": "Validation failed", "errors": errors}

    # Create agent database if it doesn't exist
    try:
        agent_db_path = _create_agent_database()
    except Exception as e:
        return {"status": "error", "message": f"Failed to create agent database: {e}"}

    # Load issues from agent database
    if db_file is not None:
        agent_db_path = db_file

    try:
        issues = load_issues_db(agent_db_path)
    except Exception as e:
        return {"status": "error", "message": f"Failed to load database: {e}"}

    # Find the issue
    issue = next((i for i in issues if i["number"] == issue_number), None)
    if not issue:
        return {"status": "error", "message": f"Issue #{issue_number} not found"}

    # Ensure recommendations field exists
    if "recommendations" not in issue:
        issue["recommendations"] = []

    # Create new recommendation
    new_recommendation = {
        "severity": severity,
        "frequency": frequency,
        "prevalence": prevalence,
        "report": report.strip(),
        "recommendation": recommendation,
        "solution_complexity": solution_complexity,
        "solution_risk": solution_risk,
        "timestamp": datetime.now().isoformat(),
    }

    # Add recommendation
    issue["recommendations"].append(new_recommendation)
    recommendation_count = len(issue["recommendations"])

    # Save updated database
    try:
        with gzip.open(agent_db_path, "wt") as f:
            json.dump(issues, f)

        # Clear cache to force reload
        if agent_db_path in _issues_cache:
            del _issues_cache[agent_db_path]

        # Generate ordinal number text
        ordinals = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth"}
        ordinal_text = ordinals.get(recommendation_count, f"{recommendation_count}th")

        return {
            "status": "success",
            "message": f"Added {ordinal_text} recommendation for issue #{issue_number}",
            "issue_number": issue_number,
            "recommendation_count": recommendation_count,
            "recommendation": new_recommendation,
        }

    except Exception as e:
        return {"status": "error", "message": f"Failed to save database: {e}"}


@mcp.tool()
def get_first_issue_without_recommendation(db_file: str | None = None) -> dict[str, Any] | None:
    """Get the first issue without any recommendations."""
    db_file = _get_preferred_db_file(db_file)
    issues = load_issues_db(db_file)

    if not issues:
        return None

    # Find first issue without recommendations
    for issue in issues:
        recommendations = issue.get("recommendations", [])
        if not recommendations or len(recommendations) == 0:
            return {
                "number": issue["number"],
                "title": issue.get("title"),
                "summary": issue.get("summary"),
                "url": issue.get("url"),
                "state": issue.get("state"),
                "recommendations": recommendations
            }

    return None


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
