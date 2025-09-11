#!/usr/bin/env python3
"""Pull GitHub issues using updatedAt chunking."""

import gzip
import json
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


def fetch_issues_chunk(
    repo: str,
    updated_range: tuple[str, str],
    include_closed: bool = False,
    limit: int | None = 500,
) -> list[dict[str, Any]]:
    """Fetch a chunk of issues from GitHub using gh CLI with updatedAt date filtering."""
    state = "all" if include_closed else "open"
    updated_since, updated_until = updated_range
    
    # Use GitHub's date range syntax with updatedAt field
    search_query = f"updated:{updated_since}..{updated_until}"

    cmd = [
        "gh",
        "issue",
        "list",
        "--repo",
        repo,
        "--state",
        state,
        "--json",
        "number,title,body,state,createdAt,updatedAt,author,labels,assignees,url,comments,reactionGroups",
        "--search",
        search_query,
    ]
    
    if limit is not None:
        cmd.extend(["--limit", str(limit)])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"❌ GitHub CLI failed: {e}")
        print(f"Error output: {e.stderr}")
        raise
    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse JSON response: {e}")
        raise


def generate_date_ranges(
    start_date: str, end_date: str | None = None, days: int = 7
) -> list[tuple[str, str]]:
    """Generate date ranges from start_date to end_date (or now) in chunks of specified days."""
    start = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
    end = (
        datetime.now().replace(tzinfo=start.tzinfo)
        if end_date is None
        else datetime.fromisoformat(end_date.replace("Z", "+00:00"))
    )

    ranges = []
    current = start

    while current < end:
        next_chunk = min(current + timedelta(days=days), end)
        ranges.append((
            current.strftime("%Y-%m-%d"), 
            next_chunk.strftime("%Y-%m-%d")
        ))
        current = next_chunk

    return ranges


def merge_issues(
    existing_issues: list[dict[str, Any]], new_issues: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Merge new issues with existing ones, updating duplicates based on issue number."""
    issue_map = {issue["number"]: issue for issue in existing_issues}
    
    for issue in new_issues:
        issue_map[issue["number"]] = issue

    return sorted(issue_map.values(), key=lambda x: x["number"])


def fetch_issues(
    repo: str,
    include_closed: bool = False,
    limit: int | None = 500,
    start_date: str = "2025-01-01",
    chunk_days: int = 7,
) -> list[dict[str, Any]]:
    """Fetch issues from GitHub using updatedAt chunking with specified day ranges."""
    
    date_ranges = generate_date_ranges(start_date, days=chunk_days)
    print(f"📊 Fetching issues in {len(date_ranges)} chunks of {chunk_days} days each")

    all_issues = []

    for i, (since_date, until_date) in enumerate(date_ranges, 1):
        print(f"🔍 Fetching chunk {i}/{len(date_ranges)}: {since_date} to {until_date}")

        chunk_issues = fetch_issues_chunk(
            repo=repo,
            updated_range=(since_date, until_date),
            include_closed=include_closed,
            limit=limit,
        )

        print(f"  📥 Retrieved {len(chunk_issues)} issues")
        all_issues.extend(chunk_issues)

    print(f"📊 Total issues fetched: {len(all_issues)}")
    
    merged_issues = merge_issues([], all_issues)
    print(f"📋 Final unique issue count: {len(merged_issues)}")

    return merged_issues


def save_raw_issues(issues: list[dict[str, Any]], filepath: Path) -> None:
    """Save raw issues to gzipped JSON file."""
    filepath.parent.mkdir(exist_ok=True)

    with gzip.open(filepath, "wt", encoding="utf-8") as f:
        json.dump(issues, f, indent=None, separators=(",", ":"))
