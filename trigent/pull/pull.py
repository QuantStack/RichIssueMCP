#!/usr/bin/env python3
"""Pull GitHub issues and save raw data."""

import argparse
import gzip
import json
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


def fetch_issues_chunk(
    repo: str,
    include_closed: bool = False,
    limit: int | None = None,
    updated_since: str | None = None,
    updated_until: str | None = None,
) -> list[dict[str, Any]]:
    """Fetch a chunk of issues from GitHub using gh CLI with date filtering."""
    state = "all" if include_closed else "open"

    # Build search query with date filters
    search_parts = []
    if updated_since:
        search_parts.append(f"updated:>={updated_since}")
    if updated_until:
        search_parts.append(f"updated:<={updated_until}")

    search_query = " ".join(search_parts) if search_parts else None

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
    ]

    if search_query:
        cmd.extend(["--search", search_query])

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


def load_state(state_file: Path) -> dict[str, Any]:
    """Load state from JSON file."""
    if not state_file.exists():
        return {}

    try:
        with open(state_file) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def save_state(state: dict[str, Any], state_file: Path) -> None:
    """Save state to JSON file."""
    state_file.parent.mkdir(exist_ok=True)
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)


def generate_date_ranges(
    start_date: str, end_date: str | None = None
) -> list[tuple[str, str]]:
    """Generate weekly date ranges from start_date to end_date (or now)."""
    start = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
    end = (
        datetime.now()
        if end_date is None
        else datetime.fromisoformat(end_date.replace("Z", "+00:00"))
    )

    ranges = []
    current = start

    while current < end:
        next_week = min(current + timedelta(days=7), end)
        ranges.append((current.strftime("%Y-%m-%d"), next_week.strftime("%Y-%m-%d")))
        current = next_week

    return ranges


def merge_issues(
    existing_issues: list[dict[str, Any]], new_issues: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Merge new issues with existing ones, updating duplicates based on issue number."""
    # Create a mapping of issue number to issue for existing issues
    issue_map = {issue["number"]: issue for issue in existing_issues}

    # Update with new issues
    for issue in new_issues:
        issue_map[issue["number"]] = issue

    # Return sorted by issue number
    return sorted(issue_map.values(), key=lambda x: x["number"])


def fetch_issues(
    repo: str,
    include_closed: bool = False,
    limit: int | None = None,
    start_date: str = "2025-01-01",
    refetch: bool = False,
) -> list[dict[str, Any]]:
    """Fetch issues from GitHub using paged requests with weekly chunks."""
    repo_name = repo.replace("/", "-")
    state_file = Path("data") / f"state-{repo_name}.json"
    output_file = Path("data") / f"raw-issues-{repo_name}.json.gz"

    # Load existing state and issues
    state = load_state(state_file)
    existing_issues = []

    if not refetch and output_file.exists():
        try:
            with gzip.open(output_file, "rt", encoding="utf-8") as f:
                existing_issues = json.load(f)
            print(f"📂 Loaded {len(existing_issues)} existing issues")
        except (OSError, json.JSONDecodeError) as e:
            print(f"⚠️  Could not load existing issues: {e}")
            existing_issues = []

    # Determine start date for fetching
    if refetch:
        fetch_start_date = start_date
        print(f"🔄 Full refetch requested, starting from {fetch_start_date}")
    else:
        fetch_start_date = state.get("last_updated_at", start_date)
        print(f"📅 Incremental fetch from {fetch_start_date}")

    # Generate date ranges for weekly chunks
    date_ranges = generate_date_ranges(fetch_start_date)

    if not date_ranges:
        print("✅ No new date ranges to fetch")
        return existing_issues

    print(f"📊 Fetching {len(date_ranges)} weekly chunks")

    all_new_issues = []
    latest_updated_at = fetch_start_date

    for i, (since_date, until_date) in enumerate(date_ranges, 1):
        print(f"🔍 Fetching chunk {i}/{len(date_ranges)}: {since_date} to {until_date}")

        chunk_issues = fetch_issues_chunk(
            repo=repo,
            include_closed=include_closed,
            limit=limit,
            updated_since=since_date,
            updated_until=until_date,
        )

        print(f"  📥 Retrieved {len(chunk_issues)} issues")
        all_new_issues.extend(chunk_issues)

        # Track the latest updatedAt timestamp
        for issue in chunk_issues:
            if issue.get("updatedAt") and issue["updatedAt"] > latest_updated_at:
                latest_updated_at = issue["updatedAt"]

    print(f"📊 Total new/updated issues: {len(all_new_issues)}")

    # Merge with existing issues
    merged_issues = merge_issues(existing_issues, all_new_issues)
    print(f"📋 Final issue count: {len(merged_issues)}")

    # Update state
    state["last_updated_at"] = latest_updated_at
    state["last_fetch_time"] = datetime.now().isoformat()
    save_state(state, state_file)

    return merged_issues


def save_raw_issues(issues: list[dict[str, Any]], filepath: Path) -> None:
    """Save raw issues to gzipped JSON file."""
    filepath.parent.mkdir(exist_ok=True)

    with gzip.open(filepath, "wt", encoding="utf-8") as f:
        json.dump(issues, f, indent=None, separators=(",", ":"))


def main() -> None:
    """Main pull entry point."""
    parser = argparse.ArgumentParser(description="Pull GitHub issues and save raw data")
    parser.add_argument(
        "repo", nargs="?", default="jupyterlab/jupyterlab", help="Repository to analyze"
    )
    parser.add_argument(
        "--include-closed", "-c", action="store_true", help="Include closed issues"
    )
    parser.add_argument("--limit", "-l", type=int, help="Limit number of issues")
    parser.add_argument(
        "--output", help="Output file path (default: data/raw-issues-{repo}.json.gz)"
    )
    parser.add_argument(
        "--start-date",
        default="2025-01-01",
        help="Start date for fetching issues (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--refetch", action="store_true", help="Refetch all issues from start date"
    )

    args = parser.parse_args()

    print(f"🔍 Fetching issues from {args.repo}...")

    raw_issues = fetch_issues(
        repo=args.repo,
        include_closed=args.include_closed,
        limit=args.limit,
        start_date=args.start_date,
        refetch=args.refetch,
    )
    print(f"📥 Retrieved {len(raw_issues)} issues")

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        repo_name = args.repo.replace("/", "-")
        output_path = Path("data") / f"raw-issues-{repo_name}.json.gz"

    save_raw_issues(raw_issues, output_path)
    print(f"✅ Raw issue database saved to {output_path}")


if __name__ == "__main__":
    main()
