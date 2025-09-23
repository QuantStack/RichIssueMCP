#!/usr/bin/env python3
"""Pull GitHub issues using REST API with timeline for cross-references."""

import gzip
import json
import os
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import requests
import toml

from rich_issue_mcp.config import get_cache, get_data_directory


def get_github_token() -> str | None:
    """Get GitHub token from config or environment."""
    # Try config file first
    config_path = Path("config.toml")
    if config_path.exists():
        try:
            config = toml.load(config_path)
            # Try both locations for token
            token = config.get("github", {}).get("token") or config.get("token")
            if token:
                return token
        except Exception:
            pass
    else:
        raise ValueError(
            "No GitHub token found. Set token in config.toml, use 'gh auth login', or set GITHUB_TOKEN"
        )


def make_rest_request(url: str, params: dict[str, Any] | None = None, max_retries: int = 5) -> requests.Response:
    """Make a REST API request to GitHub with rate limit handling."""
    token = get_github_token()
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "RichIssueMCP/1.0",
    }
    
    for attempt in range(max_retries):
        response = requests.get(url, headers=headers, params=params)
        
        # If successful, return immediately
        if response.status_code == 200:
            return response
        
        # Handle rate limit errors
        if response.status_code == 403:
            try:
                error_data = response.json()
                if "rate limit" in error_data.get("message", "").lower():
                    # Calculate wait time: start with 60 seconds, double each retry
                    wait_time = 60 * (2 ** attempt)
                    
                    # Check if we have rate limit reset info in headers
                    rate_limit_reset = response.headers.get('X-RateLimit-Reset')
                    if rate_limit_reset:
                        reset_time = int(rate_limit_reset)
                        current_time = int(time.time())
                        time_until_reset = reset_time - current_time + 10  # Add 10 second buffer
                        
                        # Use the shorter of exponential backoff or time until reset
                        wait_time = min(wait_time, max(time_until_reset, 60))
                    
                    print(f"⚠️  Rate limit hit (attempt {attempt + 1}/{max_retries}). Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
            except (ValueError, KeyError):
                # Not a JSON response or missing fields, treat as non-rate-limit 403
                pass
        
        # For non-rate-limit errors or final attempt, return the response
        if attempt == max_retries - 1 or response.status_code != 403:
            return response
        
        # For other 403 errors, wait a short time before retry
        print(f"⚠️  Request failed with 403 (attempt {attempt + 1}/{max_retries}). Waiting 30 seconds...")
        time.sleep(30)
    
    return response


def get_timeline_cross_references(repo: str, issue_number: int) -> list[dict[str, Any]]:
    """Get cross-referenced issues and PRs using REST API timeline."""
    try:
        # Use REST API to get timeline with cross-referenced events
        timeline_url = f"https://api.github.com/repos/{repo}/issues/{issue_number}/timeline"
        
        response = make_rest_request(timeline_url, {"per_page": 100})
        
        if response.status_code != 200:
            print(f"⚠️  Failed to get timeline for issue #{issue_number}: {response.status_code}")
            return []
        
        timeline_events = response.json()
        
        # Filter and transform cross-referenced events
        cross_references = []
        for event in timeline_events:
            if event.get("event") == "cross-referenced" and event.get("source"):
                source = event["source"]
                
                # Determine if it's an issue or PR
                if "issue" in source:
                    item = source["issue"]
                    item_type = "issue"
                elif "pull_request" in source:
                    item = source["pull_request"]
                    item_type = "pr"
                else:
                    continue
                
                cross_ref = {
                    "number": item.get("number"),
                    "type": item_type,
                    "title": item.get("title"),
                    "url": item.get("html_url"),
                    "author": item.get("user", {}).get("login")
                }
                cross_references.append(cross_ref)
        
        return cross_references
        
    except Exception as e:
        print(f"⚠️  Failed to get cross-references for issue #{issue_number}: {e}")
        return []


@get_cache().memoize(typed=True)  # Cache with no expiration, type-sensitive
def fetch_issues_chunk_rest(
    repo: str,
    date_range: tuple[str, str],
    include_closed: bool = True,
    limit: int | None = 500,
) -> list[dict[str, Any]]:
    """Fetch a chunk of issues from GitHub using REST API search with date range filtering."""
    start_date, end_date = date_range
    
    print(f"🔍 Fetching chunk {start_date}..{end_date} (will use cache if available)")
    
    # Build search query with date range
    state_filter = "is:issue"
    if not include_closed:
        state_filter += " is:open"
    
    search_query = f"repo:{repo} {state_filter} updated:{start_date}..{end_date}"
    
    # Search for issues
    search_url = "https://api.github.com/search/issues"
    search_params = {
        "q": search_query,
        "per_page": min(limit if limit else 100, 100),
        "sort": "updated",
        "order": "desc"
    }
    
    response = make_rest_request(search_url, search_params)
    
    if response.status_code != 200:
        raise Exception(f"REST API search failed: {response.status_code} - {response.text}")
    
    data = response.json()
    issues = data["items"]
    
    print(f"    📄 REST search: {len(issues)} issues, Rate limit: {response.headers.get('X-RateLimit-Remaining', 'unknown')}")
    
    # Transform issues and fetch additional data
    transformed_issues = []
    
    for issue in issues:
        issue_number = issue["number"]
        
        # Get comments for this issue
        comments_url = f"https://api.github.com/repos/{repo}/issues/{issue_number}/comments"
        comments_response = make_rest_request(comments_url, {"per_page": 100})
        
        # Get comments as a simple list
        comments_list = []
        if comments_response.status_code == 200:
            comments = comments_response.json()
            comments_list = [
                {
                    "id": str(comment["id"]),
                    "body": comment["body"],
                    "createdAt": comment["created_at"],
                    "updatedAt": comment["updated_at"],
                    "author": {"login": comment["user"]["login"]},
                    "authorAssociation": comment.get("author_association", "NONE"),
                    "reactions": {"totalCount": comment.get("reactions", {}).get("total_count", 0)}
                }
                for comment in comments
            ]
        
        # Get cross-references using gh CLI timeline API
        cross_references = get_timeline_cross_references(repo, issue_number)
        
        # Transform to match expected format
        transformed_issue = {
            "number": issue["number"],
            "title": issue["title"],
            "body": issue["body"],
            "state": issue["state"],
            "createdAt": issue["created_at"],
            "updatedAt": issue["updated_at"],
            "url": issue["html_url"],
            "author": {"login": issue["user"]["login"]},
            "labels": [
                {"name": label["name"], "color": label["color"]}
                for label in issue.get("labels", [])
            ],
            "assignees": [
                {"login": assignee["login"]}
                for assignee in issue.get("assignees", [])
            ],
            "comments": comments_list,  # Direct list of comments
            "number_of_comments": len(comments_list),  # Separate count column
            "reactionGroups": [],  # REST API doesn't provide detailed reaction groups easily
            "cross_references": cross_references
        }
        
        transformed_issues.append(transformed_issue)
    
    return transformed_issues


def generate_date_ranges_rest(
    start_date: str, end_date: str | None = None, days: int = 7
) -> list[tuple[str, str]]:
    """Generate date ranges for REST API search filtering (start..end pairs)."""
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
        start_str = current.strftime("%Y-%m-%d")
        end_str = next_chunk.strftime("%Y-%m-%d")
        ranges.append((start_str, end_str))
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
    include_closed: bool = True,
    limit: int | None = 500,
    start_date: str = "2025-01-01",
    chunk_days: int = 7,
    include_cross_references: bool = True,
    refetch: bool = False,
) -> list[dict[str, Any]]:
    """Fetch issues from GitHub using REST API with date range chunking."""

    # Load existing issues to merge with new/updated ones
    existing_issues = []
    try:
        data_dir = get_data_directory()
        repo_name = repo.replace("/", "-")
        existing_file = data_dir / f"raw-issues-{repo_name}.json.gz"

        if existing_file.exists() and not refetch:
            with gzip.open(existing_file, "rt", encoding="utf-8") as f:
                existing_issues = json.load(f)
            print(f"📂 Loaded {len(existing_issues)} existing issues for merging")
        elif refetch:
            print("🔄 Refetch requested - will fetch all issues from start date")
    except Exception as e:
        print(f"⚠️  Could not load existing issues: {e}")
        existing_issues = []

    date_ranges = generate_date_ranges_rest(start_date, days=chunk_days)
    print(
        f"📊 Fetching issues in {len(date_ranges)} chunks of {chunk_days} days each (REST API)"
    )

    all_issues = []
    total_cross_refs = 0
    total_comments = 0

    for i, date_range in enumerate(date_ranges, 1):
        start_date_str, end_date_str = date_range
        print(
            f"🔍 Fetching chunk {i}/{len(date_ranges)}: {start_date_str}..{end_date_str}"
        )

        chunk_issues = fetch_issues_chunk_rest(
            repo=repo,
            date_range=date_range,
            include_closed=include_closed,
            limit=limit,
        )

        # Count cross-references and comments
        chunk_cross_refs = 0
        chunk_comments = 0
        issues_with_refs = 0
        for issue in chunk_issues:
            cross_refs = len(issue.get("cross_references", []))
            if cross_refs > 0:
                issues_with_refs += 1
                chunk_cross_refs += cross_refs

            # Count total comments with text
            comment_count = issue.get("number_of_comments", 0)
            chunk_comments += comment_count

        print(f"  📥 Retrieved {len(chunk_issues)} issues")
        print(f"  💬 Comments: {chunk_comments} total with full text")
        if include_cross_references:
            print(
                f"  🔗 Cross-references: {chunk_cross_refs} total, {issues_with_refs} issues with refs"
            )

        all_issues.extend(chunk_issues)
        total_cross_refs += chunk_cross_refs
        total_comments += chunk_comments

    print(f"📊 Total issues fetched: {len(all_issues)}")
    print(f"💬 Total comments with full text: {total_comments}")

    # Merge with existing issues - updatedAt filtering ensures we only get new/changed issues
    merged_issues = merge_issues(existing_issues, all_issues)
    print(f"📋 Final unique issue count: {len(merged_issues)}")

    if include_cross_references:
        print(f"🔗 Total cross-references found: {total_cross_refs}")
        print("✅ Cross-references fetched via REST timeline API")

    return merged_issues


def save_raw_issues(issues: list[dict[str, Any]], filepath: Path) -> None:
    """Save raw issues to gzipped JSON file."""
    filepath.parent.mkdir(exist_ok=True)

    with gzip.open(filepath, "wt", encoding="utf-8") as f:
        json.dump(issues, f, indent=None, separators=(",", ":"))
