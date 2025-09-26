#!/usr/bin/env python3
"""Pull GitHub issues using Issues REST API with intelligent page-based caching."""

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import requests
import toml

from rich_issue_mcp.database import load_issues


def get_last_updated_date(repo: str) -> datetime | None:
    """Get the most recent updated date from existing issues in database."""
    try:
        existing_issues = load_issues(repo)
        if not existing_issues:
            return None

        updated_dates = [
            datetime.fromisoformat(issue["updatedAt"].replace("Z", "+00:00"))
            for issue in existing_issues
            if "updatedAt" in issue
        ]

        if not updated_dates:
            return None

        return max(updated_dates)
    except Exception as e:
        print(f"⚠️  Could not get last updated date: {e}")
        return None


def get_existing_issue_numbers(repo: str) -> set[int]:
    """Get set of existing issue numbers in database to avoid re-pulling."""
    try:
        existing_issues = load_issues(repo)
        return {issue["number"] for issue in existing_issues if "number" in issue}
    except Exception:
        return set()


def get_database_coverage(
    repo: str, mode: str = "update"
) -> tuple[datetime, datetime] | None:
    """Get the date range covered by existing database issues.

    Args:
        repo: GitHub repository in owner/repo format
        mode: Either 'create' or 'update' - determines which timestamp field to use
    """
    try:
        existing_issues = load_issues(repo)
        if not existing_issues:
            return None

        # Use different timestamp field based on mode
        date_field = "createdAt" if mode == "create" else "updatedAt"
        dates = [
            datetime.fromisoformat(issue[date_field].replace("Z", "+00:00"))
            for issue in existing_issues
            if date_field in issue
        ]

        if not dates:
            return None

        return (min(dates), max(dates))
    except Exception as e:
        print(f"⚠️  Could not analyze database coverage: {e}")
        return None


def get_github_token() -> str:
    """Get GitHub token from config or environment."""
    # Try config file first
    config_path = Path("config.toml")
    if config_path.exists():
        try:
            config = toml.load(config_path)
            # Try both locations for token
            token = config.get("github", {}).get("token") or config.get("token")
            if token:
                return str(token)
        except Exception:
            pass

    raise ValueError(
        "No GitHub token found. Set token in config.toml, use 'gh auth login', or set GITHUB_TOKEN"
    )


def make_rest_request(
    url: str, params: dict[str, Any] | None = None, max_retries: int = 5
) -> requests.Response:
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
                    # Check if we have rate limit reset info in headers
                    rate_limit_reset = response.headers.get("X-RateLimit-Reset")
                    remaining = response.headers.get("X-RateLimit-Remaining", "0")

                    if rate_limit_reset:
                        reset_time = int(rate_limit_reset)
                        current_time = int(time.time())
                        time_until_reset = (
                            reset_time - current_time + 10
                        )  # Add 10 second buffer

                        # Wait until rate limit resets
                        wait_time = max(time_until_reset, 60)  # Minimum 60 seconds

                        reset_datetime = datetime.fromtimestamp(reset_time)
                        print(
                            f"⚠️  Rate limit exceeded (remaining: {remaining}). "
                            f"Waiting {wait_time} seconds until reset at {reset_datetime.strftime('%H:%M:%S')}..."
                        )
                    else:
                        # Fallback: exponential backoff
                        wait_time = 60 * (2**attempt)
                        print(
                            f"⚠️  Rate limit hit (attempt {attempt + 1}/{max_retries}). "
                            f"Waiting {wait_time} seconds..."
                        )

                    time.sleep(wait_time)
                    continue
            except (ValueError, KeyError):
                # Not a JSON response or missing fields, treat as non-rate-limit 403
                pass

        # For non-rate-limit errors or final attempt, return the response
        if attempt == max_retries - 1 or response.status_code != 403:
            return response

        # For other 403 errors, wait a short time before retry
        print(
            f"⚠️  Request failed with 403 (attempt {attempt + 1}/{max_retries}). Waiting 30 seconds..."
        )
        time.sleep(30)

    return response


def fetch_issues_page(
    repo: str,
    page: int,
    include_closed: bool = True,
    since: str | None = None,
    mode: str = "update",
) -> list[dict[str, Any]]:
    """Fetch a single page of issues using Issues API.

    Args:
        repo: GitHub repository in owner/repo format
        page: Page number to fetch
        include_closed: Whether to include closed issues
        since: ISO timestamp to fetch issues since
        mode: Either 'create' (sort by created) or 'update' (sort by updated)
    """
    issues_url = f"https://api.github.com/repos/{repo}/issues"

    # Set sort parameters based on mode
    sort_field = "created" if mode == "create" else "updated"

    params = {
        "sort": sort_field,
        "direction": "asc",
        "per_page": 100,
        "page": page,
        "state": "all" if include_closed else "open",
    }

    if since:
        params["since"] = since

    response = make_rest_request(issues_url, params)

    if response.status_code != 200:
        raise Exception(f"Issues API failed: {response.status_code} - {response.text}")

    issues = response.json()

    # Get rate limit info
    remaining = response.headers.get("X-RateLimit-Remaining", "unknown")
    reset_timestamp = response.headers.get("X-RateLimit-Reset")

    # Get pagination info from Link header if available
    link_header = response.headers.get("Link", "")
    total_pages = "unknown"
    if 'rel="last"' in link_header:
        # Extract last page number from Link header
        import re

        last_match = re.search(r'page=(\d+)[^>]*>;\s*rel="last"', link_header)
        if last_match:
            total_pages = last_match.group(1)

    if reset_timestamp:
        reset_time = datetime.fromtimestamp(int(reset_timestamp))
        print(
            f"📄 Fetched page {page}/{total_pages}: {len(issues)} issues, Rate limit: {remaining} remaining (resets at {reset_time.strftime('%H:%M:%S')})"
        )
    else:
        print(
            f"📄 Fetched page {page}/{total_pages}: {len(issues)} issues, Rate limit: {remaining} remaining"
        )

    return issues


def filter_new_issues_for_create_mode(
    page_issues: list[dict[str, Any]], existing_numbers: set[int]
) -> list[dict[str, Any]]:
    """Filter out issues that already exist in database for create mode."""
    return [issue for issue in page_issues if issue["number"] not in existing_numbers]


def page_needs_processing(
    page_issues: list[dict[str, Any]],
    coverage: tuple[datetime, datetime] | None,
    mode: str = "update",
) -> bool:
    """Check if any issue in the page needs processing (outside existing coverage).

    Uses a one-day margin to handle cases where many issues have the same timestamp,
    which can cause skipped pages when some issues with identical timestamps are missing.

    Args:
        page_issues: List of issues from the current page
        coverage: Database coverage range (min_date, max_date) or None
        mode: Either 'create' or 'update' - determines which timestamp field to check
    """
    if not coverage or not page_issues:
        return True

    earliest_db, latest_db = coverage

    # Add one-day margin to coverage range to avoid skipping issues with identical timestamps
    earliest_with_margin = earliest_db - timedelta(days=1)
    latest_with_margin = latest_db + timedelta(days=1)

    # Use different timestamp field based on mode
    timestamp_field = "created_at" if mode == "create" else "updated_at"

    # Check if any issue in this page is outside the database coverage (with margin)
    for issue in page_issues:
        issue_timestamp = datetime.fromisoformat(
            issue[timestamp_field].replace("Z", "+00:00")
        )
        if (
            issue_timestamp < earliest_with_margin
            or issue_timestamp > latest_with_margin
        ):
            return True

    return False


def fetch_all_comments(repo: str, issue_number: int) -> list[dict[str, Any]] | None:
    """Fetch all comments for an issue with pagination.

    Returns:
        List of comments if successful, None if failed to fetch
    """
    all_comments = []
    page = 1

    while True:
        comments_url = (
            f"https://api.github.com/repos/{repo}/issues/{issue_number}/comments"
        )
        response = make_rest_request(comments_url, {"per_page": 100, "page": page})

        if response.status_code != 200:
            print(
                f"⚠️  Failed to get comments for issue #{issue_number}: {response.status_code}"
            )
            return None

        comments = response.json()
        if not comments:
            break

        all_comments.extend(comments)
        page += 1

    return [
        {
            "id": str(comment["id"]),
            "body": comment["body"],
            "createdAt": comment["created_at"],
            "updatedAt": comment["updated_at"],
            "author": {"login": comment["user"]["login"]},
            "authorAssociation": comment.get("author_association", "NONE"),
            "reactions": {
                "totalCount": comment.get("reactions", {}).get("total_count", 0)
            },
        }
        for comment in all_comments
    ]


def fetch_all_timeline_cross_references(
    repo: str, issue_number: int
) -> list[dict[str, Any]] | None:
    """Fetch all cross-references from timeline with pagination.

    Returns:
        List of cross-references if successful, None if failed to fetch
    """
    all_cross_references = []
    page = 1

    while True:
        timeline_url = (
            f"https://api.github.com/repos/{repo}/issues/{issue_number}/timeline"
        )
        response = make_rest_request(timeline_url, {"per_page": 100, "page": page})

        if response.status_code != 200:
            print(
                f"⚠️  Failed to get timeline for issue #{issue_number}: {response.status_code}"
            )
            return None

        timeline_events = response.json()
        if not timeline_events:
            break

        # Filter cross-referenced events
        for event in timeline_events:
            if event.get("event") == "cross-referenced" and event.get("source"):
                source = event["source"]

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
                    "author": item.get("user", {}).get("login"),
                }
                all_cross_references.append(cross_ref)

        page += 1

    return all_cross_references


def process_and_save_issue(repo: str, issue: dict[str, Any]) -> dict[str, Any] | None:
    """Process a single issue by fetching comments and cross-references, then save to database.

    Returns:
        Processed issue if successful, None if comments or cross-references could not be fetched
    """
    from rich_issue_mcp.database import upsert_issues

    issue_number = issue["number"]

    # Fetch all comments with pagination
    comments_list = fetch_all_comments(repo, issue_number)
    if comments_list is None:
        print(f"  ✗ Issue #{issue_number}: Failed to fetch comments - skipping save")
        return None

    # Fetch all cross-references with pagination
    cross_references = fetch_all_timeline_cross_references(repo, issue_number)
    if cross_references is None:
        print(
            f"  ✗ Issue #{issue_number}: Failed to fetch cross-references - skipping save"
        )
        return None

    # Transform to expected format
    processed_issue = {
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
            {"login": assignee["login"]} for assignee in issue.get("assignees", [])
        ],
        "comments": comments_list,
        "number_of_comments": len(comments_list),
        "reactionGroups": [],  # Issues API doesn't provide detailed reaction groups easily
        "cross_references": cross_references,
    }

    # Save immediately to database
    upsert_issues(repo, [processed_issue])

    print(
        f"  ✓ Issue #{issue_number} (updated: {issue['updated_at']}): {len(comments_list)} comments, {len(cross_references)} cross-refs - saved to database"
    )

    return processed_issue


def process_page_issues(
    repo: str, page_issues: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Process a page of issues by fetching comments and cross-references with pagination."""
    processed_issues = []

    for issue in page_issues:
        processed_issue = process_and_save_issue(repo, issue)
        if processed_issue is not None:
            processed_issues.append(processed_issue)

    return processed_issues


def fetch_specific_issue(repo: str, issue_number: int) -> dict[str, Any] | None:
    """Fetch a specific issue by number using Issues API."""
    issue_url = f"https://api.github.com/repos/{repo}/issues/{issue_number}"

    response = make_rest_request(issue_url)

    if response.status_code == 404:
        print(f"⚠️  Issue #{issue_number} not found")
        return None
    elif response.status_code != 200:
        print(
            f"⚠️  Failed to fetch issue #{issue_number}: {response.status_code} - {response.text}"
        )
        return None

    return response.json()


def fetch_specific_issues(repo: str, issue_numbers: list[int]) -> list[dict[str, Any]]:
    """Fetch specific issues by number and process them."""
    print(
        f"🎯 Fetching {len(issue_numbers)} specific issues: {', '.join(map(str, issue_numbers))}"
    )

    processed_issues = []

    for issue_number in issue_numbers:
        print(f"\n🔍 Fetching issue #{issue_number}...")

        # Fetch the issue
        issue = fetch_specific_issue(repo, issue_number)
        if issue is None:
            continue

        # Process and save the issue
        processed_issue = process_and_save_issue(repo, issue)
        if processed_issue is not None:
            processed_issues.append(processed_issue)

    return processed_issues


def fetch_issues(
    repo: str,
    include_closed: bool = True,
    limit: int | None = None,
    start_date: str | None = None,
    refetch: bool = False,
    mode: str = "update",
    issue_numbers: list[int] | None = None,
    **kwargs: Any,  # Backward compatibility for unused params
) -> list[dict[str, Any]]:
    """Fetch issues from GitHub using Issues API with intelligent page-based processing.

    Args:
        repo: GitHub repository in owner/repo format
        include_closed: Whether to include closed issues
        limit: Maximum number of issues to process (for testing)
        start_date: Start date in YYYY-MM-DD format
        refetch: Whether to refetch all issues from start date
        mode: Either 'create' or 'update'
            - 'create': Sort by created date, use start_date as since, avoid re-pulling existing issues
            - 'update': Pull from last updated date in database
        issue_numbers: Specific issue numbers to refetch (always refetches even if they exist)
    """

    # Handle specific issue numbers if provided
    if issue_numbers:
        print(f"🚀 Starting specific issue fetch for {repo}")
        specific_issues = fetch_specific_issues(repo, issue_numbers)

        # Load and return all issues in database
        try:
            final_issues = load_issues(repo)
            print("\n📋 Specific issue fetch complete:")
            print(f"  🎯 Fetched {len(specific_issues)} specific issues")
            print(f"  📁 Total issues in database: {len(final_issues)}")
            return final_issues
        except Exception as e:
            print(f"⚠️  Could not load final results: {e}")
            return specific_issues

    print(f"🚀 Starting Issues API fetch for {repo} in '{mode}' mode")

    # Parse start_date if provided
    start_datetime = None
    if start_date:
        try:
            start_datetime = datetime.fromisoformat(start_date + "T00:00:00+00:00")
        except ValueError as e:
            raise ValueError(
                f"Invalid start_date format: {start_date}. Expected YYYY-MM-DD"
            ) from e

    # Get existing issue numbers for create mode filtering
    existing_numbers = set() if refetch else get_existing_issue_numbers(repo)

    # Analyze existing database coverage
    coverage = None if refetch else get_database_coverage(repo, mode)

    # Determine the 'since' parameter for GitHub API based on mode
    since_param = None

    if mode == "create":
        # CREATE MODE: Use start_date as since, avoid re-pulling existing issues
        if start_datetime:
            since_param = start_datetime.isoformat()
            print(f"📅 CREATE mode: Using start_date as since: {start_datetime.date()}")
            if existing_numbers:
                print(f"📊 Will skip {len(existing_numbers)} existing issues")
        else:
            if coverage:
                earliest, latest = coverage
                print(
                    f"📊 Existing database covers: {earliest.date()} to {latest.date()}"
                )
                since_param = earliest.isoformat()
                print(
                    f"📅 CREATE mode: Using earliest created date as since: {earliest.date()}"
                )
            else:
                print("📊 CREATE mode: No existing database - will fetch all issues")
    else:
        # UPDATE MODE: Pull from last updated date in database
        if refetch and start_datetime:
            # If refetch with start_date, use start_date
            since_param = start_datetime.isoformat()
            print(
                f"📅 UPDATE mode (refetch): Using start_date as since: {start_datetime.date()}"
            )
        else:
            # Use last updated date from database
            last_updated = get_last_updated_date(repo)
            if last_updated:
                since_param = last_updated.isoformat()
                print(
                    f"📅 UPDATE mode: Using last updated date as since: {last_updated.date()}"
                )
            elif start_datetime:
                since_param = start_datetime.isoformat()
                print(
                    f"📅 UPDATE mode: No existing database, using start_date as since: {start_datetime.date()}"
                )
            else:
                print(
                    "📊 UPDATE mode: No existing database or start_date - will fetch all issues"
                )

    if coverage:
        earliest, latest = coverage
        print(f"📊 Existing database coverage: {earliest.date()} to {latest.date()}")

    page = 1
    total_processed = 0
    total_skipped = 0
    total_comments = 0
    total_cross_refs = 0

    while True:
        # Fetch page of issues
        print(f"\n🔍 Fetching page {page}...")
        page_issues = fetch_issues_page(repo, page, include_closed, since_param, mode)

        if not page_issues:
            print("✅ No more issues found - fetch complete")
            break

        # Filter out existing issues in create mode
        if mode == "create" and existing_numbers:
            original_count = len(page_issues)
            page_issues = filter_new_issues_for_create_mode(
                page_issues, existing_numbers
            )
            filtered_count = original_count - len(page_issues)
            if filtered_count > 0:
                print(
                    f"  ⏭️  Filtered out {filtered_count} existing issues ({len(page_issues)} new)"
                )

        if not page_issues:
            print("  ⏭️  All issues in page already exist - skipping")
            page += 1
            continue

        # Check if this page needs processing (for update mode with coverage)
        needs_processing = True
        if mode == "update":
            needs_processing = page_needs_processing(page_issues, coverage, mode)

        if needs_processing:
            print(f"🔧 Processing page {page} ({len(page_issues)} issues)...")
            processed_issues = process_page_issues(repo, page_issues)

            # Count stats
            page_comments = sum(len(issue["comments"]) for issue in processed_issues)
            page_cross_refs = sum(
                len(issue["cross_references"]) for issue in processed_issues
            )

            total_processed += len(processed_issues)
            total_comments += page_comments
            total_cross_refs += page_cross_refs

            print(
                f"  📊 Page stats: {page_comments} comments, {page_cross_refs} cross-refs"
            )

        else:
            print(f"⏭️  Skipping page {page} - all issues already in database")
            total_skipped += len(page_issues)

        page += 1

        # Apply limit if specified (for testing)
        if limit and total_processed >= limit:
            print(f"🛑 Reached limit of {limit} processed issues")
            break

    # Load final results
    try:
        final_issues = load_issues(repo)
        print("\n📋 Final results:")
        print(f"  📁 Total issues in database: {len(final_issues)}")
        print(f"  ✅ Processed this run: {total_processed}")
        print(f"  ⏭️  Skipped this run: {total_skipped}")
        print(f"  💬 Comments fetched: {total_comments}")
        print(f"  🔗 Cross-references fetched: {total_cross_refs}")
        print("  💾 Each issue saved to database immediately after processing")

        return final_issues
    except Exception as e:
        print(f"⚠️  Could not load final results: {e}")
        return []
