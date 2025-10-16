#!/usr/bin/env python3
"""Main CLI entry point for Rich Issue MCP."""

import argparse

from rich_issue_mcp.config import get_config, get_data_directory
from rich_issue_mcp.database import load_issues, save_issues
from rich_issue_mcp.enrich import (
    add_k4_distances,
    add_quartile_columns,
    add_summaries,
    enrich_issue,
    perform_hdbscan_clustering,
    print_stats,
)
from rich_issue_mcp.mcp_server import run_mcp_server
from rich_issue_mcp.pull import fetch_issues
from rich_issue_mcp.validate import validate_database
from rich_issue_mcp.visualize import visualize_issues


def cmd_pull(args) -> None:
    """Execute pull command."""
    print(f"🔍 Fetching issues from {args.repo}...")

    raw_issues = fetch_issues(
        repo=args.repo,
        include_closed=not args.exclude_closed,  # Invert the flag
        limit=args.limit,
        start_date=getattr(args, "start_date", "2025-01-01"),
        refetch=getattr(args, "refetch", False),
        mode=getattr(args, "mode", "update"),
        issue_numbers=getattr(args, "issue_numbers", None),
        item_types=getattr(args, "item_types", "both"),
    )
    print(f"📥 Retrieved {len(raw_issues)} issues")

    save_issues(args.repo, raw_issues)
    print("✅ Raw issue database saved to TinyDB")


def cmd_enrich(args) -> None:
    """Execute enrich command."""
    print(f"🔍 Loading raw issues from {args.repo}...")
    raw_issues = load_issues(args.repo)
    print(f"📥 Retrieved {len(raw_issues)} issues")

    # Get API key from config
    config = get_config()
    api_key = config.get("api", {}).get("mistral_api_key")
    if not api_key:
        raise ValueError("Mistral API key required in config.toml [api] section")

    # Enrich issues
    enriched = [enrich_issue(issue, api_key, args.model) for issue in raw_issues]

    print("🔧 Computing quartile assignments...")
    enriched = add_quartile_columns(enriched)

    if not args.skip_summaries:
        print("📝 Generating AI summaries...")
        enriched = add_summaries(enriched, api_key)
    else:
        print("⏭️  Skipping AI summaries...")
        for issue in enriched:
            issue["summary"] = None

    print("🔧 Computing k-4 nearest neighbor distances...")
    enriched = add_k4_distances(enriched)

    print("🔍 Performing HDBSCAN clustering...")
    enriched = perform_hdbscan_clustering(enriched, api_key)

    save_issues(args.repo, enriched)
    print("✅ Enriched issue database saved to TinyDB")
    print_stats(enriched)


def cmd_mcp(args) -> None:
    """Execute MCP server."""
    run_mcp_server(host=args.host, port=args.port, repo=args.repo)


def cmd_visualize(args) -> None:
    """Execute visualize command."""
    print(f"📊 Visualizing repository: {args.repo}")

    visualize_issues(args.repo, args.output, scale=args.scale)


def cmd_clean(args) -> None:
    """Execute clean command to remove downloaded data."""
    from rich_issue_mcp.database import get_database_path

    if hasattr(args, 'repo') and args.repo:
        # Clean specific repository
        repo = args.repo
        db_path = get_database_path(repo)

        if not db_path.exists():
            print(f"📁 No database file found for {repo}")
            return

        files_to_delete = [db_path]
        print(f"🗑️  Database file for {repo}:")
    else:
        # Clean all repositories
        data_dir = get_data_directory()

        if not data_dir.exists():
            print("📁 No data directory found")
            return

        # Find data files (TinyDB database files)
        patterns = ["issues-*.db"]
        files_to_delete = []

        for pattern in patterns:
            files_to_delete.extend(data_dir.glob(pattern))

        if not files_to_delete:
            print("📁 No data files found to clean")
            return

        print("🗑️  Files to be deleted:")

    # Show files that would be deleted
    for file_path in sorted(files_to_delete):
        file_size = file_path.stat().st_size
        if file_size > 1024 * 1024:
            size_str = f"{file_size / (1024 * 1024):.1f} MB"
        elif file_size > 1024:
            size_str = f"{file_size / 1024:.1f} KB"
        else:
            size_str = f"{file_size} bytes"
        print(f"  - {file_path} ({size_str})")

    # Ask for confirmation unless --yes flag is used
    if not args.yes:
        response = input("\n❓ Delete these files? (y/N): ").strip().lower()
        if response not in ("y", "yes"):
            print("❌ Clean operation cancelled")
            return

    # Delete the files
    deleted_count = 0
    for file_path in files_to_delete:
        try:
            file_path.unlink()
            deleted_count += 1
            print(f"🗑️  Deleted {file_path}")
        except OSError as e:
            print(f"❌ Failed to delete {file_path}: {e}")

    if hasattr(args, 'repo') and args.repo:
        print(f"✅ Cleaned database for {args.repo}")
    else:
        print(f"✅ Cleaned {deleted_count} files from {get_data_directory()}")


def cmd_validate(args) -> None:
    """Execute validate command to check database integrity."""
    success = validate_database(
        args.repo, delete_invalid=getattr(args, "delete_invalid", False)
    )
    if not success:
        exit(1)


def cmd_tui(args) -> None:
    """Execute TUI command to browse database interactively."""
    from rich_issue_mcp.tui import run_tui

    try:
        run_tui(args.repo)
    except KeyboardInterrupt:
        print("\nTUI exited by user")
    except Exception as e:
        print(f"Error running TUI: {e}")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Rich Issue MCP - Enhanced repo information for AI triaging"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Pull command
    pull_parser = subparsers.add_parser("pull", help="Pull raw issues from GitHub")
    pull_parser.add_argument(
        "repo", help="Repository to analyze (e.g., 'owner/repo')"
    )
    pull_parser.add_argument(
        "--exclude-closed",
        action="store_true",
        help="Exclude closed issues (default: include all)",
    )
    pull_parser.add_argument("--limit", "-l", type=int, help="Limit number of issues")
    pull_parser.add_argument(
        "--start-date",
        default="2025-01-01",
        help="Start date for fetching issues (YYYY-MM-DD)",
    )
    pull_parser.add_argument(
        "--refetch", action="store_true", help="Refetch all issues from start date"
    )
    pull_parser.add_argument(
        "--mode",
        choices=["create", "update"],
        default="update",
        help=(
            "Pull mode: 'create' sorts by created date and avoids re-pulling existing issues, "
            "'update' pulls from last updated date in database (default: update)"
        ),
    )
    pull_parser.add_argument(
        "--issue-numbers",
        nargs="+",
        type=int,
        help="Specific issue numbers to refetch (always refetches even if they exist)",
    )
    pull_parser.add_argument(
        "--item-types",
        choices=["issues", "prs", "both"],
        default="both",
        help="What to fetch: 'issues' only, 'prs' only, or 'both' (default: both)",
    )
    pull_parser.set_defaults(func=cmd_pull)

    # Enrich command
    enrich_parser = subparsers.add_parser(
        "enrich", help="Enrich raw issues with embeddings and metrics"
    )
    enrich_parser.add_argument(
        "repo", help="GitHub repository (e.g., 'jupyterlab/jupyterlab')"
    )
    enrich_parser.add_argument("--model", default="mistral-embed", help="Mistral model")
    enrich_parser.add_argument(
        "--skip-summaries",
        action="store_true",
        help="Skip LLM summarization to save time",
    )
    enrich_parser.set_defaults(func=cmd_enrich)

    # MCP command
    mcp_parser = subparsers.add_parser("mcp", help="Start MCP server")
    mcp_parser.add_argument(
        "repo", help="Repository to serve (e.g., 'owner/repo')"
    )
    mcp_parser.add_argument("--host", default="localhost", help="Host to bind to")
    mcp_parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    mcp_parser.set_defaults(func=cmd_mcp)

    # Visualize command
    visualize_parser = subparsers.add_parser(
        "visualize",
        help="Create T-SNE visualization and GraphML network from enriched issues in TinyDB",
    )
    visualize_parser.add_argument(
        "repo", help="Repository to visualize (e.g., 'owner/repo')"
    )
    visualize_parser.add_argument(
        "--output",
        help="Output file path (.graphml) or directory (default: owner_repo_issues.graphml in current directory)",
    )
    visualize_parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Scale factor for embedding coordinates (default: 1.0)",
    )
    visualize_parser.set_defaults(func=cmd_visualize)

    # Clean command
    clean_parser = subparsers.add_parser("clean", help="Clean downloaded data files")
    clean_parser.add_argument(
        "repo", nargs="?", help="Repository to clean (if not specified, cleans all)"
    )
    clean_parser.add_argument(
        "--yes", "-y", action="store_true", help="Skip confirmation prompt"
    )
    clean_parser.set_defaults(func=cmd_clean)

    # Validate command
    validate_parser = subparsers.add_parser(
        "validate", help="Validate database integrity and completeness"
    )
    validate_parser.add_argument(
        "repo", help="Repository to validate (e.g., 'owner/repo')"
    )
    validate_parser.add_argument(
        "--delete-invalid",
        action="store_true",
        help="Delete invalid entries from database after confirmation",
    )
    validate_parser.set_defaults(func=cmd_validate)

    # TUI command
    tui_parser = subparsers.add_parser(
        "tui", help="Browse database interactively with Terminal UI"
    )
    tui_parser.add_argument(
        "repo", help="Repository to browse (e.g., 'owner/repo')"
    )
    tui_parser.set_defaults(func=cmd_tui)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
