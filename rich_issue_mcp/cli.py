#!/usr/bin/env python3
"""Main CLI entry point for Rich Issue MCP."""

import argparse
from pathlib import Path

from rich_issue_mcp.config import get_config, get_data_directory
from rich_issue_mcp.database import load_issues, save_issues
from rich_issue_mcp.enrich import (
    add_k4_distances,
    add_quartile_columns,
    add_summaries,
    enrich_issue,
    print_stats,
)
from rich_issue_mcp.mcp_server import run_mcp_server
from rich_issue_mcp.pull import fetch_issues
from rich_issue_mcp.visualize import visualize_issues


def cmd_pull(args) -> None:
    """Execute pull command."""
    print(f"🔍 Fetching issues from {args.repo}...")

    raw_issues = fetch_issues(
        repo=args.repo,
        include_closed=not args.exclude_closed,  # Invert the flag
        limit=args.limit,
        start_date=getattr(args, "start_date", "2025-01-01"),
        chunk_days=getattr(args, "chunk_days", 7),
        include_cross_references=getattr(args, "include_cross_references", True),
        refetch=getattr(args, "refetch", False),
        alignment_date=getattr(args, "alignment_date", None),
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

    save_issues(args.repo, enriched)
    print("✅ Enriched issue database saved to TinyDB")
    print_stats(enriched)


def cmd_mcp(args) -> None:
    """Execute MCP server."""
    run_mcp_server(host=args.host, port=args.port, repo=args.repo)


def cmd_visualize(args) -> None:
    """Execute visualize command."""
    repo = args.repo or "jupyterlab/jupyterlab"
    print(f"📊 Visualizing repository: {repo}")
    
    visualize_issues(repo, args.output, scale=args.scale)


def cmd_clean(args) -> None:
    """Execute clean command to remove downloaded data."""
    data_dir = get_data_directory()

    if not data_dir.exists():
        print("📁 No data directory found")
        return

    # Find data files
    patterns = ["raw-issues-*.json.gz", "enriched-issues-*.json.gz", "state-*.json"]
    files_to_delete = []

    for pattern in patterns:
        files_to_delete.extend(data_dir.glob(pattern))

    if not files_to_delete:
        print("📁 No data files found to clean")
        return

    # Show files that would be deleted
    print("🗑️  Files to be deleted:")
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

    print(f"✅ Cleaned {deleted_count} files from {data_dir}")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Rich Issue MCP - Enhanced repo information for AI triaging"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Pull command
    pull_parser = subparsers.add_parser("pull", help="Pull raw issues from GitHub")
    pull_parser.add_argument(
        "repo", nargs="?", default="jupyterlab/jupyterlab", help="Repository to analyze"
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
        "--no-cross-references",
        dest="include_cross_references",
        action="store_false",
        help="Skip fetching cross-reference data from timeline API",
    )
    pull_parser.add_argument(
        "--chunk-days",
        type=int,
        default=7,
        help="Number of days per chunk for date range processing (default: 7)",
    )
    pull_parser.add_argument(
        "--alignment-date",
        help="Date alignment anchor for cache optimization (YYYY-MM-DD, default from config or 2024-01-01)",
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
    mcp_parser.add_argument("--host", default="localhost", help="Host to bind to")
    mcp_parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    mcp_parser.add_argument(
        "--repo", help="Repository name (defaults to jupyterlab/jupyterlab)"
    )
    mcp_parser.set_defaults(func=cmd_mcp)

    # Visualize command
    visualize_parser = subparsers.add_parser(
        "visualize",
        help="Create T-SNE visualization and GraphML network from enriched issues in TinyDB",
    )
    visualize_parser.add_argument(
        "--repo", 
        help="Repository name (e.g., 'owner/repo') to load from TinyDB (default: jupyterlab/jupyterlab)"
    )
    visualize_parser.add_argument(
        "--output", 
        help="Output file path (.graphml) or directory (default: owner_repo_issues.graphml in current directory)"
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
        "--yes", "-y", action="store_true", help="Skip confirmation prompt"
    )
    clean_parser.set_defaults(func=cmd_clean)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
