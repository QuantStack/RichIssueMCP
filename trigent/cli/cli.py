#!/usr/bin/env python3
"""Main CLI entry point for Trigent."""

import argparse
import glob
import os
from pathlib import Path

from trigent.config import get_config
from trigent.enrich.enrich import (
    add_quartile_columns,
    add_umap_projection,
    enrich_issue,
    load_raw_issues,
    print_stats,
    save_enriched_issues,
)
from trigent.mcp.mcp_server import run_mcp_server
from trigent.pull.pull import fetch_issues, save_raw_issues


def cmd_pull(args):
    """Execute pull command."""
    print(f"🔍 Fetching issues from {args.repo}...")

    raw_issues = fetch_issues(
        repo=args.repo,
        include_closed=args.include_closed,
        limit=args.limit,
        start_date=getattr(args, "start_date", "2025-01-01"),
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


def cmd_enrich(args):
    """Execute enrich command."""
    input_path = Path(args.input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        basename = input_path.name.replace("raw-", "enriched-")
        output_path = Path("data") / basename

    output_path.parent.mkdir(exist_ok=True)

    print(f"🔍 Loading raw issues from {input_path}...")
    raw_issues = load_raw_issues(input_path)
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

    # Add UMAP projection if requested
    if not args.skip_umap:
        enriched = add_umap_projection(enriched)

    save_enriched_issues(enriched, output_path)
    print(f"✅ Enriched issue database saved to {output_path}")
    print_stats(enriched)


def cmd_mcp(args):
    """Execute MCP server."""
    run_mcp_server(host=args.host, port=args.port, db_file=args.db_file)


def cmd_agent(args):
    """Execute agent command (placeholder for now)."""
    print("🤖 Agent command not yet implemented")
    print("This would launch Claude Code with MCP server access")


def cmd_clean(args):
    """Execute clean command to remove downloaded data."""
    data_dir = Path("data")
    
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
        if response not in ('y', 'yes'):
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


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Trigent - GitHub issue triaging agent"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Pull command
    pull_parser = subparsers.add_parser("pull", help="Pull raw issues from GitHub")
    pull_parser.add_argument(
        "repo", nargs="?", default="jupyterlab/jupyterlab", help="Repository to analyze"
    )
    pull_parser.add_argument(
        "--include-closed", "-c", action="store_true", help="Include closed issues"
    )
    pull_parser.add_argument("--limit", "-l", type=int, help="Limit number of issues")
    pull_parser.add_argument("--output", help="Output file path")
    pull_parser.add_argument(
        "--start-date",
        default="2025-01-01",
        help="Start date for fetching issues (YYYY-MM-DD)",
    )
    pull_parser.add_argument(
        "--refetch", action="store_true", help="Refetch all issues from start date"
    )
    pull_parser.set_defaults(func=cmd_pull)

    # Enrich command
    enrich_parser = subparsers.add_parser(
        "enrich", help="Enrich raw issues with embeddings and metrics"
    )
    enrich_parser.add_argument("input_file", help="Path to raw issues JSON.gz file")
    enrich_parser.add_argument("--model", default="mistral-embed", help="Mistral model")
    enrich_parser.add_argument("--output", help="Output file path")
    enrich_parser.add_argument(
        "--skip-umap", action="store_true", help="Skip UMAP projection"
    )
    enrich_parser.set_defaults(func=cmd_enrich)

    # MCP command
    mcp_parser = subparsers.add_parser("mcp", help="Start MCP server")
    mcp_parser.add_argument("--host", default="localhost", help="Host to bind to")
    mcp_parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    mcp_parser.add_argument("--db-file", help="Database file path")
    mcp_parser.set_defaults(func=cmd_mcp)

    # Agent command
    agent_parser = subparsers.add_parser("agent", help="Run triaging agent")
    agent_parser.add_argument(
        "--priority-order", action="store_true", help="Process by priority"
    )
    agent_parser.add_argument(
        "--limit", type=int, help="Limit number of issues to process"
    )
    agent_parser.set_defaults(func=cmd_agent)

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
