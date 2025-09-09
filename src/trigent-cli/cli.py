#!/usr/bin/env python3
"""Main CLI entry point for Trigent."""

import argparse
import subprocess
import sys
from pathlib import Path


def cmd_pull(args):
    """Execute pull command."""
    cmd = [sys.executable, "-m", "src.trigent-pull.pull"]
    
    if args.repo:
        cmd.append(args.repo)
    if args.include_closed:
        cmd.append("--include-closed")
    if args.limit:
        cmd.extend(["--limit", str(args.limit)])
    if args.output:
        cmd.extend(["--output", args.output])
    
    subprocess.run(cmd, check=True)


def cmd_enrich(args):
    """Execute enrich command."""
    cmd = [sys.executable, "-m", "src.trigent-enrich.enrich", args.input_file]
    
    if args.api_key:
        cmd.extend(["--api-key", args.api_key])
    if args.model:
        cmd.extend(["--model", args.model])
    if args.output:
        cmd.extend(["--output", args.output])
    if args.skip_umap:
        cmd.append("--skip-umap")
    
    subprocess.run(cmd, check=True)


def cmd_mcp(args):
    """Execute MCP server."""
    cmd = [sys.executable, "-m", "src.trigent-mcp.mcp_server"]
    
    if args.host:
        cmd.extend(["--host", args.host])
    if args.port:
        cmd.extend(["--port", str(args.port)])
    if args.db_file:
        cmd.extend(["--db-file", args.db_file])
    
    subprocess.run(cmd, check=True)


def cmd_agent(args):
    """Execute agent command (placeholder for now)."""
    print("🤖 Agent command not yet implemented")
    print("This would launch Claude Code with MCP server access")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Trigent - GitHub issue triaging agent")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Pull command
    pull_parser = subparsers.add_parser("pull", help="Pull raw issues from GitHub")
    pull_parser.add_argument("repo", nargs="?", default="jupyterlab/jupyterlab", help="Repository to analyze")
    pull_parser.add_argument("--include-closed", "-c", action="store_true", help="Include closed issues")
    pull_parser.add_argument("--limit", "-l", type=int, help="Limit number of issues")
    pull_parser.add_argument("--output", help="Output file path")
    pull_parser.set_defaults(func=cmd_pull)
    
    # Enrich command
    enrich_parser = subparsers.add_parser("enrich", help="Enrich raw issues with embeddings and metrics")
    enrich_parser.add_argument("input_file", help="Path to raw issues JSON.gz file")
    enrich_parser.add_argument("--api-key", help="Mistral API key")
    enrich_parser.add_argument("--model", default="mistral-embed", help="Mistral model")
    enrich_parser.add_argument("--output", help="Output file path")
    enrich_parser.add_argument("--skip-umap", action="store_true", help="Skip UMAP projection")
    enrich_parser.set_defaults(func=cmd_enrich)
    
    # MCP command
    mcp_parser = subparsers.add_parser("mcp", help="Start MCP server")
    mcp_parser.add_argument("--host", default="localhost", help="Host to bind to")
    mcp_parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    mcp_parser.add_argument("--db-file", help="Database file path")
    mcp_parser.set_defaults(func=cmd_mcp)
    
    # Agent command
    agent_parser = subparsers.add_parser("agent", help="Run triaging agent")
    agent_parser.add_argument("--priority-order", action="store_true", help="Process by priority")
    agent_parser.add_argument("--limit", type=int, help="Limit number of issues to process")
    agent_parser.set_defaults(func=cmd_agent)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    args.func(args)


if __name__ == "__main__":
    main()