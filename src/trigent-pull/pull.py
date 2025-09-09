#!/usr/bin/env python3
"""Pull GitHub issues and save raw data."""

import json
import gzip
import subprocess
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional


def fetch_issues(repo: str, include_closed: bool = False, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Fetch issues from GitHub using gh CLI."""
    state = "all" if include_closed else "open"
    
    cmd = [
        "gh", "issue", "list",
        "--repo", repo,
        "--state", state,
        "--json", "number,title,body,state,createdAt,updatedAt,author,labels,assignees,url,comments,reactionGroups"
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


def save_raw_issues(issues: List[Dict[str, Any]], filepath: Path) -> None:
    """Save raw issues to gzipped JSON file."""
    filepath.parent.mkdir(exist_ok=True)
    
    with gzip.open(filepath, 'wt', encoding='utf-8') as f:
        json.dump(issues, f, indent=None, separators=(',', ':'))


def main():
    """Main pull entry point."""
    parser = argparse.ArgumentParser(description="Pull GitHub issues and save raw data")
    parser.add_argument("repo", nargs="?", default="jupyterlab/jupyterlab", help="Repository to analyze")
    parser.add_argument("--include-closed", "-c", action="store_true", help="Include closed issues")
    parser.add_argument("--limit", "-l", type=int, help="Limit number of issues")
    parser.add_argument("--output", help="Output file path (default: data/raw-issues-{repo}.json.gz)")
    
    args = parser.parse_args()
    
    print(f"🔍 Fetching issues from {args.repo}...")
    
    raw_issues = fetch_issues(args.repo, args.include_closed, args.limit)
    print(f"📥 Retrieved {len(raw_issues)} issues")
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        repo_name = args.repo.replace('/', '-')
        output_path = Path("data") / f"raw-issues-{repo_name}.json.gz"
    
    save_raw_issues(raw_issues, output_path)
    print(f"✅ Raw issue database saved to {output_path}")


if __name__ == "__main__":
    main()