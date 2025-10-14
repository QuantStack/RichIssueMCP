#!/usr/bin/env python3
"""Database operations using TinyDB for the Rich Issue MCP system."""

from pathlib import Path
from typing import Any

from tinydb import Query, TinyDB
from tinyrecord import transaction
from BetterJSONStorage import BetterJSONStorage
from blosc2 import compress
from orjson import dumps


class FixedBetterJSONStorage(BetterJSONStorage):
    """BetterJSONStorage with fix for blosc2 typesize issue."""
    
    def _BetterJSONStorage__file_writer(self):
        """Fixed file writer that uses typesize=None to avoid the compression error."""
        self._shutdown_lock.acquire()
        while self._running:
            if self._changed:
                self._changed = False
                self._handle.seek(0)
                self._handle.truncate()
                # Fix: Use typesize=None to avoid the "multiple of typesize (8)" error
                self._handle.write(compress(dumps(self._data, **self._dump_kwargs), typesize=None))
                self._handle.flush()
        self._shutdown_lock.release()

from rich_issue_mcp.config import get_data_directory

# Global cache for database instances to prevent BetterJSONStorage conflicts
_database_cache: dict[str, TinyDB] = {}


def get_database_path(repo: str) -> Path:
    """Get the TinyDB database path for a repository."""
    data_dir = get_data_directory()
    repo_name = repo.replace("/", "-")
    return data_dir / f"issues-{repo_name}.db"


def get_database(repo: str) -> TinyDB:
    """Get or create a TinyDB database for a repository."""
    # Use cached instance if available
    if repo in _database_cache:
        return _database_cache[repo]
    
    db_path = get_database_path(repo)
    db_path.parent.mkdir(exist_ok=True)
    db = TinyDB(db_path, access_mode="r+", storage=FixedBetterJSONStorage)
    
    # Cache the database instance
    _database_cache[repo] = db
    return db


def save_issues(repo: str, issues: list[dict[str, Any]]) -> None:
    """Save issues to TinyDB database using TinyRecord for transaction safety."""
    # Validate repo parameter to prevent incorrect file naming
    if ".db" in repo or ".json" in repo or ".gz" in repo or "issues-" in repo or "enriched-" in repo:
        raise ValueError(
            f"Invalid repo name '{repo}': should be 'owner/repo', not a file path"
        )

    db = get_database(repo)

    # Clear existing data first
    db.truncate()

    # Insert all new issues in a transaction
    with transaction(db) as tr:
        tr.insert_multiple(issues)


def load_issues(repo: str) -> list[dict[str, Any]]:
    """Load all issues from TinyDB database."""
    # Validate repo parameter to prevent incorrect file naming
    if ".db" in repo or ".json" in repo or ".gz" in repo or "issues-" in repo or "enriched-" in repo:
        raise ValueError(
            f"Invalid repo name '{repo}': should be 'owner/repo', not a file path"
        )

    db = get_database(repo)
    return db.all()


def upsert_issues(repo: str, issues: list[dict[str, Any]]) -> None:
    """Upsert issues to TinyDB database (update existing, insert new)."""
    # Validate repo parameter to prevent incorrect file naming
    if ".db" in repo or ".json" in repo or ".gz" in repo or "issues-" in repo or "enriched-" in repo:
        raise ValueError(
            f"Invalid repo name '{repo}': should be 'owner/repo', not a file path"
        )

    db = get_database(repo)
    Issue = Query()

    with transaction(db) as tr:
        for issue in issues:
            issue_number = issue["number"]
            # Check if issue exists
            existing = db.search(Issue.number == issue_number)
            if existing:
                # Update existing issue
                tr.update(issue, Issue.number == issue_number)
            else:
                # Insert new issue
                tr.insert(issue)
