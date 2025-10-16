#!/usr/bin/env python3
"""Database operations using TinyDB for the Rich Issue MCP system."""

from pathlib import Path
from typing import Any

from BetterJSONStorage import BetterJSONStorage
from blosc2 import compress
from orjson import dumps
from tinydb import Query, TinyDB
from tinyrecord import transaction


class FixedBetterJSONStorage(BetterJSONStorage):
    """BetterJSONStorage with fix for blosc2 typesize issue and backup protection."""

    def _BetterJSONStorage__file_writer(self):
        """Fixed file writer that uses typesize=None and creates backup before truncate."""
        self._shutdown_lock.acquire()
        while self._running:
            if self._changed:
                self._changed = False

                # Create backup before truncating
                import shutil
                backup_path = str(self._handle.name) + '.backup'
                try:
                    if hasattr(self._handle, 'name') and Path(self._handle.name).exists():
                        if Path(self._handle.name).stat().st_size > 0:
                            shutil.copy2(self._handle.name, backup_path)
                except (OSError, AttributeError):
                    pass  # Continue without backup if copy fails

                try:
                    self._handle.seek(0)
                    self._handle.truncate()
                    # Fix: Use typesize=None to avoid the "multiple of typesize (8)" error
                    self._handle.write(compress(dumps(self._data, **self._dump_kwargs), typesize=None))
                    self._handle.flush()

                    # Remove backup on successful write
                    try:
                        if Path(backup_path).exists():
                            Path(backup_path).unlink()
                    except OSError:
                        pass

                except Exception as e:
                    # Restore from backup on write failure
                    try:
                        if Path(backup_path).exists():
                            shutil.copy2(backup_path, self._handle.name)
                            Path(backup_path).unlink()
                    except (OSError, AttributeError):
                        pass
                    # Re-raise the original exception
                    raise e

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


def convert_numpy_types(obj: Any) -> Any:
    """Recursively convert NumPy types to Python native types for JSON serialization."""
    import numpy as np
    
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


def save_issues(repo: str, issues: list[dict[str, Any]]) -> None:
    """Save issues to TinyDB database using TinyRecord for transaction safety."""
    # Validate repo parameter to prevent incorrect file naming
    if ".db" in repo or ".json" in repo or ".gz" in repo or "issues-" in repo or "enriched-" in repo:
        raise ValueError(
            f"Invalid repo name '{repo}': should be 'owner/repo', not a file path"
        )

    # Convert NumPy types to Python native types for JSON serialization
    serializable_issues = [convert_numpy_types(issue) for issue in issues]

    db = get_database(repo)

    # Clear existing data first
    db.truncate()

    # Insert all new issues in a transaction
    with transaction(db) as tr:
        tr.insert_multiple(serializable_issues)
    
    # Force close to flush BetterJSONStorage background writer
    db.close()
    
    # Remove from cache since we closed it
    if repo in _database_cache:
        del _database_cache[repo]


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
