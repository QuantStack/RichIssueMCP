#!/usr/bin/env python3
"""Database operations using TinyDB for the Rich Issue MCP system."""

from pathlib import Path
from typing import Any

from tinydb import Query, TinyDB
from tinyrecord import transaction

from rich_issue_mcp.config import get_data_directory


def get_database_path(repo: str) -> Path:
    """Get the TinyDB database path for a repository."""
    data_dir = get_data_directory()
    repo_name = repo.replace("/", "-")
    return data_dir / f"issues-{repo_name}.json"


def get_database(repo: str) -> TinyDB:
    """Get or create a TinyDB database for a repository."""
    db_path = get_database_path(repo)
    db_path.parent.mkdir(exist_ok=True)
    return TinyDB(db_path)


def save_issues(repo: str, issues: list[dict[str, Any]]) -> None:
    """Save issues to TinyDB database using TinyRecord for transaction safety."""
    db = get_database(repo)

    with transaction(db) as tr:
        tr.truncate()
        tr.insert_multiple(issues)


def load_issues(repo: str) -> list[dict[str, Any]]:
    """Load all issues from TinyDB database."""
    db = get_database(repo)
    return db.all()


def upsert_issues(repo: str, issues: list[dict[str, Any]]) -> None:
    """Upsert issues to TinyDB database (update existing, insert new)."""
    db = get_database(repo)
    Issue = Query()

    with transaction(db) as tr:
        for issue in issues:
            issue_number = issue["number"]
            tr.upsert(issue, Issue.number == issue_number)
