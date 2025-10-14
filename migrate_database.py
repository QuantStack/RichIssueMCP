#!/usr/bin/env python3
"""Migration script to convert TinyDB JSON files to BetterJSONStorage format."""

from pathlib import Path
from tinydb import TinyDB
from rich_issue_mcp.database import FixedBetterJSONStorage, get_data_directory


def migrate_file(json_file: Path):
    """Migrate a JSON file to TinyDB with BetterJSONStorage."""
    print(f"Migrating {json_file.name}...")
    
    # Read data using standard TinyDB
    with TinyDB(json_file, access_mode="r") as old_db:
        issues_list = old_db.all()
    
    print(f"  Read {len(issues_list)} records from original file")
    
    # Create new .db file path
    db_file = json_file.with_suffix('.db')
    print(f"  Creating new database: {db_file.name}")
    
    # Write to new database with BetterJSONStorage using batch insert
    with TinyDB(db_file, access_mode="r+", storage=FixedBetterJSONStorage) as new_db:
        # Clear any existing data and batch insert all issues
        new_db.truncate()
        new_db.insert_multiple(issues_list)
    
    print(f"  ✅ Migration complete: {len(issues_list)} issues written")


def main():
    """Find and migrate all JSON files in the data directory."""
    data_dir = get_data_directory()
    json_files = list(data_dir.glob("*.json"))
    
    if not json_files:
        print("No JSON files found to migrate")
        return
    
    for json_file in json_files:
        migrate_file(json_file)
        print()


if __name__ == "__main__":
    main()