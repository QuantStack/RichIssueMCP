"""FastMCP server for accessing Rich Issues database."""

import gzip
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP


mcp = FastMCP("Rich Issues Server")
_issues_cache: Dict[str, List[Dict]] = {}


def load_issues_db(db_file: str) -> List[Dict]:
    """Load and cache issues from compressed JSON database."""
    if db_file in _issues_cache:
        return _issues_cache[db_file]
    
    db_path = Path(db_file)
    if not db_path.exists():
        raise FileNotFoundError(f"Database file not found: {db_file}")
    
    try:
        with gzip.open(db_path, 'rt') as f:
            issues = json.load(f)
        _issues_cache[db_file] = issues
        return issues
    except Exception as e:
        raise RuntimeError(f"Failed to load database: {e}") from e


@mcp.tool()
def get_issue(issue_number: int, db_file: str = "rich-issues-jupyterlab-jupyterlab.json.gz") -> Optional[Dict[str, Any]]:
    """Get specific issue with all metadata and enrichment data."""
    issues = load_issues_db(db_file)
    return next((i for i in issues if i["number"] == issue_number), None)


@mcp.tool()
def find_similar_issues(
    issue_number: int, 
    threshold: float = 0.8, 
    limit: int = 10,
    db_file: str = "rich-issues-jupyterlab-jupyterlab.json.gz"
) -> List[Dict[str, Any]]:
    """Find issues similar to target issue using embeddings."""
    issues = load_issues_db(db_file)
    target = next((i for i in issues if i["number"] == issue_number), None)
    
    if not target or not target.get("embedding"):
        return []
    
    def cosine_similarity(a: List[float], b: List[float]) -> float:
        dot_product = sum(x * y for x, y in zip(a, b))
        magnitude_a = sum(x * x for x in a) ** 0.5
        magnitude_b = sum(x * x for x in b) ** 0.5
        return dot_product / (magnitude_a * magnitude_b) if magnitude_a and magnitude_b else 0
    
    similar = []
    for issue in issues:
        if issue["number"] == issue_number or not issue.get("embedding"):
            continue
        
        similarity = cosine_similarity(target["embedding"], issue["embedding"])
        if similarity >= threshold:
            result = issue.copy()
            result["similarity"] = similarity
            similar.append(result)
    
    return sorted(similar, key=lambda x: x["similarity"], reverse=True)[:limit]


@mcp.tool()
def find_linked_issues(issue_number: int, db_file: str = "rich-issues-jupyterlab-jupyterlab.json.gz") -> List[Dict[str, Any]]:
    """Find issues referenced in the body/comments of target issue."""
    issues = load_issues_db(db_file)
    target = next((i for i in issues if i["number"] == issue_number), None)
    
    if not target:
        return []
    
    # Extract issue numbers from text (e.g., #1234, #5678)
    text = f"{target.get('title', '')} {target.get('body', '')}"
    for comment in target.get("comments", []):
        text += f" {comment.get('body', '')}"
    
    linked_numbers = set(int(m.group(1)) for m in re.finditer(r'#(\d+)', text))
    linked_numbers.discard(issue_number)  # Remove self-reference
    
    return [i for i in issues if i["number"] in linked_numbers]


@mcp.tool()
def get_issue_metrics(issue_number: int, db_file: str = "rich-issues-jupyterlab-jupyterlab.json.gz") -> Optional[Dict[str, Any]]:
    """Get enrichment metrics for a specific issue."""
    issues = load_issues_db(db_file)
    issue = next((i for i in issues if i["number"] == issue_number), None)
    
    if not issue:
        return None
    
    return {
        "priority_score": issue.get("priority_score", 0),
        "frequency_score": issue.get("frequency_score", 0),
        "severity_score": issue.get("severity_score", 0),
        "comment_count": issue.get("comment_count", 0),
        "total_reactions": issue.get("total_reactions", 0),
        "positive_reactions": issue.get("positive_reactions", 0),
        "negative_reactions": issue.get("negative_reactions", 0),
        "age_days": issue.get("age_days", 0),
        "has_embedding": issue.get("embedding") is not None,
    }


if __name__ == "__main__":
    mcp.run()