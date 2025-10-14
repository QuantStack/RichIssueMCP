# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RICH ISSUE MCP (Enhanced Repo Information for AI Triaging) is an MCP server that provides enriched GitHub issue data to help AI agents effectively triage thousands of issues in upstream projects like JupyterLab. The system enriches raw issue data with semantic embeddings, metrics computation, and intelligent analysis to enable better AI-powered decision-making.

## Architecture

The system consists of several Python modules under `rich_issue_mcp/`:

### Core Modules
- **rich_issue_mcp/pull.py**: Data pulling module that fetches raw issues from GitHub repositories using intelligent paging
  - Uses `gh` CLI for GitHub API access with weekly chunking based on `updatedAt` timestamps
  - Implements incremental updates to avoid refetching unchanged issues
  - Uses TinyDB for persistent storage and direct issue comparison for updates
  - Merges new/updated issues with existing data while preserving all information
  - Stores data directly in TinyDB database files

- **rich_issue_mcp/enrich.py**: Data enrichment module that processes raw issue data
  - Adds embeddings for semantic search (via Mistral API)
  - Computes metrics: reactions, comments, age, activity scores
  - Assigns quartiles for all metrics using pandas `qcut()` with descriptive labels (Bottom25%, Bottom50%, Top50%, Top25%)
  - Updates TinyDB database with enriched data

- **rich_issue_mcp/mcp_server.py**: FastMCP server providing database access tools
  - Serves enriched issue data to AI agents
  - Tools: get_issue, find_similar_issues, find_linked_issues, get_issue_metrics

- **rich_issue_mcp/cli.py**: CLI orchestration module
  - Unified `rich_issue_mcp` command with subcommands
  - Orchestrates the entire workflow from pull to triaging

### Additional Modules
- **rich_issue_mcp/config.py**: Configuration management and caching
- **rich_issue_mcp/database.py**: Database utilities and operations
- **rich_issue_mcp/validate.py**: Data validation utilities
- **rich_issue_mcp/visualize.py**: Data visualization tools
- **rich_issue_mcp/tui.py**: Text user interface components

## Workflow

```bash
# Install the package
pip install -e .

# 1. Pull raw issue data (intelligent paging from 2025-01-01)
rich_issue_mcp pull jupyterlab/jupyterlab --start-date 2025-01-01

# 1a. Incremental update (only new/updated issues since last fetch)
rich_issue_mcp pull jupyterlab/jupyterlab

# 1b. Force full refetch from start date
rich_issue_mcp pull jupyterlab/jupyterlab --refetch

# 2. Enrich data with embeddings and metrics
rich_issue_mcp enrich jupyterlab/jupyterlab

# 3. Start MCP server for database access
rich_issue_mcp mcp &

# 4. Clean data files when needed
rich_issue_mcp clean
```

## Development Commands

### Setup
```bash
# Install with development dependencies
pip install -e ".[dev]"

# Configure Mistral API key in config.toml
cp config.toml.example config.toml
# Edit config.toml and add your Mistral API key
```

### Code Quality
```bash
# Lint, format, and type check
ruff check rich_issue_mcp/ && ruff format rich_issue_mcp/ && mypy rich_issue_mcp/
```

## Key Files

- `rich_issue_mcp/cli.py`: Main CLI entry point with all subcommands
- `rich_issue_mcp/pull.py`: Python module for fetching raw issues from GitHub
- `rich_issue_mcp/enrich.py`: Python enrichment pipeline with embeddings/metrics
- `rich_issue_mcp/mcp_server.py`: FastMCP server for database access
- `rich_issue_mcp/config.py`: Configuration management and API key handling
- `pyproject.toml`: Project configuration

## Dependencies

- **Python 3.12+**: Core language with modern type hints (updated requirement)
- **pandas, numpy**: Data processing and quartile calculations  
- **requests**: HTTP client for Mistral API
- **FastMCP**: Minimal server for database access
- **scikit-learn**: Machine learning utilities for k-nearest neighbors
- **diskcache**: Persistent caching for API responses
- **toml**: Configuration file parsing
- **ipython, ipdb**: Interactive development and debugging
- **gh CLI**: GitHub issue fetching (external dependency)

## Architecture Notes

- **Unified Python**: All components integrated in single Python package with clean module separation
- **Intelligent Paging**: GitHub issues fetched via `gh` CLI with weekly chunking and incremental updates
- **State Management**: Pull module tracks last fetch timestamps to enable efficient incremental updates
- **Issue Merging**: Smart merge logic updates existing issues while preserving all data integrity  
- **Enriched Data**: Pandas-based processing adds embeddings and quartiles (UMAP removed)
- **MCP Server**: FastMCP provides database access tools for AI agents
- **CLI Integration**: Single `rich_issue_mcp` command orchestrates entire pipeline with direct Python imports
- **Direct Integration**: No subprocess calls between internal modules - all use direct Python imports

## Project Structure

```
RichIssueMCP/
├── rich_issue_mcp/           # Main Python package
│   ├── __init__.py
│   ├── __main__.py          # Entry point for python -m rich_issue_mcp
│   ├── cli.py               # CLI orchestration with all subcommands
│   ├── pull.py              # GitHub issue fetching via gh CLI
│   ├── enrich.py            # Data enrichment with embeddings/metrics
│   ├── database.py          # TinyDB operations and utilities
│   ├── mcp_server.py        # FastMCP server for database access
│   ├── config.py            # Configuration management and caching
│   ├── validate.py          # Data validation utilities
│   ├── visualize.py         # Data visualization tools
│   └── tui.py               # Text user interface components
├── data/                    # Data storage directory
│   └── issues-{repo}.db     # TinyDB database files (e.g., issues-jupyterlab-jupyterlab.db)
├── dcache/                  # Diskcache directory for API response caching
├── example/                 # Example implementations and agents
├── config.toml              # Configuration file (API keys, settings)
├── config.toml.example      # Example configuration template
├── pyproject.toml           # Python project configuration
├── README.md                # Project documentation
├── CLAUDE.md                # Development instructions for Claude Code
└── uv.lock                  # Dependency lock file
```

## Development Notes

### Loading Database for Testing
To test database functionality, load the database the same way as the MCP server:

```python
from rich_issue_mcp.database import load_issues

def _get_repo_name(repo=None):
    """Get repository name, defaulting to jupyterlab/jupyterlab."""
    return repo or "jupyterlab/jupyterlab"

# Load exactly like MCP server  
repo = _get_repo_name()
issues = load_issues(repo)

# Find specific issue
issue_3224 = next((i for i in issues if i["number"] == 3224), None)
```

**Note**: The database must be populated first by running:
1. `rich_issue_mcp pull jupyterlab/jupyterlab --mode create` (to fetch raw issues in create mode)
2. `rich_issue_mcp enrich jupyterlab/jupyterlab` (to add embeddings and metrics)
3. Subsequent updates use: `rich_issue_mcp pull jupyterlab/jupyterlab --mode update`