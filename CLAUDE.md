# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RICH ISSUE MCP (Enhanced Repo Information for AI Triaging) is an MCP server that provides enriched GitHub issue data to help AI agents effectively triage thousands of issues in upstream projects like JupyterLab. The system enriches raw issue data with semantic embeddings, metrics computation, and intelligent analysis to enable better AI-powered decision-making.

## Architecture

The system consists of four main Python packages under `rich_issue_mcp/`:

### 1. Data Pulling (rich_issue_mcp/pull/)
- Python package that pulls raw issues from GitHub repositories using intelligent paging
- Uses `gh` CLI for GitHub API access with weekly chunking based on `updatedAt` timestamps
- Implements incremental updates to avoid refetching unchanged issues
- Maintains state tracking in `data/state-{repo}.json` for last fetch timestamps
- Merges new/updated issues with existing data while preserving all information
- Saves raw data as gzipped JSON in /data folder

### 2. Data Enrichment (rich_issue_mcp/enrich/)
- Python package that processes raw issue data
- Adds embeddings for semantic search (via Mistral API)
- Computes metrics: reactions, comments, age, activity scores
- Assigns quartiles for all metrics using pandas `qcut()`
- Saves enriched data as gzipped JSON in /data folder

### 3. MCP Server (rich_issue_mcp/mcp/)
- FastMCP server providing database access tools
- Serves enriched issue data to AI agents
- Tools: get_issue, find_similar_issues, find_linked_issues, get_issue_metrics

### 4. CLI Orchestration (rich_issue_mcp/cli/)
- Python CLI for coordinating all components
- Unified `rich_issue_mcp` command with subcommands
- Orchestrates the entire workflow from pull to triaging

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
rich_issue_mcp enrich data/raw-issues-jupyterlab-jupyterlab.json.gz

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

- `rich_issue_mcp/cli/cli.py`: Main CLI entry point with all subcommands
- `rich_issue_mcp/pull/pull.py`: Python module for fetching raw issues from GitHub
- `rich_issue_mcp/enrich/enrich.py`: Python enrichment pipeline with embeddings/metrics
- `rich_issue_mcp/mcp/mcp_server.py`: FastMCP server for database access
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