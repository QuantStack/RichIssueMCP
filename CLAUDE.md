# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Trigent is a GitHub issue triaging agent designed to help manage thousands of issues in upstream projects like JupyterLab. The project focuses on organizing, labeling, linking, and summarizing issues using AI-powered analysis with semantic search, metrics computation, and intelligent data management.

## Architecture

The system consists of four main Python packages under `trigent/`:

### 1. Data Pulling (trigent/pull/)
- Python package that pulls raw issues from GitHub repositories using intelligent paging
- Uses `gh` CLI for GitHub API access with weekly chunking based on `updatedAt` timestamps
- Implements incremental updates to avoid refetching unchanged issues
- Maintains state tracking in `data/state-{repo}.json` for last fetch timestamps
- Merges new/updated issues with existing data while preserving all information
- Saves raw data as gzipped JSON in /data folder

### 2. Data Enrichment (trigent/enrich/)
- Python package that processes raw issue data
- Adds embeddings for semantic search (via Mistral API)
- Computes metrics: reactions, comments, age, activity scores
- Assigns quartiles for all metrics using pandas `qcut()`
- Saves enriched data as gzipped JSON in /data folder

### 3. MCP Server (trigent/mcp/)
- FastMCP server providing database access tools
- Serves enriched issue data to AI agents
- Tools: get_issue, find_similar_issues, find_linked_issues, get_issue_metrics

### 4. CLI Orchestration (trigent/cli/)
- Python CLI for coordinating all components
- Unified `trigent` command with subcommands
- Orchestrates the entire workflow from pull to triaging

## Workflow

```bash
# Install the package
pip install -e .

# 1. Pull raw issue data (intelligent paging from 2025-01-01)
trigent pull jupyterlab/jupyterlab --start-date 2025-01-01

# 1a. Incremental update (only new/updated issues since last fetch)
trigent pull jupyterlab/jupyterlab

# 1b. Force full refetch from start date
trigent pull jupyterlab/jupyterlab --refetch

# 2. Enrich data with embeddings and metrics
trigent enrich data/raw-issues-jupyterlab-jupyterlab.json.gz

# 3. Start MCP server for database access
trigent mcp &

# 4. Clean data files when needed
trigent clean
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
ruff check trigent/ && ruff format trigent/ && mypy trigent/
```

## Key Files

- `trigent/cli/cli.py`: Main CLI entry point with all subcommands
- `trigent/pull/pull.py`: Python module for fetching raw issues from GitHub
- `trigent/enrich/enrich.py`: Python enrichment pipeline with embeddings/metrics
- `trigent/mcp/mcp_server.py`: FastMCP server for database access
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
- **CLI Integration**: Single `trigent` command orchestrates entire pipeline with direct Python imports
- **Direct Integration**: No subprocess calls between internal modules - all use direct Python imports