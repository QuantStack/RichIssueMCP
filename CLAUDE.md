# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Trigent is a GitHub issue triaging agent designed to help manage thousands of issues in upstream projects like JupyterLab. The project focuses on organizing, labeling, linking, and summarizing issues using AI-powered analysis.

## Architecture

The system consists of four main Python packages under `src/`:

### 1. Data Pulling (src/trigent-pull/)
- Python package that pulls raw issues from GitHub repositories
- Uses `gh` CLI for GitHub API access
- Saves raw data as gzipped JSON in /data folder

### 2. Data Enrichment (src/trigent-enrich/)
- Python package that processes raw issue data
- Adds embeddings for semantic search (via Mistral API)
- Computes metrics: reactions, comments, age, activity scores
- Assigns quartiles for all metrics using pandas `qcut()`
- Generates UMAP 2D projections for visualization
- Saves enriched data as gzipped JSON in /data folder

### 3. MCP Server (src/trigent-mcp/)
- FastMCP server providing database access tools
- Serves enriched issue data to AI agents
- Tools: get_issue, find_similar_issues, find_linked_issues, get_issue_metrics

### 4. CLI Orchestration (src/trigent-cli/)
- Python CLI for coordinating all components
- Unified `trigent` command with subcommands
- Orchestrates the entire workflow from pull to triaging

## Workflow

```bash
# Install the package
pip install -e .

# 1. Pull raw issue data
trigent pull jupyterlab/jupyterlab --limit 100

# 2. Enrich data with embeddings and metrics
trigent enrich data/raw-issues-jupyterlab-jupyterlab.json.gz --api-key $MISTRAL_API_KEY

# 3. Start MCP server for database access
trigent mcp &

# 4. Process issues with Claude Code agent  
trigent agent --priority-order --limit 5
```

## Development Commands

### Setup
```bash
pip install -e ".[dev]"
```

### Code Quality
```bash
ruff check src/ && ruff format src/ && mypy src/
```

## Key Files

- `src/trigent-cli/cli.py`: Main CLI entry point with all subcommands
- `src/trigent-pull/pull.py`: Python module for fetching raw issues from GitHub
- `src/trigent-enrich/enrich.py`: Python enrichment pipeline with embeddings/metrics
- `src/trigent-mcp/mcp_server.py`: FastMCP server for database access
- `pyproject.toml`: Project configuration with src layout

## Dependencies

- **Python 3.11+**: Core language with modern type hints
- **pandas, numpy**: Data processing and quartile calculations
- **umap-learn**: Dimensionality reduction for embeddings
- **requests**: HTTP client for Mistral API
- **FastMCP**: Minimal server for database access
- **subprocess**: For calling `gh` CLI
- **Claude Code**: AI agent for triaging

## Architecture Notes

- **Unified Python**: All components now in Python with clean package separation
- **Raw Data**: GitHub issues fetched via Python subprocess calling `gh` CLI
- **Enriched Data**: Pandas-based processing adds embeddings, quartiles, UMAP projections
- **MCP Server**: FastMCP provides 4 tools for Claude Code agent access
- **CLI Integration**: Single `trigent` command orchestrates entire pipeline
- **Src Layout**: Clean package structure under `src/` following Python best practices