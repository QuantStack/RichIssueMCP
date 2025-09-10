# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Trigent is a GitHub issue triaging agent designed to help manage thousands of issues in upstream projects like JupyterLab. The project focuses on organizing, labeling, linking, and summarizing issues using AI-powered analysis.

## Architecture

The system consists of four main Python packages under `trigent/`:

### 1. Data Pulling (trigent/pull/)
- Python package that pulls raw issues from GitHub repositories
- Uses `gh` CLI for GitHub API access
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
ruff check trigent/ && ruff format trigent/ && mypy trigent/
```

## Key Files

- `trigent/cli/cli.py`: Main CLI entry point with all subcommands
- `trigent/pull/pull.py`: Python module for fetching raw issues from GitHub
- `trigent/enrich/enrich.py`: Python enrichment pipeline with embeddings/metrics
- `trigent/mcp/mcp_server.py`: FastMCP server for database access
- `pyproject.toml`: Project configuration

## Dependencies

- **Python 3.11+**: Core language with modern type hints
- **pandas, numpy**: Data processing and quartile calculations
- **requests**: HTTP client for Mistral API
- **FastMCP**: Minimal server for database access
- **subprocess**: Only for calling `gh` CLI in pull module
- **Claude Code**: AI agent for triaging

## Architecture Notes

- **Unified Python**: All components integrated in single Python package with clean module separation
- **Raw Data**: GitHub issues fetched via `gh` CLI subprocess in pull module only
- **Enriched Data**: Pandas-based processing adds embeddings and quartiles (UMAP removed)
- **MCP Server**: FastMCP provides 4 tools for Claude Code agent access
- **CLI Integration**: Single `trigent` command orchestrates entire pipeline with direct Python imports
- **Direct Integration**: No subprocess calls between internal modules - all use direct Python imports