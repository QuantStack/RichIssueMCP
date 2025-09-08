# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Trigent is a GitHub issue triaging agent designed to help manage thousands of issues in upstream projects like JupyterLab. The project focuses on organizing, labeling, linking, and summarizing issues using AI-powered analysis.

## Architecture

The system consists of two main components:

### Rich Issues MCP Server
- Pulls all open issues and PRs from configured upstream repositories (e.g., JupyterLab/JupyterLab)
- Enriches issue data with:
  - Embeddings for semantic search and similarity detection (concatenated title + body + comments)
  - Heuristics for prioritization (reactions, comments, similar issues)
  - LLM-based judgments on frequency, severity, and impact
  - Linked issue analysis

### Triaging Agent
- Works on issues individually or in clustered groups
- Has access to:
  - Rich Issues MCP data
  - Shallow clone of the target codebase
  - Existing PRs (linked or similar via embeddings)
- Proposes actions: "close as done", "close as duplicate", or "keep open"
- For open issues, evaluates solutions on axes of simplicity and impact

## Workflow

```bash
use trigent

# 1. Create enriched issue database with embeddings
trigent pull jupyterlab/jupyterlab --api-key $MISTRAL_API_KEY --limit 100

# 2. Start server for database access
trigent mcp &

# 3. Process issues with Claude Code agent  
trigent agent --priority-order --limit 5
```

## Development Commands

### Setup
```bash
pip install -e ".[dev]"
```

### Code Quality
```bash
ruff check trigent-mcp/ && ruff format trigent-mcp/ && mypy trigent-mcp/
```

## Key Files

- `trigent/mod.nu`: Main module with subcommand exports
- `trigent/pull.nu`: Creates enriched issue database with embeddings/heuristics
- `trigent/agent.nu`: Launches Claude Code agent on issues from database
- `trigent/mcp.nu`: MCP server launcher wrapper
- `trigent-mcp/mcp_server.py`: Minimal FastMCP server for database access

## Dependencies

- **nushell**: Database creation and agent orchestration
- **FastMCP**: Minimal server for database access
- **Mistral API**: Embeddings (via nushell http)
- **gh CLI**: GitHub issue fetching
- **Claude Code**: AI agent for triaging

## Architecture Notes

- **Rich Issue DB**: Created in nushell, stored as compressed JSON
- **MCP Server**: Python server provides 4 tools: get_issue, find_similar_issues, find_linked_issues, get_issue_metrics
- **Agent**: Claude Code processes issues one by one with MCP access
- Compact design: no unnecessary abstractions or heavy dependencies