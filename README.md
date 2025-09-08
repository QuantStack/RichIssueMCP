# Trigent — AI-Powered GitHub Issue Triaging Agent

Efficiently triage thousands of GitHub issues using semantic analysis and AI decision-making.

## Quick Start

```nu
use trigent

# 1. Create rich issue database with embeddings
trigent pull jupyterlab/jupyterlab --api-key $MISTRAL_API_KEY

# 2. Start MCP server for database access  
trigent mcp &

# 3. Launch Claude Code agent to process issues
trigent agent --priority-order --limit 10
```

## Architecture

### 1. Rich Issue Database (nushell)
Fetches GitHub issues via `gh` CLI and enriches with:
- **Embeddings**: Mistral API embeddings of title + body + comments for semantic similarity
- **Reaction metrics**: Positive/negative reaction counts across issue and comments
- **Engagement heuristics**: Comment frequency, age, priority scores
- **Link detection**: Extract referenced issue numbers (#1234)

Output: Compressed JSON database with all enrichment data.

### 2. Rich Issues MCP Server (Python)
Minimal FastMCP server providing focused tools:
- `get_issue(number)` - Full issue details and metadata
- `find_similar_issues(number)` - Semantic similarity via embeddings  
- `find_linked_issues(number)` - Explicitly referenced issues
- `get_issue_metrics(number)` - Priority/engagement scores

### 3. AI Triaging Agent (Claude Code)
Processes issues individually with MCP context access. For each issue, proposes:
- **"close as done"** - Issue appears resolved
- **"close as duplicate"** - Duplicate of existing issue
- **"keep open"** - Requires continued attention

For "keep open" decisions, evaluates solution paths on:
- **Simplicity vs Impact**: Easy fixes vs. broad solutions
- **Risk assessment**: Likelihood of breaking changes
- **Issue clustering**: Solutions affecting multiple related issues

## Files

- `trigent/mod.nu` - Main module with subcommand exports
- `trigent/pull.nu` - Database creation with enrichment  
- `trigent/agent.nu` - Agent orchestration system
- `trigent/mcp.nu` - MCP server launcher
- `trigent-mcp/mcp_server.py` - MCP server for database access

## Dependencies

- **nushell** - Database creation and orchestration
- **gh CLI** - GitHub issue fetching
- **Mistral API** - Embeddings generation
- **Claude Code** - AI triaging agent
- **FastMCP** - MCP server framework

Designed for large-scale issue management with minimal dependencies and maximum efficiency.
