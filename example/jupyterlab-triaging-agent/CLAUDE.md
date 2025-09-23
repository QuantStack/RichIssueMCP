# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Building and Development
- `npm run build` - Build development version (alias for `build:dev`)
- `npm run build:dev` - Full development build (integrity check + build:all + dev_mode build)
- `npm run build:dev:prod` - Production build for development
- `npm run build:core` - Build core JupyterLab staging package
- `npm run watch` - Watch mode for development (runs `scripts/watch_dev.py`)
- `npm run clean` - Clean development and packages
- `npm run clean:slate` - Complete clean and reinstall from scratch

### Testing
- `npm run test` - Run tests for all packages except galata and template
- `npm run test:all` - Run all tests with no bail on failures
- `npm run test:galata` - Run Playwright end-to-end tests
- `npm run test:examples` - Test example applications

### Code Quality
- `npm run lint` - Run prettier, eslint, and stylelint
- `npm run lint:check` - Check code quality without fixing
- `npm run prettier` - Format all TypeScript, JavaScript, CSS, JSON, and Markdown files
- `npm run eslint` - Run ESLint with TypeScript support and fix issues
- `npm run stylelint` - Run stylelint for CSS files

### Package Management
- `npm run integrity` - Ensure repository consistency and buildutils
- `npm run deduplicate` - Deduplicate yarn dependencies
- `npm run build:packages` - Build specific packages using Lerna

## Architecture Overview

JupyterLab is a monorepo with 102 packages organized into core libraries and extensions.

### Core Architecture Layers

**Application Layer**
- `@jupyterlab/application` - Main application shell and plugin system with JupyterFrontEnd
- `@jupyterlab/application-extension` - Core application plugins

**Services Layer**  
- `@jupyterlab/services` - Backend connectivity to Jupyter Server REST APIs
- `@jupyterlab/apputils` - Application utilities (command palette, dialogs, theming)
- `@jupyterlab/ui-components` - React-based UI component library

**Document System**
- `@jupyterlab/docregistry` - Document type registry and factories
- `@jupyterlab/notebook` - Notebook interface components
- `@jupyterlab/cells` - Individual cell widgets (code, markdown, raw)
- `@jupyterlab/rendermime` - Output rendering for MIME types

**Editor System**
- `@jupyterlab/codeeditor` - Abstract code editor interfaces  
- `@jupyterlab/codemirror` - CodeMirror 6 integration
- `@jupyterlab/completer` - Code completion system

**Foundation**
- `@jupyterlab/coreutils` - Essential utilities (paths, URLs, activities)
- Lumino packages - Core widget system, commands, signaling

### Key Patterns

**Plugin Architecture**: Extensions export `JupyterFrontEndPlugin[]` objects with token-based dependency injection. Each core package typically has a corresponding `-extension` package.

**Monorepo Structure**: Yarn workspaces with Lerna orchestration. All packages share the same version. Standard layout: `src/`, `lib/` (compiled), `style/`, `test/`.

**Technology Stack**: TypeScript throughout, React 18+ for modern UI components, Lumino widgets for core framework, CodeMirror 6 for editing.

## Working with Packages

### Package Locations
- Core packages: `packages/` directory
- Example applications: `examples/` directory  
- Build utilities: `buildutils/`
- Test utilities: `testutils/`
- End-to-end tests: `galata/`

### Building Individual Packages
- `lerna run build --scope "@jupyterlab/package-name"` - Build specific package
- `lerna run build --concurrency 1` - Build all packages sequentially
- `npm run build:src` - Build all core packages (excludes test/example packages)

### Testing Individual Packages
- `lerna run test --scope "@jupyterlab/package-name"` - Test specific package
- Tests use Jest framework with `@jupyterlab/testing` utilities

## Python Components

The repository also contains Python components in:
- `jupyterlab/` - Main Python package for JupyterLab server extension
- `buildapi.py` - Python build API
- Various test and configuration files

Python development uses standard pip/conda installation and pytest for testing.

## Issue Triage Workflow

When triaging GitHub issues follow this systematic approach:

### 1. Issue Selection and Data Gathering

Use Rich Issue MCP tools for efficient issue management:

**Get Next Issue:**
- `mcp__rich_issue_mcp__get_first_issue_without_recommendation` - Find next untriaged issue

**Gather Comprehensive Data:**
- `mcp__rich_issue_mcp__get_issue` - Full conversation and summary
- `mcp__rich_issue_mcp__get_issue_metrics` - Engagement metrics and quartile rankings
- `mcp__rich_issue_mcp__find_similar_issues` - Related issues for context
- `mcp__rich_issue_mcp__find_linked_issues` - Direct issue references

**Additional Tools:**
- `mcp__rich_issue_mcp__get_available_sort_columns` - List sortable columns
- `mcp__rich_issue_mcp__get_top_issues` - Get top N issues by metric
- `mcp__rich_issue_mcp__export_all_open_issues` - Export issue data to JSON

Use the Deepwiki mcp if available to ask about this issue

### 2. Assessment Framework

**Severity** (how strongly are users affected?):
- **High**: Breaks core functionality, blocks workflows and no or only difficult workarounds only.
- **Medium**: Impacts user experience but usable workarounds within Jupyter exist.
- **Low**: Minor inconvenience or edge case.

**Frequency** (how often is this encountered by affected users?):
- **High**: Daily
- **Medium**: Weekly/monthly
- **Low**: Rare or specific use cases only

**Fraction** (what portion of user base is affected?):
- **High**: Most JupyterLab users encounter this
- **Medium**: Significant subset (intermediate/advanced users)
- **Low**: Small subset or specific user groups

### 3. Technical Assessment

Evaluate using JupyterLab architecture knowledge:
- Is the issue still relevant? (May have been fixed in recent versions)
- Which packages would require modification?
- Solution complexity and potential side effects/risks
- Alignment with existing patterns and architectural decisions

Use the deepwiki mcp as well to answer these (only ask a single question).

### 4. Final Recommendation

- **prioritize**: High-impact, well-defined issues with clear user benefit
- **neutral**: Legitimate improvements that can be addressed when resources permit
- **deprioritize**: Low-impact or already partially addressed through alternatives
- **close**: No longer relevant, duplicate, or out of scope

### 5. Update Issue Database

Use Rich Issue MCP to add assessment recommendations:

```
mcp__rich_issue_mcp__add_recommendation
```

Required parameters:
- `issue_number` - GitHub issue number
- `severity` - Assessment: "low", "medium", "high"
- `frequency` - Assessment: "low", "medium", "high"  
- `prevalence` - Assessment: "low", "medium", "high"
- `recommendation` - Final decision: "close", "deprioritize", "keep", "prioritize"
- `report` - Comprehensive analysis text explaining the assessment
- `solution_complexity` - Implementation effort: "low", "medium", "high"
- `solution_risk` - Implementation risk: "low", "medium", "high"

This workflow ensures consistent, thorough evaluation of each issue while leveraging both quantitative metrics and qualitative JupyterLab expertise.
