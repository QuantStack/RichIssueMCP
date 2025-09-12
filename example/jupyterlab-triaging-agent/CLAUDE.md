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