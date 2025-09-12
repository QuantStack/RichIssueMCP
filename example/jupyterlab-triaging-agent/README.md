# JupyterLab Issue Triaging Agent

## Quickstart

First, follow instructions to create the issue db with rich issue mcp.

Then in another clean folder do `gh repo clone "jupyterlab/jupyterlab" -- --depth=1`.

Then copy the CLAUDE.md from this folder into the jupyterlab folder.

Then start `claude`

Prompt it with:

```
Ask rich_issue_mcp what metrics it provides. Then ask it to return the top 10 issues by emoji count. Select the top issue and start triaging.
```
