# Trigent — A simple Triaging Agent for Github Issues

This agent was born out of the necessity to triage thousands of issues in the
upstream Jupyterlab project.

## Objectives

- organize issues, primarily linking, labeling.
- summarize issues, group similar themes, provide various statistics.

## Architecture

The agent works as follows:

### Rich Issues MCP

First we spin up the "Rich Issues" MCP server. When this server starts,
it pulls all open (configurable as all including closed) Issues and PRs from
the upstream repo (for example configured as Jupyterlab/Jupyterlab). It then
enriches the data with things like

- Embeddings (on concatenated title + body + comments) for each issue. This is
  for semantic search and the detection of similar issues.
- heuristics:
  - number of total, affirmative and negative reactionGroups (=emojis) on the issue body and all comments individually. ⇒  heuristic for frequency × severity × fraction.
  - count number of comments. ⇒  heuristic for frequency × severity × fraction.
  - count number of very similar issues ⇒  heuristic for frequency × severity × fraction
  - let cheap LLM judge on frequency, severity, fraction based on simple prompt
    and text analysis.
  - count number of linked issues.

We run this upfront to avoid reembedding issues over and over again, and to be
able to do upfront prioritization before unleashing the agent that is way more
costly from top to bottom in order of priority.

### Agent

The agent afterwards works on *issue by issue*. Another working mode in a later
version could be to work on clustered issue groups to try resolving several
together and we don't want to bake in the issue-by-issue mode to tightly.

For example, the agent could start working from the issues with the highest
priority *heuristic* downwards. When opening an issue, the agent should propose
actions and argue why based on all the context, including
- what the "Rich Issues" MCP provides,
- including access to a shallow clone of the actual jupyterlab codebase,
- including existing PRs that are linked to the issue or detected as similar
  through the embeddings.

Proposed actions on open issues are "close as done", "close as duplicate", or
"keep open". When keeping open, the agent should explore solution paths based
on access to the codebase. It should judge these solution paths based on the
axes: simplicity and impact. For example, a solution that is easy to implement
but only resolves one low priority issue is easy but low impact. A solution
that is hard to implement but affects many issues at once is difficult with
high impact. The agent should also judge whether the solution is likely to
break other things.
