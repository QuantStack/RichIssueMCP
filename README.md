# Trigent — A simple Triaging Agent for Github Issues

This agent was born out of the necessity to triage thousands of issues in the
upstream Jupyterlab project.

## Objectives

- organize issues, primarily linking, labeling.
- summarize issues, group similar themes, provide various statistics

## Technologies

- gh cli to pull data from GitHub
- structured data processing: for now Python or Nushell
- different llm endpoints for embeddings and simple text analysis
- claude code for in depth analysis

## Prioritization

We prioritize issues that are *severe*, encountered *often* by *many*. We will
try to characterize these three dimensions based on different heuristics.

In a second step we plan to judge whether issues are actionable. Including
whether a change can affect many issues.

## Challenges

- while some user problems may be concentrated in a single issue, others may
  be spread over many similar issues groups. We somehow need to find a way
  to detect such groups of similar issues and weight them in addition to single
  important issues.
