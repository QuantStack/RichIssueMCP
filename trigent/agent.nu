# Launch Claude Code agent on issues

export def main [
    db_file: string = "rich-issues-jupyterlab-jupyterlab.json.gz"  # Rich issues database
    --start-from(-s): int = 1                                      # Start from issue number
    --limit(-l): int = 10                                          # Number of issues to process
    --priority-order(-p)                                           # Process by priority score
] {
    print $"🤖 Launching triaging agent on ($db_file)..."
    
    let issues = load_and_sort_issues $db_file $priority_order
    let filtered = ($issues | skip ($start_from - 1) | first $limit)
    
    print $"📋 Processing ($filtered | length) issues"
    if $priority_order {
        print "🔀 Ordered by priority score (highest first)"
    }
    
    ensure_mcp_server
    
    $filtered | each {|issue| 
        process_issue $issue.number $db_file
    }
    
    print "✅ Agent processing complete"
}

def load_and_sort_issues [db_file: string, priority_order: bool] {
    let issues = (gzip -d < $db_file | from json)
    
    if $priority_order {
        $issues | sort-by priority_score --reverse
    } else {
        $issues | sort-by number
    }
}

def process_issue [issue_number: int, db_file: string] {
    print $"\n🔍 Processing Issue #($issue_number)"
    print "=" * 50
    
    let agent_prompt = build_agent_prompt $issue_number
    
    print $"🧠 Launching Claude Code agent for issue #($issue_number)..."
    
    run-external "claude" $agent_prompt
}

def build_agent_prompt [issue_number: int] {
    $"You are a GitHub issue triaging agent with access to rich issue data via MCP.

Your task: Analyze issue #($issue_number) and propose one of these actions:
1. 'close as done' - Issue appears to be resolved
2. 'close as duplicate' - Issue is duplicate of another  
3. 'keep open' - Issue should remain open

You have access to these MCP tools:
- get_issue($issue_number) - Get full issue details
- find_similar_issues($issue_number) - Find semantically similar issues
- find_linked_issues($issue_number) - Find explicitly linked issues
- get_issue_metrics($issue_number) - Get priority/engagement metrics

For 'keep open' decisions, also explore solution paths considering:
- Simplicity vs Impact tradeoffs
- Risk of breaking existing functionality  
- Alignment with similar issues

Start by gathering data about issue #($issue_number) using the MCP tools, then provide your analysis and recommendation."
}

def ensure_mcp_server [] {
    let server_check = (try { 
        http get "http://localhost:8000/health" 
    } catch { 
        null 
    })
    
    if ($server_check | is-empty) {
        print "🚀 Starting MCP server in background..."
        run-external "python" "trigent-mcp/mcp_server.py" "&"
        sleep 2sec
    } else {
        print "✅ MCP server already running"
    }
}