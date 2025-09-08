# Pull GitHub issues and create rich database with embeddings

export def main [
    repo: string = "jupyterlab/jupyterlab"  # Repository to analyze
    --include-closed(-c)                    # Include closed issues
    --limit(-l): int                        # Limit number of issues
    --api_key: string                       # Mistral API key for embeddings
    --model: string = "mistral-embed"       # Mistral embedding model
] {
    print $"🔍 Fetching issues from ($repo)..."
    
    let raw_issues = fetch_issues $repo $include_closed $limit
    print $"📥 Retrieved ($raw_issues | length) issues"
    
    let enriched = ($raw_issues | each {|issue|
        print $"🔧 Enriching issue #($issue.number): ($issue.title | str substring 0..50)..."
        enrich_issue $issue $api_key $model
    })
    
    print "🔧 Computing quartile assignments..."
    let enriched = $enriched | add_quartile_columns

    let db_file = $"rich-issues-($repo | str replace '/' '-').json.gz"
    $enriched | to json | gzip | save -f $db_file

    print $"✅ Rich issue database saved to ($db_file)"
    print_stats $enriched
}

def fetch_issues [repo: string, include_closed: bool, limit?: int] {
    let state = if $include_closed { "all" } else { "open" }
    let gh_cmd = [
        "gh" "issue" "list" 
        "--repo" $repo 
        "--state" $state
        "--json" "number,title,body,state,createdAt,updatedAt,author,labels,assignees,url,comments,reactionGroups"
    ]
    
    let gh_cmd = if ($limit | is-not-empty) { 
        $gh_cmd | append ["--limit" ($limit | into string)] 
    } else { $gh_cmd }
    
    run-external ...$gh_cmd | from json
}

def enrich_issue [issue: record, api_key?: string, model?: string] {
    let embedding = get_issue_embedding $issue $api_key $model
    let comment_count = calc_comment_count $issue
    let reactions = calc_reaction_metrics $issue  
    let age_days = calc_age_days $issue
    let activity_score = calc_activity_score $comment_count $age_days
    
    $issue | upsert embedding $embedding
           | upsert comment_count $comment_count
           | upsert conversation_total_emojis $reactions.conversation_total_emojis
           | upsert issue_positive_emojis $reactions.issue_positive_emojis
           | upsert issue_negative_emojis $reactions.issue_negative_emojis
           | upsert age_days $age_days
           | upsert activity_score $activity_score
}

def get_issue_embedding [issue: record, api_key?: string, model?: string] {
    if ($api_key | is-empty) { return null }
    
    let content = [
        $issue.title
        ($issue.body | default "")
        ($issue.comments | each {|c| $c.body | default ""} | str join "\n")
    ] | str join "\n" | str trim
    
    if ($content | str length) == 0 { return null }
    
    get_mistral_embedding $content $api_key $model
}

def calc_comment_count [issue: record] {
    $issue.comments | length
}

def calc_reaction_metrics [issue: record] {
    let issue_reactions = calc_reaction_totals ($issue.reactionGroups | default [])
    # note that negative and positive reactions on a comment are not interpretable
#   # if we don't know whether the comment is affirmative or negative
    let comment_reactions = ($issue.comments | each {|c| 
        calc_reaction_totals ($c.reactionGroups | default [])
    } | reduce -f {total: 0} {|it, acc| {
        total: ($acc.total + $it.total)
    }})
    
    {
        issue_total_emojis: ($issue_reactions.total + $comment_reactions.total)
        issue_positive_emojis: ($issue_reactions.positive)
        issue_negative_emojis: ($issue_reactions.negative)
        conversation_total_emojis: ($issue_reactions.total + $comment_reactions.total)
    }
}

def calc_reaction_totals [reaction_groups: list] {
    let total = ($reaction_groups | each {|r| $r.totalCount? | default 0} | default -e [0] | math sum)
    let positive = ($reaction_groups | where content in ["THUMBS_UP", "HEART", "HOORAY"] | each {|r| $r.totalCount? | default 0} | default -e [0] | math sum)
    let negative = ($reaction_groups | where content in ["THUMBS_DOWN", "CONFUSED"] | each {|r| $r.totalCount? | default 0} | default -e [0] | math sum)
    
    {total: $total, positive: $positive, negative: $negative}
}

def calc_age_days [issue: record] {
    let created = ($issue.createdAt | into datetime)
    let updated = ($issue.updatedAt | into datetime)
    (($updated - $created) / 1day) | math round | into int
}

def calc_activity_score [comment_count: int, age_days: int] {
    if $age_days > 0 { $comment_count / $age_days } else { 0.0 }
}

def get_mistral_embedding [content: string, api_key: string, model: string] {
    try {
        let payload = {
            model: $model
            input: [$content]
        }
        
        let response = (http post "https://api.mistral.ai/v1/embeddings" 
            --headers {
                "Authorization": $"Bearer ($api_key)"
                "Content-Type": "application/json"
            } 
            ($payload | to json))
        $response.data.0.embedding
    } catch {|err|
        print $"❌ API call failed: ($err.msg)"
        null
    }
}

def histogram_to_sparkline [hist: table] {
    let sparkline_chars = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"]
    let quantiles = ($hist | get quantile)
    let max_quantile = ($quantiles | math max)
    
    if $max_quantile == 0 { return "" }
    
    let sparkline = ($quantiles | each {|quantile|
        let normalized = ($quantile / $max_quantile * 7) | math round | into int
        $sparkline_chars | get $normalized
    } | str join "")
    
    let range_start = ($hist | first | get value)
    let range_end = ($hist | last | get value)
    $"($sparkline) \(($range_start)-($range_end)\)"
}

def add_quartile_columns []: table -> table {
    let metrics = ["comment_count", "age_days", "activity_score", "issue_positive_emojis", "issue_negative_emojis", "conversation_total_emojis"]
    let polarsDF = $in | select ...($metrics | append "number") | polars into-df
    
    mut df = $polarsDF
    for $metric in $metrics {
        # Create expressions to assign quartiles based on quantile thresholds
        let exprs = [0.25 0.5 0.75 1.0] | each {|q| $q; #strange workaround for a bug 
            polars when ((polars col $metric) <= (polars col $metric | polars quantile $q)) $q
                | polars otherwise NaN | polars as $"q_($q)"
        }
        
        # Add quartile columns and find the minimum (first matching quartile)
        $df = ($df | polars with-column (polars horizontal min ...$exprs | polars as $"($metric)_q"))
    }
    
    $df | polars collect | polars into-nu
}

def quartile_sparkline [data: list, metric: string] {
    let sparkline_chars = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"]
    let q_col = $"($metric)_q"
    
    let quartile_values = ($data | select $metric $q_col)
    let quartiles = [0.25, 0.5, 0.75, 1.0]
    let counts = ($quartiles | each {|q|
        let filtered_quartile_values = ($quartile_values | where {|r| ((($r | get $q_col) - $q) | math abs) <= 0.0001})
        let count = ($filtered_quartile_values | length)
        let max_val = ($filtered_quartile_values | get $metric | default -e [0] | math max)
        {quartile: $q, count: $count, max: $max_val}
    })
    
    let max_count = ($counts | get count | math max)
    
    let sparkline = if $max_count == 0 {
        "▁▁▁▁"
    } else {
        ($counts | each {|item|
            let normalized = ($item.count / $max_count * 7) | math round | into int
            $sparkline_chars | get $normalized
        } | str join "")
    }
    
    let legend_parts = ($counts | each {|item|
        let pct = ($item.quartile * 100 | into int)
        $"($pct)%:($item.max):($item.count)"
    })
    
    $"($sparkline) \(≤($legend_parts | str join ' / ')\)"
}

def print_stats [enriched: list] {
    print $"📊 Statistics:"
    print $"  Total issues: ($enriched | length)"
    print $"  With embeddings: ($enriched | where embedding != null | length)"
    print ""
    
    print $"📈 Comment Count: (quartile_sparkline $enriched 'comment_count')"
    print $"📈 Age \(days\): (quartile_sparkline $enriched 'age_days')"  
    print $"📈 Activity Score: (quartile_sparkline $enriched 'activity_score')"
    print $"📈 Issue Positive Emojis: (quartile_sparkline $enriched 'issue_positive_emojis')"
    print $"📈 Issue Negative Emojis: (quartile_sparkline $enriched 'issue_negative_emojis')"
    print $"📈 Conversation Total Emojis: (quartile_sparkline $enriched 'conversation_total_emojis')"
}
