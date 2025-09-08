# Pull GitHub issues and create rich database with embeddings

export def main [
    repo: string = "jupyterlab/jupyterlab"  # Repository to analyze
    --include-closed(-c)                    # Include closed issues
    --limit(-l): int                        # Limit number of issues
    --api-key: string                       # Mistral API key for embeddings
    --model: string = "mistral-embed"       # Mistral embedding model
] {
    print $"🔍 Fetching issues from ($repo)..."
    
    let raw_issues = fetch_issues $repo $include_closed $limit
    print $"📥 Retrieved ($raw_issues | length) issues"
    
    let enriched = ($raw_issues | each {|issue|
        print $"🔧 Enriching issue #($issue.number): ($issue.title | str substring 0..50)..."
        enrich_issue $issue $api_key $model
    })
    
    let db_file = $"rich-issues-($repo | str replace '/' '-').json.gz"
    $enriched | to json | gzip | save $db_file
    
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
    let frequency_score = calc_frequency_score $comment_count $age_days
    let priority_score = calc_priority_score $comment_count $reactions.positive $reactions.total
    
    $issue | upsert embedding $embedding
           | upsert comment_count $comment_count
           | upsert total_reactions $reactions.total
           | upsert positive_reactions $reactions.positive
           | upsert negative_reactions $reactions.negative
           | upsert age_days $age_days
           | upsert frequency_score $frequency_score
           | upsert priority_score $priority_score
           | upsert severity_score $reactions.positive
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
    let comment_reactions = ($issue.comments | each {|c| 
        calc_reaction_totals ($c.reactionGroups | default [])
    } | reduce {|it, acc| {
        total: ($acc.total + $it.total)
        positive: ($acc.positive + $it.positive) 
        negative: ($acc.negative + $it.negative)
    }} --init {total: 0, positive: 0, negative: 0})
    
    {
        total: ($issue_reactions.total + $comment_reactions.total)
        positive: ($issue_reactions.positive + $comment_reactions.positive)
        negative: ($issue_reactions.negative + $comment_reactions.negative)
    }
}

def calc_reaction_totals [reaction_groups: list] {
    let total = ($reaction_groups | each {|r| $r.totalCount} | math sum)
    let positive = ($reaction_groups | where content in ["THUMBS_UP", "HEART", "HOORAY"] | get totalCount | math sum)
    let negative = ($reaction_groups | where content in ["THUMBS_DOWN", "CONFUSED"] | get totalCount | math sum)
    
    {total: $total, positive: $positive, negative: $negative}
}

def calc_age_days [issue: record] {
    let created = ($issue.createdAt | into datetime)
    let updated = ($issue.updatedAt | into datetime)
    ($updated - $created) | format duration day | into int
}

def calc_frequency_score [comment_count: int, age_days: int] {
    if $age_days > 0 { $comment_count / $age_days } else { 0 }
}

def calc_priority_score [comment_count: int, positive_reactions: int, total_reactions: int] {
    $comment_count + $positive_reactions + $total_reactions
}

def get_mistral_embedding [content: string, api_key: string, model: string] {
    try {
        let response = (http post "https://api.mistral.ai/v1/embeddings" 
            --headers {
                "Authorization": $"Bearer ($api_key)"
                "Content-Type": "application/json"
            } 
            {
                model: $model
                input: [$content]
            })
        $response.data.0.embedding
    } catch {
        print $"⚠️  Failed to get embedding for content length: ($content | str length)"
        null
    }
}

def print_stats [enriched: list] {
    print $"📊 Statistics:"
    print $"  Total issues: ($enriched | length)"
    print $"  With embeddings: ($enriched | where embedding != null | length)"
    print $"  Average comments: ($enriched | get comment_count | math avg | math round --precision 1)"
    print $"  Average priority score: ($enriched | get priority_score | math avg | math round --precision 1)"
}