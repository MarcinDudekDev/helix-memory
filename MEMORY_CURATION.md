# Memory Curation Process

Automated garbage detection + manual review for helix-memory quality.

## Quick Start

```bash
# 1. Check current state
memory status

# 2. Run automated scan (outputs candidate IDs)
memory-curate scan

# 3. Or generate full report
memory-curate report > /tmp/curation-report.md

# 4. Review each candidate
memory get <id>

# 5. Take action: DELETE / REWRITE / LINK / KEEP
```

## The `memory-curate` Tool

```bash
memory-curate scan      # Output candidate IDs with metadata
memory-curate report    # Generate full markdown report
memory-curate patterns  # List all garbage detection patterns
```

Output format for `scan`:
```
<id>|<matched_pattern>|<category>|<auto_flag>
```

## Decision Tree

For each candidate:

```
Is it a complete, standalone sentence?
├─ NO → Can it be REWRITTEN to stand alone?
│       ├─ YES → REWRITE (add context, merge with related)
│       └─ NO  → DELETE
│
└─ YES → Does it name specific files/tools/patterns?
         ├─ NO → Is it LINKED to memories that provide context?
         │       ├─ YES → KEEP (graph provides meaning)
         │       └─ NO  → Can you LINK it to related memories?
         │               ├─ YES → LINK then KEEP
         │               └─ NO  → DELETE (orphan with no value)
         │
         └─ YES → Would a DIFFERENT session find this useful?
                  ├─ YES → KEEP
                  └─ NO  → DELETE (session-specific)
```

## Actions

### DELETE - Confirmed garbage
```bash
memory delete <id> --force
```

### REWRITE - Salvageable with better content
```bash
# 1. Get the original
memory get <id>

# 2. Delete the bad version
memory delete <id> --force

# 3. Store improved version
memorize -t <category> -i <importance> "<improved content>"
```

**Rewrite examples:**
```bash
# BAD: "Decided to remove the symlink"
# GOOD: "Removed ~/Tools/memory symlink - now points to ~/Tools/helix-memory/memory directly"

# BAD: "1. Extract memories via session_extract.py"
# GOOD: "Memory extraction pipeline: session_extract.py reads transcript → spawns simple_scribe.py → stores via main.py"

# BAD: "It works now"
# GOOD: "CORS fix verified: CORSMiddleware must be added BEFORE route definitions in FastAPI"
```

### LINK - Add graph connections for context
```bash
# 1. Find related memories
memory search "<related topic>"

# 2. Link them (creates RELATED edge)
memory link <source_id> <target_id>

# Or create explicit edge types:
memory link <source_id> <target_id> --type IMPLIES
memory link <source_id> <target_id> --type SOLVES
```

**When to LINK instead of DELETE:**
- Fragment has high importance (7+) but missing context
- Multiple fragments together form a coherent thought
- Memory references another memory that explains it

### KEEP - Good memory, false positive
No action needed. Pattern matched but memory is valid.

## Garbage Patterns by Category

### Session-Specific (usually DELETE)
- "committed", "pushed", "verified", "confirmed", "pulled"
- "Worked on multiple projects", "Worked across"
- "Plan saved at", "New session started"

### Fragments (REWRITE or DELETE)
- "and/or", "Option", "Possible", "Also"
- Numbered items: "1.", "2.", "3." without context

### Meta-Comments (DELETE)
- "Added memory about", "stored", "summary of the conversation"
- "assistant suggested", "LLM"

### Vague References (REWRITE or DELETE)
- "decided to", "removed", "the symlink"
- "it works", "working"

### Outdated Status (DELETE)
- "memories available", "ACTIVE with", "grew from"
- Specific counts that change over time

### Emotional/Status (DELETE)
- "satisfied", "thanks", "finally"
- "I think" (unless followed by decision rationale)

## Batch Processing

For automated cleanup with confirmation:

```bash
# Generate candidate list
memory-curate scan > /tmp/candidates.txt

# Review and mark for deletion (edit file, remove lines to keep)
vim /tmp/candidates.txt

# Delete all remaining candidates
while IFS='|' read -r id rest; do
    echo "Deleting: $id"
    memory delete "$id" --force
done < /tmp/candidates.txt
```

## Monthly Curation Checklist

1. [ ] Run `memory status` - note starting count
2. [ ] Run `memory-curate report > /tmp/curation-$(date +%Y%m).md`
3. [ ] Review report, mark actions (DELETE/REWRITE/LINK/KEEP)
4. [ ] Process deletions first (quick wins)
5. [ ] Process rewrites (improve quality)
6. [ ] Process links (strengthen graph)
7. [ ] Run `memory status` - note ending count
8. [ ] Store summary: `memorize "Curation $(date +%Y-%m): deleted X, rewrote Y, linked Z"`

## Metrics

Healthy curation session:
- Delete 5-15% of flagged candidates
- Rewrite 1-5% (high-value salvage)
- Link 2-10% (strengthen graph)
- Keep 70-90% (false positives are normal)

If >50% are garbage, review extraction quality or update `/memorize` guidelines.
