# HelixDB Long-Term Memory for Claude Code

Persistent memory system enabling Claude to remember user preferences, project context, and important decisions across sessions.

## IMPORTANT: Always Use the Bash CLI

**ALWAYS use the `memory` bash script** - never call Python scripts directly.

```bash
MEMORY="/Users/cminds/Tools/helix-memory/memory"
```

## Quick Links

- **Main Skill:** [SKILL.md](SKILL.md)
- **Schema Reference:** [references/schema.md](references/schema.md)
- **Query Reference:** [references/queries.md](references/queries.md)
- **Examples:** [examples/](examples/)

## Installation Status

- ✅ HelixDB CLI installed
- ✅ Project initialized at `/Users/cminds/Tools/helix-memory`
- ✅ Schema defined and compiled
- ✅ Queries implemented
- ✅ Bash CLI with Docker auto-start

## Quick Start

### Start Database (auto-starts Docker if needed)
```bash
/Users/cminds/Tools/helix-memory/memory start
```

### Check Status
```bash
/Users/cminds/Tools/helix-memory/memory status
```

### Store a Memory
```bash
/Users/cminds/Tools/helix-memory/memory store "User prefers concise responses" -t preference -i 10 -g "communication,style"
```

### Quick Store (auto-categorizes)
```bash
/Users/cminds/Tools/helix-memory/memory remember "User prefers concise responses"
```

### List Memories
```bash
/Users/cminds/Tools/helix-memory/memory list --limit 10
```

### Stop Database
```bash
/Users/cminds/Tools/helix-memory/memory stop
```

## File Structure

```
helix-memory/
├── skill.md              # Main skill documentation
├── README.md             # This file
├── references/
│   ├── schema.md         # Database schema reference
│   └── queries.md        # Query reference
├── scripts/
│   ├── memory_helper.py  # Python CLI for memory operations
│   └── status.sh         # Status check script
└── examples/
    ├── store_preference.md   # Example: storing user preferences
    ├── session_workflow.md   # Example: complete session workflow
    └── semantic_search.md    # Example: semantic search patterns
```

## Core Concepts

### Memory Types (Categories)
1. **preference** - User preferences and settings (importance: 7-10)
2. **fact** - Factual information (importance: 5-9)
3. **context** - Project/domain background (importance: 4-8)
4. **decision** - Important decisions with rationale (importance: 6-10)
5. **task** - Ongoing/future tasks (importance: 3-9)

### Data Model
- **Memory nodes** - Individual memory entries
- **MemoryEmbedding vectors** - Enable semantic search
- **Context nodes** - Group memories by project/session
- **Relationships** - Connect related memories via graph

### Key Operations
- **Store** - Save new memory with embedding
- **Search** - Find similar memories semantically
- **Link** - Connect memories to contexts and each other
- **Retrieve** - Get memories by context, relationships, or ID

## Usage Patterns

### Session Start
1. Check DB status
2. Load user preferences (GetMemoriesInContext)
3. Search for relevant context (SearchMemoriesBySimilarity)
4. Apply learned patterns

### During Session
1. Detect important information
2. Store with appropriate category/importance
3. Generate and store embedding
4. Link to context and related memories

### Session End
1. Summarize learnings
2. Update task status
3. Store decisions made
4. Link to session context

## CLI Reference

**Always use the `memory` bash script:**

```bash
MEMORY="/Users/cminds/Tools/helix-memory/memory"

# Service commands
$MEMORY start      # Start HelixDB (auto-starts Docker Desktop if needed)
$MEMORY stop       # Stop HelixDB
$MEMORY restart    # Restart HelixDB
$MEMORY status     # Check status and memory count

# Memory commands
$MEMORY search "query"              # Search memories
$MEMORY list --limit 10             # List all memories
$MEMORY remember "text"             # Quick store (auto-categorizes)
$MEMORY store "text" -t cat -i N    # Store with category/importance
$MEMORY delete <id>                 # Delete by ID
$MEMORY tag "tagname"               # Find by tag
$MEMORY help                        # Show all commands
```

## API Access (When Deployed)

**Base URL:** http://localhost:6969

**Query Endpoint:** POST `/query/{query_name}`

**Example:**
```bash
curl -X POST http://localhost:6969/query/GetAllMemories \
  -H "Content-Type: application/json"
```

## Examples

### Store User Preference
See [examples/store_preference.md](examples/store_preference.md)

### Complete Session Workflow
See [examples/session_workflow.md](examples/session_workflow.md)

### Semantic Search Patterns
See [examples/semantic_search.md](examples/semantic_search.md)

## Current Limitations

1. **Docker build failing** - Cargo compilation issue, non-critical
2. **No embedding generation** - Requires separate service
3. **API not yet tested** - Pending successful deployment
4. **No automated workflows** - Manual operations for now

## Future Enhancements

- [ ] Automated embedding generation
- [ ] Smart duplicate detection
- [ ] Importance auto-adjustment
- [ ] Memory decay over time
- [ ] Export/import functionality
- [ ] Backup automation
- [ ] Web UI for management
- [ ] Integration with Claude Code workflows

## Troubleshooting

### DB Won't Start
```bash
# Use the memory script (handles Docker auto-start)
/Users/cminds/Tools/helix-memory/memory start

# Check container status
docker ps | grep helix
```

### Connection Issues
```bash
# Restart using the memory script
/Users/cminds/Tools/helix-memory/memory restart
```

## Resources

- **HelixDB Docs:** https://docs.helix-db.com
- **GitHub:** https://github.com/HelixDB/helix-db
- **Quickstart:** https://github.com/HelixDB/quickstart
- **Project:** /Users/cminds/Tools/helix-memory

## License

This skill documentation is part of the user's private Claude Code skills.
HelixDB is licensed under AGPL-3.0 (see HelixDB repository).
