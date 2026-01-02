# Helix Memory

Long-term memory system for [Claude Code](https://docs.anthropic.com/en/docs/claude-code) using [HelixDB](https://www.helix-db.com/) graph-vector database.

Store and retrieve facts, preferences, context, and relationships across sessions using **semantic search**, **reasoning chains**, and **time-window filtering**.

## Features

- **Persistent Memory** - Remember user preferences, decisions, and project context across sessions
- **Semantic Search** - Find memories by meaning, not just keywords (via Ollama embeddings)
- **Graph Relationships** - Create IMPLIES, CONTRADICTS, BECAUSE, SUPERSEDES links between memories
- **Time-Window Filtering** - Query recent (4h), contextual (30d), deep (90d), or all memories
- **Auto-Categorization** - Memories are automatically categorized with importance scores
- **Claude Code Hooks** - Automatic memory storage and retrieval via plugin hooks

## Installation

### One-liner (Recommended)

```bash
curl -fsSL https://raw.githubusercontent.com/MarcinDudekDev/helix-memory/main/install.sh | bash
```

This installs to `~/.claude/skills/helix-memory/`, sets up HelixDB, and configures hooks automatically. Restart Claude Code to activate.

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop) (for HelixDB)
- Python 3.8+
- [Claude Code](https://docs.anthropic.com/en/docs/claude-code) CLI

### Manual Install

1. Install [Helix CLI](https://www.helix-db.com/):
```bash
curl -fsSL https://www.helix-db.com/install.sh | bash
```

2. Clone repository:
```bash
git clone https://github.com/MarcinDudekDev/helix-memory ~/.claude/skills/helix-memory
cd ~/.claude/skills/helix-memory
chmod +x memory hooks/*.py
```

3. Start HelixDB:
```bash
helix push dev
```

4. (Optional) Add alias:
```bash
echo "alias memory='~/.claude/skills/helix-memory/memory'" >> ~/.zshrc
```

## Configuration

Helix Memory reads settings from `~/.helix-memory.conf` if it exists:

```ini
[helix]
url = http://localhost:6969
data_dir = ~/.claude/skills/helix-memory

[paths]
helix_bin = ~/.local/bin/helix
cache_dir = ~/.cache/helix-memory
```

All values have sensible defaults - the config file is optional.

## Usage

### CLI Commands

```bash
# Service
memory start      # Start HelixDB (auto-starts Docker)
memory stop       # Stop HelixDB
memory status     # Check status and memory count

# Memory operations
memory search "pytest"                    # Semantic search
memory list --limit 10                    # List by importance
memory remember "User prefers FastAPI"    # Quick store with auto-categorization
memory store "content" -t fact -i 8       # Store with explicit category
memory delete abc123                      # Delete by ID
memory tag "wordpress"                    # Find by tag
```

### In Claude Code

Just mention things naturally - hooks will capture them:
- "Remember this: always use port 3000 for dev"
- "I prefer pytest over unittest"
- "The API key is stored in .env.local"

Or use explicit commands:
- `/recall pytest` - Search memories
- `/memorize User prefers tabs over spaces` - Store manually

### Hook Behavior

The plugin configures these hooks automatically:

| Hook | Action |
|------|--------|
| **SessionStart** | Loads critical preferences (importance 9+) |
| **UserPromptSubmit** | Searches relevant memories for context |
| **Stop** | Extracts and stores new learnings |
| **SessionEnd** | Saves session summary |

To configure hooks manually, add to `~/.claude/settings.json`:

```json
{
  "hooks": {
    "SessionStart": [{
      "hooks": [{
        "type": "command",
        "command": "~/.claude/skills/helix-memory/hooks/session_start.py",
        "timeout": 30
      }]
    }],
    "UserPromptSubmit": [{
      "hooks": [{
        "type": "command",
        "command": "~/.claude/skills/helix-memory/hooks/load_memories.py",
        "timeout": 10
      }]
    }],
    "Stop": [{
      "hooks": [{
        "type": "command",
        "command": "~/.claude/skills/helix-memory/hooks/session_extract.py",
        "timeout": 60
      }]
    }]
  }
}
```

## Memory Categories

| Category | Importance | Description |
|----------|------------|-------------|
| **preference** | 7-10 | User preferences that guide interactions |
| **fact** | 5-9 | Factual info about user/projects/environment |
| **context** | 4-8 | Project/domain background |
| **decision** | 6-10 | Architectural decisions with rationale |
| **task** | 3-9 | Ongoing/future tasks |
| **solution** | 6-9 | Bug fixes, problem solutions |

## Graph Schema

### Nodes
- **Memory** - Core storage unit with content, category, importance, tags
- **MemoryEmbedding** - Vector embeddings for semantic search (1536-dim)
- **Context** - Groups for project/session/topic
- **Concept** - Categorical groupings (skills, domains)

### Reasoning Edges
- **Implies** - Logical consequence ("prefers Python" → "avoid Node.js suggestions")
- **Contradicts** - Conflict detection ("use tabs" ⟷ "use spaces")
- **Because** - Causal chain ("migrated to FastAPI" ← "Flask too slow")
- **Supersedes** - Version history (new preference replaces old)

## Semantic Search Setup

### Ollama (Recommended - Local & Private)

```bash
brew install ollama
ollama pull nomic-embed-text
brew services start ollama
```

Without Ollama, falls back to keyword-based matching.

## Maintenance

```bash
cd ~/.claude/skills/helix-memory

# Cleanup junk memories
python3 smart_cleanup.py --execute

# Consolidate similar memories
python3 consolidate_memories.py --execute

# Decay old memories (reduce importance over time)
python3 memory_lifecycle.py decay --execute
```

## API Endpoints

All endpoints use POST with JSON body at `http://localhost:6969`:

```bash
# Store memory
curl -X POST http://localhost:6969/StoreMemory \
  -H "Content-Type: application/json" \
  -d '{"content": "...", "category": "preference", "importance": 8}'

# Search by similarity
curl -X POST http://localhost:6969/SearchBySimilarity \
  -H "Content-Type: application/json" \
  -d '{"query_vector": [...], "k": 10}'

# Get all memories
curl -X POST http://localhost:6969/GetAllMemories \
  -H "Content-Type: application/json" -d '{}'
```

## Troubleshooting

### HelixDB Won't Start

```bash
# Check Docker
docker ps

# Restart manually
cd ~/.claude/skills/helix-memory
helix stop dev
helix push dev
```

### Ollama Not Working

```bash
brew services restart ollama
ollama list  # Should show nomic-embed-text
```

### Vector Dimension Errors

HelixDB expects 1536-dim vectors. The code auto-pads smaller embeddings (Ollama: 768).

### Update

```bash
cd ~/.claude/skills/helix-memory && git pull
```

## Project Structure

```
~/.claude/skills/helix-memory/
├── .claude-plugin/
│   └── plugin.json         # Plugin manifest
├── db/
│   ├── schema.hx           # Graph schema (nodes, edges, vectors)
│   └── queries.hx          # HelixQL query definitions
├── hooks/
│   ├── hooks.json          # Hook configuration for plugin
│   ├── common.py           # Shared utilities
│   ├── load_memories.py    # UserPromptSubmit hook
│   ├── session_extract.py  # Stop hook
│   ├── session_start.py    # SessionStart hook
│   └── session_summary.py  # SessionEnd hook
├── skills/
│   └── helix-memory/
│       ├── SKILL.md        # Skill definition
│       └── examples/       # Usage examples
├── .helix/                 # Memory data (gitignored)
├── memory                  # CLI wrapper script
├── install.sh              # One-liner installer
├── SKILL.md                # Skill definition (symlinked)
└── *.py                    # Maintenance scripts
```

## License

MIT

## Author

[Marcin Dudek](https://github.com/MarcinDudekDev)

## Related

- [HelixDB](https://www.helix-db.com/) - Graph-vector database
- [Claude Code](https://docs.anthropic.com/en/docs/claude-code) - AI coding assistant
- [Ollama](https://ollama.ai/) - Local LLM inference
