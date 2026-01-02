<p align="center">
  <img src="assets/header.png" alt="Helix Memory - Long-term memory for Claude Code" width="100%">
</p>

# Helix Memory

Long-term memory system for [Claude Code](https://docs.anthropic.com/en/docs/claude-code) using [HelixDB](https://www.helix-db.com/) graph-vector database.

Store and retrieve facts, preferences, context, and relationships across sessions using **semantic search**, **reasoning chains**, and **time-window filtering**.

## Features

- **Persistent Memory** - Remember user preferences, decisions, and project context across sessions
- **Semantic Search** - Find memories by meaning, not just keywords (via Ollama or Gemini embeddings)
- **Graph Relationships** - Create IMPLIES, CONTRADICTS, BECAUSE, SUPERSEDES links between memories
- **Time-Window Filtering** - Query recent (4h), contextual (30d), deep (90d), or all memories
- **Auto-Categorization** - Memories are automatically categorized with importance scores
- **Claude Code Hooks** - Automatic memory storage and retrieval via hooks

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop) (for HelixDB)
- Python 3.8+
- [Claude Code](https://docs.anthropic.com/en/docs/claude-code) CLI

Optional (for semantic search):
- [Ollama](https://ollama.ai/) with `nomic-embed-text` model (recommended, local)
- Or: `GEMINI_API_KEY` environment variable (free tier, external API)

## Installation

### Quick Install

```bash
git clone https://github.com/MarcinDudekDev/helix-memory
cd helix-memory
./install.sh
```

### Manual Install

1. Install [Helix CLI](https://www.helix-db.com/):
```bash
curl -fsSL https://www.helix-db.com/install.sh | bash
```

2. Copy files:
```bash
# Tool files
mkdir -p ~/Tools/helix-memory
cp -r db hooks scripts memory helix.toml *.py ~/Tools/helix-memory/
chmod +x ~/Tools/helix-memory/memory

# Skill files (for Claude Code)
mkdir -p ~/.claude/skills/helix-memory
cp -r skills/helix-memory/* ~/.claude/skills/helix-memory/
```

3. Start HelixDB:
```bash
cd ~/Tools/helix-memory
helix push dev
```

## Usage

### CLI Commands

```bash
MEMORY=~/Tools/helix-memory/memory

# Service
$MEMORY start      # Start HelixDB (auto-starts Docker)
$MEMORY stop       # Stop HelixDB
$MEMORY status     # Check status and memory count

# Memory operations
$MEMORY search "pytest"                    # Search memories
$MEMORY list --limit 10                    # List by importance
$MEMORY remember "User prefers FastAPI"    # Quick store with auto-categorization
$MEMORY store "content" -t fact -i 8       # Store with explicit category
$MEMORY delete abc123                      # Delete by ID
$MEMORY tag "wordpress"                    # Find by tag
```

### In Claude Code

Just mention things naturally - hooks will capture them:
- "Remember this: always use port 3000 for dev"
- "I prefer pytest over unittest"
- "The API key is stored in .env.local"

Or use explicit commands:
- `/recall pytest` - Search memories
- `/memorize User prefers tabs over spaces` - Store manually

### Hook Configuration

Add to `~/.claude/settings.json`:

```json
{
  "hooks": {
    "Stop": [{
      "hooks": [{
        "type": "command",
        "command": "~/Tools/helix-memory/hooks/reflect_and_store.py",
        "timeout": 60
      }]
    }],
    "UserPromptSubmit": [{
      "hooks": [{
        "type": "command",
        "command": "~/Tools/helix-memory/hooks/load_memories.py",
        "timeout": 30
      }]
    }],
    "SessionStart": [{
      "hooks": [{
        "type": "command",
        "command": "~/Tools/helix-memory/hooks/session_start.py",
        "timeout": 30
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

### Option 1: Ollama (Recommended - Local & Private)

```bash
brew install ollama
ollama pull nomic-embed-text
brew services start ollama
```

### Option 2: Gemini API (Free Tier - External)

```bash
export GEMINI_API_KEY="your-api-key"
```

Get a free key at: https://makersuite.google.com/app/apikey

Without either, falls back to hash-based pseudo-embeddings (deterministic but not semantic).

## Maintenance

```bash
# Cleanup junk memories
python3 smart_cleanup.py --execute

# Consolidate similar memories
python3 consolidate_memories.py --execute

# Decay old memories (reduce importance over time)
python3 memory_lifecycle.py decay --execute

# Weekly digest
python3 weekly_digest.py
```

### Scheduled Decay (macOS)

Auto-run decay every Sunday at 3 AM:

```bash
cp com.helix-memory.decay.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.helix-memory.decay.plist
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
cd ~/Tools/helix-memory
helix stop dev
helix push dev
```

### Ollama Not Working

```bash
brew services restart ollama
ollama list  # Should show nomic-embed-text
```

### Vector Dimension Errors

HelixDB expects 1536-dim vectors. The code auto-pads smaller embeddings (Ollama: 768, Gemini: 768).

## Project Structure

```
helix-memory/
├── db/
│   ├── schema.hx          # Graph schema (nodes, edges, vectors)
│   └── queries.hx         # HelixQL query definitions
├── hooks/
│   ├── common.py          # Shared utilities
│   ├── load_memories.py   # UserPromptSubmit hook
│   ├── reflect_and_store.py   # Stop hook
│   └── session_start.py   # SessionStart hook
├── scripts/
│   └── memory_helper.py   # CLI helper
├── skills/
│   └── helix-memory/      # Claude Code skill
│       ├── SKILL.md       # Skill definition
│       └── examples/      # Usage examples
├── memory                 # CLI wrapper script
├── install.sh             # Installer
└── *.py                   # Maintenance scripts
```

## License

MIT

## Author

[Marcin Dudek](https://github.com/MarcinDudekDev)

## Related

- [HelixDB](https://www.helix-db.com/) - Graph-vector database
- [Claude Code](https://docs.anthropic.com/en/docs/claude-code) - AI coding assistant
- [Ollama](https://ollama.ai/) - Local LLM inference
