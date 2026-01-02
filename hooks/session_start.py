#!/usr/bin/env python3
"""
SessionStart Hook: Load Memory Status and Critical Preferences

Executes at the START of each session (before any user interaction).
Provides Claude with:
1. HelixDB status (running/not running, memory count)
2. All high-importance memories (preferences, critical facts)

This ensures Claude always knows user preferences from the first message.
"""

import sys
import json
from pathlib import Path

# Add hooks directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from common import (
    check_helix_running,
    get_all_memories,
    detect_project,
    format_hook_output,
    print_hook_output
)
import os


def get_memory_status() -> dict:
    """Get HelixDB status and memory counts."""
    if not check_helix_running():
        return {"running": False, "count": 0, "categories": {}}

    memories = get_all_memories()

    # Count by category
    categories = {}
    for m in memories:
        cat = m.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1

    return {
        "running": True,
        "count": len(memories),
        "categories": categories
    }


def get_critical_memories(min_importance: int = 8, project: str = None) -> list:
    """
    Get high-importance memories for session start.

    If project specified, prioritizes project-specific memories first,
    then global critical preferences.

    Returns memories sorted by importance (highest first),
    then by category priority (preferences first).
    """
    if not check_helix_running():
        return []

    memories = get_all_memories()
    if not memories:
        return []

    # Filter by importance
    high_imp = [m for m in memories if m.get("importance", 0) >= min_importance]

    # If project specified, prioritize project-specific memories
    if project:
        project_lower = project.lower()
        project_memories = []
        global_memories = []

        for m in high_imp:
            tags = m.get("tags", "").lower()
            if project_lower in tags:
                project_memories.append(m)
            else:
                global_memories.append(m)

        # Sort each group by importance
        category_priority = {"preference": 0, "decision": 1, "fact": 2, "solution": 3, "context": 4, "task": 5}
        sort_key = lambda x: (-x.get("importance", 0), category_priority.get(x.get("category", ""), 6))

        project_memories.sort(key=sort_key)
        global_memories.sort(key=sort_key)

        # Combine: project-specific first (top 5), then global (top 5)
        return project_memories[:5] + global_memories[:5]

    # No project filter - just sort all
    category_priority = {"preference": 0, "decision": 1, "fact": 2, "context": 3, "task": 4}
    high_imp.sort(key=lambda x: (
        -x.get("importance", 0),
        category_priority.get(x.get("category", ""), 5)
    ))

    return high_imp[:10]


def format_session_context(status: dict, memories: list, project: str = None) -> str:
    """Format status and memories as session start context.

    OPTIMIZED: Minimal context injection to reduce token overhead.
    Only loads importance>=9 memories (truly critical preferences).
    Use /recall or trigger words to load more context on-demand.
    """
    parts = []

    # Minimal status - just availability notice
    if status["running"]:
        parts.append(f"HelixDB Memory: ACTIVE ({status['count']} memories available)")
        parts.append("Use 'recall' or 'remember' in prompts to load relevant context.")
    else:
        parts.append("HelixDB Memory: NOT RUNNING")
        return "\n".join(parts)

    # Only show importance>=9 (truly critical, ~10-15 items max)
    if memories:
        parts.append("\n## Critical Preferences (importance ≥9):\n")
        for m in memories:
            cat = m.get("category", "unknown").upper()
            imp = m.get("importance", "?")
            content = m.get("content", "")
            parts.append(f"[{cat} - {imp}] {content}\n")

        # Count lower-importance memories not shown
        all_count = status.get("count", 0)
        shown_count = len(memories)
        hidden_count = all_count - shown_count
        if hidden_count > 0:
            parts.append(f"\n📚 {hidden_count} more memories available (importance <9). Use `/recall <topic>` to search.")

    # Meta-instruction for recall
    if project:
        parts.append(f"\n## Current Project: {project}")
        parts.append("If you need facts about this project (URLs, credentials, past decisions), ")
        parts.append("use `/recall [topic]` or include 'recall' in your prompt to search memories.")

    return "\n".join(parts)


def main():
    """Main hook execution."""
    # Detect current project
    cwd = os.environ.get('CWD', os.getcwd())
    project = detect_project(cwd)

    # Get status
    status = get_memory_status()

    # Get critical memories (importance>=9, prioritize current project)
    memories = get_critical_memories(min_importance=9, project=project)

    # Format context with project info and meta-instruction
    context = format_session_context(status, memories, project=project)

    # Return as session start context
    output = {
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": context
        },
        "suppressOutput": True
    }

    print_hook_output(output)
    sys.exit(0)


if __name__ == "__main__":
    main()
