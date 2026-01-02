#!/usr/bin/env python3
"""
Memory Helper - CLI tool for helix-memory operations.
Provides simple commands for storing, searching, and listing memories.
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from hooks.common import (
    store_memory,
    get_all_memories,
    check_helix_running,
    ensure_helix_running,
    find_similar_memories,
    delete_memory,
    llm_generate,
    extract_json_array,
    hybrid_search,
    generate_embedding,
    store_memory_embedding,
    contextual_search,
    detect_environment_from_path,
    link_memory_to_environment,
    HELIX_URL
)
import requests
import os

def cmd_store(args):
    """Store a new memory."""
    if not ensure_helix_running():
        print("ERROR: HelixDB not running", file=sys.stderr)
        sys.exit(1)

    # Check for duplicates
    similar = find_similar_memories(args.content, args.category, args.tags)
    if similar:
        print(f"WARNING: Found {len(similar)} similar memory(ies):", file=sys.stderr)
        for m in similar[:3]:
            print(f"  - [{m.get('category')}] {m.get('content')[:60]}...", file=sys.stderr)
        if not args.force:
            print("Use --force to store anyway", file=sys.stderr)
            sys.exit(1)

    memory_id = store_memory(args.content, args.category, args.importance, args.tags)
    if memory_id:
        print(f"Stored memory: {memory_id}")
        print(f"  Category: {args.category}")
        print(f"  Importance: {args.importance}")
        print(f"  Tags: {args.tags}")
    else:
        print("ERROR: Failed to store memory", file=sys.stderr)
        sys.exit(1)

def cmd_search(args):
    """Search memories using vector/semantic search + text matching."""
    if not check_helix_running():
        print("ERROR: HelixDB not running", file=sys.stderr)
        sys.exit(1)

    # Use contextual search if --context flag is set, otherwise use hybrid search
    if args.contextual:
        cwd = os.getcwd()
        matches = contextual_search(args.query, k=args.limit, cwd=cwd)
        search_type = "contextual"
    else:
        # Use hybrid search (vector similarity + text matching)
        # window="full" searches all time, not just recent memories
        matches = hybrid_search(args.query, k=args.limit, window="full")
        search_type = "hybrid"

    if not matches:
        print("No memories found matching query")
        return

    print(f"Found {len(matches)} memory(ies) ({search_type} search):\n")
    for m in matches:
        importance = m.get('importance', '?')
        category = m.get('category', 'unknown').upper()
        content = m.get('content', '')[:100]
        tags = m.get('tags', '')
        mid = m.get('id', '')  # Show full ID for easy copy-paste
        print(f"[{category} - {importance}] {content}...")
        print(f"  Tags: {tags}")
        print(f"  ID: {mid}\n")

def cmd_list(args):
    """List all memories."""
    if not check_helix_running():
        print("ERROR: HelixDB not running", file=sys.stderr)
        sys.exit(1)

    memories = get_all_memories()

    if args.category:
        memories = [m for m in memories if m.get('category') == args.category]

    # Sort by importance (desc)
    memories.sort(key=lambda m: m.get('importance', 0), reverse=True)

    if args.limit:
        memories = memories[:args.limit]

    if not memories:
        print("No memories found")
        return

    print(f"Total: {len(memories)} memories\n")
    for m in memories:
        importance = m.get('importance', '?')
        category = m.get('category', 'unknown').upper()
        content = m.get('content', '')[:80]
        mid = m.get('id', '')  # Show full ID
        print(f"[{category} - {importance}] {content}...")
        print(f"  ID: {mid}\n")

def cmd_by_tag(args):
    """Get memories by tag."""
    if not check_helix_running():
        print("ERROR: HelixDB not running", file=sys.stderr)
        sys.exit(1)

    memories = get_all_memories()
    tag = args.tag.lower()

    matches = [m for m in memories if tag in m.get('tags', '').lower()]

    if not matches:
        print(f"No memories found with tag: {args.tag}")
        return

    print(f"Found {len(matches)} memory(ies) with tag '{args.tag}':\n")
    for m in matches:
        importance = m.get('importance', '?')
        category = m.get('category', 'unknown').upper()
        content = m.get('content', '')[:80]
        print(f"[{category} - {importance}] {content}...\n")

def cmd_delete(args):
    """Delete a memory by ID (supports partial ID prefix matching)."""
    if not check_helix_running():
        print("ERROR: HelixDB not running", file=sys.stderr)
        sys.exit(1)

    memory_id = args.id

    # Support partial ID matching (prefix)
    if len(memory_id) < 36:  # Full UUID is 36 chars
        memories = get_all_memories()
        matches = [m for m in memories if m.get('id', '').startswith(memory_id)]

        if len(matches) == 0:
            print(f"ERROR: No memory found with ID prefix: {memory_id}", file=sys.stderr)
            sys.exit(1)
        elif len(matches) > 1:
            print(f"ERROR: Multiple memories match prefix '{memory_id}':", file=sys.stderr)
            for m in matches:
                print(f"  {m.get('id')} - {m.get('content', '')[:50]}...", file=sys.stderr)
            print("Provide more characters to uniquely identify", file=sys.stderr)
            sys.exit(1)
        else:
            memory_id = matches[0].get('id')
            print(f"Matched: {memory_id}")

    if delete_memory(memory_id):
        print(f"Deleted memory: {memory_id}")
    else:
        print(f"ERROR: Failed to delete memory: {memory_id}", file=sys.stderr)
        sys.exit(1)

def cmd_categorize(args):
    """Auto-categorize content using LLM (Ollama or Haiku)."""
    content = args.content

    prompt = f'''Categorize this memory. Return ONLY JSON: {{"category": "preference|fact|decision|solution", "importance": 1-10, "tags": "comma,separated"}}
Memory: {content}'''

    output, provider = llm_generate(prompt, timeout=30)
    if output:
        # Extract JSON object
        import re
        match = re.search(r'\{[^}]+\}', output)
        if match:
            print(match.group(0))
            return

    # Fallback: return default
    print('{"category": "fact", "importance": 5, "tags": ""}')

def cmd_reindex(args):
    """Generate embeddings for all memories that don't have them."""
    if not check_helix_running():
        print("ERROR: HelixDB not running", file=sys.stderr)
        sys.exit(1)

    memories = get_all_memories()
    print(f"Total memories: {len(memories)}")

    # Count existing embeddings
    to_index = []
    for m in memories:
        mid = m.get('id')
        # Check if embedding exists
        try:
            response = requests.post(
                f"{HELIX_URL}/GetMemoryEmbedding",
                json={"memory_id": mid},
                headers={"Content-Type": "application/json"},
                timeout=5
            )
            if response.status_code == 200:
                result = response.json()
                if not result.get('embedding'):
                    to_index.append(m)
            else:
                to_index.append(m)
        except:
            to_index.append(m)

    if not to_index:
        print("All memories already have embeddings!")
        return

    print(f"Memories to index: {len(to_index)}")

    if not args.force:
        confirm = input(f"Generate embeddings for {len(to_index)} memories? [y/N] ")
        if confirm.lower() != 'y':
            print("Aborted")
            return

    # Generate embeddings
    success_count = 0
    error_count = 0

    for i, m in enumerate(to_index, 1):
        mid = m.get('id')
        content = m.get('content', '')

        if not content:
            print(f"[{i}/{len(to_index)}] Skipping memory {mid[:8]} (empty content)")
            continue

        try:
            # Generate embedding
            vector, model = generate_embedding(content)

            # Store it
            if store_memory_embedding(mid, vector, content, model):
                success_count += 1
                print(f"[{i}/{len(to_index)}] Indexed {mid[:8]} using {model}")
            else:
                error_count += 1
                print(f"[{i}/{len(to_index)}] FAILED to store embedding for {mid[:8]}", file=sys.stderr)
        except Exception as e:
            error_count += 1
            print(f"[{i}/{len(to_index)}] ERROR indexing {mid[:8]}: {e}", file=sys.stderr)

    print(f"\nCompleted: {success_count} indexed, {error_count} errors")

def cmd_link_environments(args):
    """Auto-link memories to their environment contexts based on tags and content."""
    if not check_helix_running():
        print("ERROR: HelixDB not running", file=sys.stderr)
        sys.exit(1)

    memories = get_all_memories()
    print(f"Analyzing {len(memories)} memories for environment relationships...")

    # Patterns to detect environments
    env_patterns = {
        'wp-test': ['wp-test', 'wordpress', '.wp-test', 'fiverr.loc', 'local wordpress'],
        'docker': ['docker', 'container', 'docker-compose', 'dockerfile'],
        'python-venv': ['venv', 'virtual environment', 'virtualenv', 'pip install'],
        'local-sites': ['.local', '.loc', 'local site'],
    }

    linked_count = 0
    skip_count = 0

    for m in memories:
        content = m.get('content', '').lower()
        tags = m.get('tags', '').lower()
        mid = m.get('id')
        combined = f"{content} {tags}"

        # Detect environment
        detected_env = None
        for env, patterns in env_patterns.items():
            if any(pattern in combined for pattern in patterns):
                detected_env = env
                break

        if detected_env:
            if link_memory_to_environment(mid, detected_env):
                linked_count += 1
                print(f"Linked {mid[:8]} to {detected_env}")
            else:
                skip_count += 1
        else:
            skip_count += 1

    print(f"\nCompleted: {linked_count} memories linked, {skip_count} skipped")

def cmd_dedup(args):
    """Find and remove duplicate memories."""
    if not check_helix_running():
        print("ERROR: HelixDB not running", file=sys.stderr)
        sys.exit(1)

    memories = get_all_memories()
    print(f"Analyzing {len(memories)} memories for duplicates...")

    from collections import defaultdict

    # Group by similar content (first N chars + category)
    groups = defaultdict(list)
    for m in memories:
        # Use first 60 chars of content + category as key
        content = m.get('content', '')[:60].strip().lower()
        category = m.get('category', 'unknown').lower()
        key = f"{category}:{content}"
        groups[key].append(m)

    # Find duplicates
    to_delete = []
    for key, mems in groups.items():
        if len(mems) > 1:
            # Sort by importance (highest first), then by ID (keep oldest)
            mems.sort(key=lambda x: (-x.get('importance', 0), x.get('id', '')))
            # Keep first one, mark rest for deletion
            for m in mems[1:]:
                to_delete.append(m)

    if not to_delete:
        print("No duplicates found!")
        return

    print(f"\nFound {len(to_delete)} duplicates:\n")

    # Show what will be deleted
    for m in to_delete[:20]:  # Show first 20
        importance = m.get('importance', '?')
        category = m.get('category', 'unknown').upper()
        content = m.get('content', '')[:50]
        mid = m.get('id', '')
        print(f"  [{category}-{importance}] {content}... ({mid[:8]})")

    if len(to_delete) > 20:
        print(f"  ... and {len(to_delete) - 20} more")

    if args.dry_run:
        print(f"\nDry run - {len(to_delete)} would be deleted")
        return

    if not args.force:
        confirm = input(f"\nDelete {len(to_delete)} duplicates? [y/N] ")
        if confirm.lower() != 'y':
            print("Aborted")
            return

    # Delete duplicates
    deleted = 0
    errors = 0
    for m in to_delete:
        mid = m.get('id')
        if delete_memory(mid):
            deleted += 1
        else:
            errors += 1

    print(f"\nDeleted: {deleted}, Errors: {errors}")

def cmd_status(args):
    """Check HelixDB status."""
    if check_helix_running():
        memories = get_all_memories()
        print("HelixDB: RUNNING")
        print(f"Memories: {len(memories)}")

        # Count by category
        categories = {}
        for m in memories:
            cat = m.get('category', 'unknown')
            categories[cat] = categories.get(cat, 0) + 1

        if categories:
            print("\nBy category:")
            for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
                print(f"  {cat}: {count}")
    else:
        print("HelixDB: NOT RUNNING")
        print("Start with: memory start")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Helix Memory CLI Helper")
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # store command
    store_parser = subparsers.add_parser('store', help='Store a new memory')
    store_parser.add_argument('--content', '-c', required=True, help='Memory content')
    store_parser.add_argument('--category', '-t', default='fact', choices=['preference', 'fact', 'context', 'decision', 'task', 'solution'], help='Memory category')
    store_parser.add_argument('--importance', '-i', type=int, default=5, help='Importance 1-10')
    store_parser.add_argument('--tags', '-g', default='', help='Comma-separated tags')
    store_parser.add_argument('--force', '-f', action='store_true', help='Store even if similar exists')
    store_parser.set_defaults(func=cmd_store)

    # search command
    search_parser = subparsers.add_parser('search', help='Search memories')
    search_parser.add_argument('query', help='Search query')
    search_parser.add_argument('--limit', '-n', type=int, default=20, help='Max results to return (default: 20)')
    search_parser.add_argument('--contextual', '-c', action='store_true', help='Use contextual/relationship-aware search')
    search_parser.set_defaults(func=cmd_search)

    # list command
    list_parser = subparsers.add_parser('list', help='List all memories')
    list_parser.add_argument('--category', '-t', help='Filter by category')
    list_parser.add_argument('--limit', '-l', type=int, help='Limit results')
    list_parser.set_defaults(func=cmd_list)

    # by-tag command
    tag_parser = subparsers.add_parser('by-tag', help='Get memories by tag')
    tag_parser.add_argument('tag', help='Tag to search for')
    tag_parser.set_defaults(func=cmd_by_tag)

    # delete command
    delete_parser = subparsers.add_parser('delete', help='Delete a memory')
    delete_parser.add_argument('id', help='Memory ID to delete')
    delete_parser.set_defaults(func=cmd_delete)

    # status command
    status_parser = subparsers.add_parser('status', help='Check HelixDB status')
    status_parser.set_defaults(func=cmd_status)

    # categorize command (for bash script)
    cat_parser = subparsers.add_parser('categorize', help='Auto-categorize content')
    cat_parser.add_argument('content', help='Content to categorize')
    cat_parser.set_defaults(func=cmd_categorize)

    # reindex command
    reindex_parser = subparsers.add_parser('reindex', help='Generate embeddings for all memories')
    reindex_parser.add_argument('--force', '-f', action='store_true', help='Skip confirmation prompt')
    reindex_parser.set_defaults(func=cmd_reindex)

    # link-environments command
    link_parser = subparsers.add_parser('link-environments', help='Auto-link memories to environment contexts')
    link_parser.set_defaults(func=cmd_link_environments)

    # dedup command
    dedup_parser = subparsers.add_parser('dedup', help='Find and remove duplicate memories')
    dedup_parser.add_argument('--dry-run', '-n', action='store_true', help='Show duplicates without deleting')
    dedup_parser.add_argument('--force', '-f', action='store_true', help='Delete without confirmation')
    dedup_parser.set_defaults(func=cmd_dedup)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)

if __name__ == '__main__':
    main()
