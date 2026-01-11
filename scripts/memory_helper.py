#!/usr/bin/env python3
"""
Memory Helper - CLI tool for helix-memory operations.
Provides simple commands for storing, searching, and listing memories.
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta

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


def parse_memory_timestamp(memory: dict) -> datetime | None:
    """
    Extract creation timestamp from memory.
    First tries created_at field (ISO timestamp), falls back to UUID parsing.
    """
    # Try created_at field first (ISO format)
    created_at = memory.get('created_at')
    if created_at:
        try:
            # Handle ISO format with/without microseconds
            if '.' in created_at:
                return datetime.fromisoformat(created_at)
            else:
                return datetime.fromisoformat(created_at.replace('Z', '+00:00'))
        except (ValueError, TypeError):
            pass

    # Fallback: try to parse from UUID (less reliable)
    uuid_str = memory.get('id', '')
    if uuid_str:
        try:
            # Remove hyphens and get first 12 hex chars
            hex_str = uuid_str.replace('-', '')[:12]
            unix_ms = int(hex_str, 16)
            # Validate reasonable timestamp range (2020-2030)
            if 1577836800000 < unix_ms < 1893456000000:
                return datetime.fromtimestamp(unix_ms / 1000.0)
        except (ValueError, OSError):
            pass

    return None


def parse_date_arg(date_str: str) -> datetime | None:
    """
    Parse date argument. Supports:
    - 'yesterday', 'today'
    - 'YYYY-MM-DD' format
    - 'N days ago' format
    """
    date_str = date_str.strip().lower()

    if date_str == 'today':
        return datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    elif date_str == 'yesterday':
        return (datetime.now() - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    elif date_str.endswith(' days ago'):
        try:
            days = int(date_str.split()[0])
            return (datetime.now() - timedelta(days=days)).replace(hour=0, minute=0, second=0, microsecond=0)
        except ValueError:
            pass

    # Try ISO format YYYY-MM-DD
    try:
        return datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        pass

    return None


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

    # Date filtering via UUID v7 timestamp
    show_dates = False
    since_date = None
    exact_date = None

    if args.since:
        since_date = parse_date_arg(args.since)
        if not since_date:
            print(f"ERROR: Invalid date format: {args.since}", file=sys.stderr)
            print("Use: yesterday, today, YYYY-MM-DD, or 'N days ago'", file=sys.stderr)
            sys.exit(1)
        show_dates = True

    if args.date:
        exact_date = parse_date_arg(args.date)
        if not exact_date:
            print(f"ERROR: Invalid date format: {args.date}", file=sys.stderr)
            print("Use: yesterday, today, or YYYY-MM-DD", file=sys.stderr)
            sys.exit(1)
        show_dates = True

    # Filter memories by date
    if since_date or exact_date:
        filtered = []
        for m in memories:
            created = parse_memory_timestamp(m)
            if not created:
                continue  # Skip if can't parse timestamp

            if since_date and created >= since_date:
                m['_created_at'] = created
                filtered.append(m)
            elif exact_date:
                # Exact date match (same day)
                next_day = exact_date + timedelta(days=1)
                if exact_date <= created < next_day:
                    m['_created_at'] = created
                    filtered.append(m)
        memories = filtered

    # Sort by importance (desc), or by date if filtering
    if show_dates:
        memories.sort(key=lambda m: m.get('_created_at', datetime.min), reverse=True)
    else:
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
        mid = m.get('id', '')

        if show_dates and '_created_at' in m:
            created_str = m['_created_at'].strftime('%Y-%m-%d %H:%M')
            print(f"[{category} - {importance}] {created_str}")
            print(f"  {content}...")
        else:
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


# ============================================================================
# GARBAGE DETECTION PATTERNS
# ============================================================================

GARBAGE_PATTERNS = [
    r"^(yes|no|ok|done|thanks)\.?$",  # Too short/vague
    r"were provided\.?$",              # Vague reference
    r"was (set|created|updated)\.?$",  # Vague without specifics
    r"^The (user|assistant) ",         # Meta-description
    r'^<bash-',                         # Bash output markers
    r'bash-stdout|bash-stderr',
    r'^\[.*\]$',                        # Single bracketed content
    r'^Perfect!',                       # Generic acknowledgments
    r'^Done!',
    r'^Great!',
    r'^Let me',                         # Task narration
    r'^Now let me',
    r'^Now I',
    r"^I'll",
    r"^I'm going to",
    r'^\s*\[\d+',                       # Line number prefixes
    r'^\s*cat /',                       # Raw cat output
    r'server\s*\{',                     # Nginx configs
    r'location\s*~',
    r'#!/bin/bash',                     # Script content
    r'#!/usr/bin/env',
    r'^Base directory for this skill:', # Skill loading
    r'This session is being continued', # Session continuations
    r'\[38;5;',                         # Terminal escape codes
    r'\[\?2026',
    r'^Project path:.*Context:.*<bash', # Corrupted path+bash combo
    r'^Project location:.*`\)',         # Malformed path
]


def is_garbage(content: str) -> bool:
    """Check if content matches garbage patterns."""
    import re
    content = content.strip()

    # Too short
    if len(content) < 20:
        return True

    for pattern in GARBAGE_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE | re.MULTILINE):
            return True
    return False


def is_corrupted(content: str) -> bool:
    """Check for corrupted/malformed memories."""
    content = content.strip()
    if len(content) < 20:
        return True
    if content.startswith(('[@', '[?', '</', '`)', '...')):
        return True
    if content.count('<bash') != content.count('</bash'):
        return True
    return False


def find_semantic_duplicates(memories: list, threshold: float = 0.85, use_vectors: bool = True) -> list:
    """
    Find semantic duplicates using vector similarity OR text prefix matching.
    Returns list of (keep_id, delete_ids, similarity_score) tuples.

    When use_vectors=True (default), uses actual embedding cosine similarity.
    Falls back to prefix matching for speed when vectors unavailable.
    """
    from collections import defaultdict

    duplicates = []
    processed_ids = set()

    if use_vectors:
        # Use vector similarity search from HelixDB
        from hooks.common import search_by_similarity, calculate_semantic_similarity

        for i, m in enumerate(memories):
            mid = m.get('id', '')
            if mid in processed_ids:
                continue

            content = m.get('content', '')
            if len(content) < 20:
                continue

            # Search for similar memories using vector similarity
            similar = search_by_similarity(content, k=15, window="full")

            # Group truly similar ones (>threshold)
            group = [m]
            for s in similar:
                sid = s.get('id', '')
                if sid == mid or sid in processed_ids:
                    continue

                # Calculate actual semantic similarity
                sim_score = calculate_semantic_similarity(content, s.get('content', ''))
                if sim_score >= threshold:
                    group.append(s)
                    processed_ids.add(sid)

            if len(group) > 1:
                # Sort: highest importance first, then oldest ID
                group.sort(key=lambda x: (-x.get('importance', 0), x.get('id', '')))
                keep = group[0]
                delete_ids = [g.get('id') for g in group[1:]]
                avg_sim = threshold  # Approximate
                duplicates.append((keep.get('id'), delete_ids, avg_sim))
                processed_ids.add(mid)

            # Progress indicator
            if i > 0 and i % 100 == 0:
                print(f"  Processed {i}/{len(memories)} memories...", file=sys.stderr)
    else:
        # Fallback: text prefix matching (fast but less accurate)
        groups = defaultdict(list)
        for m in memories:
            content = m.get('content', '')[:60].strip().lower()
            category = m.get('category', 'unknown').lower()
            key = f"{category}:{content}"
            groups[key].append(m)

        for key, group in groups.items():
            if len(group) > 1:
                group.sort(key=lambda x: (-x.get('importance', 0), x.get('id', '')))
                keep = group[0]
                delete_ids = [g.get('id') for g in group[1:]]
                duplicates.append((keep.get('id'), delete_ids, 0.90))

    return duplicates


def find_duplicates_fast(memories: list) -> list:
    """
    Fast duplicate detection using text prefix + hash.
    Good for initial cleanup, use find_semantic_duplicates for thorough check.
    """
    from collections import defaultdict
    from hooks.common import content_hash

    # Group by content hash (exact duplicates)
    hash_groups = defaultdict(list)
    for m in memories:
        h = content_hash(m.get('content', ''))
        hash_groups[h].append(m)

    duplicates = []
    for h, group in hash_groups.items():
        if len(group) > 1:
            group.sort(key=lambda x: (-x.get('importance', 0), x.get('id', '')))
            keep = group[0]
            delete_ids = [g.get('id') for g in group[1:]]
            duplicates.append((keep.get('id'), delete_ids, 1.0))  # Exact match

    # Also check prefix matches within same category
    prefix_groups = defaultdict(list)
    seen_ids = {d[0] for d in duplicates} | {did for d in duplicates for did in d[1]}

    for m in memories:
        if m.get('id') in seen_ids:
            continue
        content = m.get('content', '')[:60].strip().lower()
        category = m.get('category', 'unknown').lower()
        key = f"{category}:{content}"
        prefix_groups[key].append(m)

    for key, group in prefix_groups.items():
        if len(group) > 1:
            group.sort(key=lambda x: (-x.get('importance', 0), x.get('id', '')))
            keep = group[0]
            delete_ids = [g.get('id') for g in group[1:]]
            duplicates.append((keep.get('id'), delete_ids, 0.90))

    return duplicates


def cmd_health(args):
    """Show memory health report with vector-based duplicate detection."""
    if not check_helix_running():
        print("ERROR: HelixDB not running", file=sys.stderr)
        sys.exit(1)

    memories = get_all_memories()
    total = len(memories)

    # Find duplicates - fast mode uses hash+prefix, thorough uses vectors
    if args.thorough:
        print("Analyzing with vector similarity (may take a few minutes)...", file=sys.stderr)
        duplicates = find_semantic_duplicates(memories, threshold=0.85, use_vectors=True)
    else:
        duplicates = find_duplicates_fast(memories)
    dup_count = sum(len(d[1]) for d in duplicates)

    # Find garbage
    garbage = [m for m in memories if is_garbage(m.get('content', '')) or is_corrupted(m.get('content', ''))]

    # Category counts
    categories = {}
    for m in memories:
        cat = m.get('category', 'unknown')
        categories[cat] = categories.get(cat, 0) + 1

    # Importance distribution
    importance_dist = {}
    for m in memories:
        imp = m.get('importance', 0)
        importance_dist[imp] = importance_dist.get(imp, 0) + 1

    # Check for linked memories (graph connectivity)
    from hooks.common import get_related_memories
    linked_count = 0
    orphan_count = 0
    for m in memories[:100]:  # Sample first 100 for speed
        related = get_related_memories(m.get('id', ''))
        if related:
            linked_count += 1
        else:
            orphan_count += 1

    print("=" * 50)
    print("MEMORY HEALTH REPORT")
    print("=" * 50)
    print(f"\nTotal Memories: {total}")
    print(f"Duplicates:     {dup_count} (in {len(duplicates)} groups)")
    print(f"Garbage:        {len(garbage)}")
    print(f"Linked:         ~{linked_count}% (sampled)")
    print(f"Orphans:        ~{orphan_count}% (no edges)")
    print(f"Clean:          {total - dup_count - len(garbage)}")

    print("\nBy Category:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1])[:10]:
        print(f"  {cat}: {count}")

    print("\nBy Importance:")
    for imp in sorted(importance_dist.keys(), reverse=True):
        print(f"  {imp}: {importance_dist[imp]}")

    if args.verbose:
        if duplicates:
            print("\n--- Duplicate Groups (first 5) ---")
            for keep_id, delete_ids, sim_score in duplicates[:5]:
                keep_mem = next((m for m in memories if m.get('id') == keep_id), {})
                print(f"\nKEEP [{keep_mem.get('importance', '?')}]: {keep_mem.get('content', '')[:60]}...")
                print(f"  DELETE: {len(delete_ids)} duplicate(s) @ {sim_score:.0%} similarity")

        if garbage:
            print("\n--- Garbage (first 5) ---")
            for m in garbage[:5]:
                print(f"  [{m.get('category', '?')}] {m.get('content', '')[:50]}...")

    # Recommendations
    print("\n--- Recommendations ---")
    if dup_count > 0:
        print(f"  Run 'memory dedup' to remove {dup_count} duplicates")
    if len(garbage) > 0:
        print(f"  Run 'memory garbage' to remove {len(garbage)} garbage entries")
    if orphan_count > 50:
        print(f"  Run 'memory link-all' to create edges between related memories")


def cmd_garbage(args):
    """Find and optionally delete garbage memories."""
    if not check_helix_running():
        print("ERROR: HelixDB not running", file=sys.stderr)
        sys.exit(1)

    memories = get_all_memories()
    garbage = []

    for m in memories:
        content = m.get('content', '')
        reason = None

        if is_corrupted(content):
            reason = "corrupted"
        elif is_garbage(content):
            reason = "garbage"

        if reason:
            garbage.append((m, reason))

    if not garbage:
        print("No garbage found!")
        return

    print(f"Found {len(garbage)} garbage memories:\n")

    for m, reason in garbage[:30]:
        mid = m.get('id', '')[:8]
        cat = m.get('category', '?')
        content = m.get('content', '')[:50].replace('\n', ' ')
        print(f"  [{cat}] {mid}... {content}... ({reason})")

    if len(garbage) > 30:
        print(f"  ... and {len(garbage) - 30} more")

    if args.dry_run:
        print(f"\nDry run - {len(garbage)} would be deleted")
        return

    if not args.force:
        confirm = input(f"\nDelete {len(garbage)} garbage memories? [y/N] ")
        if confirm.lower() != 'y':
            print("Aborted")
            return

    # Delete
    deleted = 0
    errors = 0
    for m, _ in garbage:
        if delete_memory(m.get('id')):
            deleted += 1
        else:
            errors += 1

    print(f"\nDeleted: {deleted}, Errors: {errors}")


def cmd_link(args):
    """Create edge between two memories."""
    if not check_helix_running():
        print("ERROR: HelixDB not running", file=sys.stderr)
        sys.exit(1)

    from_id = args.from_id
    to_id = args.to_id
    relationship = args.relationship
    strength = args.strength

    # Support partial ID matching
    memories = get_all_memories()

    def resolve_id(partial):
        if len(partial) >= 36:
            return partial
        matches = [m for m in memories if m.get('id', '').startswith(partial)]
        if len(matches) == 1:
            return matches[0].get('id')
        elif len(matches) == 0:
            print(f"ERROR: No memory found with ID prefix: {partial}", file=sys.stderr)
            sys.exit(1)
        else:
            print(f"ERROR: Multiple memories match prefix '{partial}'", file=sys.stderr)
            sys.exit(1)

    from_id = resolve_id(from_id)
    to_id = resolve_id(to_id)

    from hooks.common import link_related_memories

    if link_related_memories(from_id, to_id, relationship, strength):
        print(f"Linked: {from_id[:8]}... --[{relationship}]--> {to_id[:8]}...")
    else:
        print("ERROR: Failed to create link", file=sys.stderr)
        sys.exit(1)


def cmd_merge(args):
    """Merge duplicate memories (keep one, delete others)."""
    if not check_helix_running():
        print("ERROR: HelixDB not running", file=sys.stderr)
        sys.exit(1)

    keep_id = args.keep_id
    delete_ids = args.delete_ids

    memories = get_all_memories()

    def resolve_id(partial):
        if len(partial) >= 36:
            return partial
        matches = [m for m in memories if m.get('id', '').startswith(partial)]
        if len(matches) == 1:
            return matches[0].get('id')
        elif len(matches) == 0:
            print(f"ERROR: No memory found with ID prefix: {partial}", file=sys.stderr)
            sys.exit(1)
        else:
            print(f"ERROR: Multiple memories match prefix '{partial}'", file=sys.stderr)
            sys.exit(1)

    keep_id = resolve_id(keep_id)
    delete_ids = [resolve_id(d) for d in delete_ids]

    # Show what will be merged
    keep_mem = next((m for m in memories if m.get('id') == keep_id), {})
    print(f"KEEP: [{keep_mem.get('category', '?')}] {keep_mem.get('content', '')[:60]}...")

    for did in delete_ids:
        del_mem = next((m for m in memories if m.get('id') == did), {})
        print(f"DELETE: [{del_mem.get('category', '?')}] {del_mem.get('content', '')[:60]}...")

    if args.dry_run:
        print(f"\nDry run - would delete {len(delete_ids)} memories")
        return

    if not args.force:
        confirm = input(f"\nMerge {len(delete_ids)} into keeper? [y/N] ")
        if confirm.lower() != 'y':
            print("Aborted")
            return

    # Create supersedes edges then delete
    from hooks.common import create_supersedes

    deleted = 0
    for did in delete_ids:
        # Create supersedes edge
        create_supersedes(keep_id, did)
        # Delete the duplicate
        if delete_memory(did):
            deleted += 1
            print(f"  Deleted: {did[:8]}...")

    print(f"\nMerged: {deleted} memories into {keep_id[:8]}...")


# ============================================================================
# INTELLIGENT MEMORY CURATION
# ============================================================================

class MemoryAnalysis:
    """Result of analyzing a single memory."""
    def __init__(self, memory_id: str):
        self.id = memory_id
        self.action = "keep"  # keep, delete, merge, link
        self.reason = ""
        self.related_ids = []  # List of (id, similarity, relationship_type)
        self.duplicate_of = None  # If duplicate, ID of the original
        self.similarity_score = 0.0
        self.quality_score = 0  # 0-10
        self.issues = []  # List of detected issues


def analyze_memory(memory: dict, all_memories: list, cache: dict) -> MemoryAnalysis:
    """
    Deeply analyze a single memory and determine what to do with it.

    Considers:
    - Content quality (length, structure, patterns)
    - Duplicate detection via hash and semantic similarity
    - Relationship mapping to other memories
    - Category-specific rules

    Returns MemoryAnalysis with recommended action and reasoning.
    """
    from hooks.common import (
        search_by_similarity,
        calculate_semantic_similarity,
        get_related_memories,
        content_hash
    )

    mid = memory.get('id', '')
    content = memory.get('content', '')
    category = memory.get('category', '')
    importance = memory.get('importance', 5)
    tags = memory.get('tags', '')

    analysis = MemoryAnalysis(mid)

    # === STEP 1: Quality Assessment ===

    # Check for corrupted content
    if is_corrupted(content):
        analysis.action = "delete"
        analysis.reason = "Corrupted content (truncated, malformed, or too short)"
        analysis.quality_score = 0
        analysis.issues.append("corrupted")
        return analysis

    # Check for garbage patterns
    if is_garbage(content):
        analysis.action = "delete"
        analysis.reason = "Low-value content (task narration, bash output, generic acknowledgment)"
        analysis.quality_score = 1
        analysis.issues.append("garbage_pattern")
        return analysis

    # Calculate base quality score
    quality = importance
    if len(content) > 100:
        quality += 1
    if len(content) > 200:
        quality += 1
    if tags and len(tags) > 3:
        quality += 1
    if category in ('preference', 'decision', 'solution'):
        quality += 1  # High-value categories
    analysis.quality_score = min(10, quality)

    # === STEP 2: Exact Duplicate Detection (via hash) ===

    c_hash = content_hash(content)
    if c_hash in cache.get('hashes', {}):
        existing_id = cache['hashes'][c_hash]
        if existing_id != mid:
            analysis.action = "merge"
            analysis.duplicate_of = existing_id
            analysis.similarity_score = 1.0
            analysis.reason = f"Exact duplicate of {existing_id[:8]}"
            return analysis
    else:
        cache.setdefault('hashes', {})[c_hash] = mid

    # === STEP 3: Semantic Similarity Search ===

    try:
        # Find semantically similar memories using vector search
        similar = search_by_similarity(content, k=8, window="full")

        for s in similar:
            sid = s.get('id', '')
            if sid == mid:
                continue

            # Calculate actual semantic similarity using embeddings
            try:
                sim_score = calculate_semantic_similarity(content, s.get('content', ''))
            except Exception as e:
                analysis.issues.append(f"similarity_calc_error: {e}")
                continue

            # Validate similarity score
            if not (0 <= sim_score <= 1):
                continue

            # High similarity (>0.90) = potential duplicate
            if sim_score >= 0.90:
                s_importance = s.get('importance', 5)
                if importance >= s_importance:
                    # This memory is better/equal, mark other as superseded
                    analysis.related_ids.append((sid, sim_score, "supersedes"))
                else:
                    # Other memory is better, this one should merge
                    analysis.action = "merge"
                    analysis.duplicate_of = sid
                    analysis.similarity_score = sim_score
                    analysis.reason = f"Duplicate of higher-importance memory {sid[:8]} ({sim_score:.0%} similar)"
                    return analysis

            # Moderate similarity (0.65-0.90) = related, should link
            elif sim_score >= 0.65:
                # Determine relationship type based on categories
                s_cat = s.get('category', '')

                if category == 'decision' and s_cat == 'solution':
                    rel_type = 'implies'
                elif category == 'problem' and s_cat == 'solution':
                    rel_type = 'leads_to'
                elif category == 'fact' and s_cat == 'preference':
                    rel_type = 'supports'
                else:
                    rel_type = 'related'

                analysis.related_ids.append((sid, sim_score, rel_type))

    except Exception as e:
        analysis.issues.append(f"similarity_search_error: {e}")

    # === STEP 4: Check Existing Edges ===

    try:
        existing_edges = get_related_memories(mid)
        has_sufficient_edges = len(existing_edges) >= 2
    except Exception:
        has_sufficient_edges = False
        analysis.issues.append("edge_check_failed")

    # === STEP 5: Determine Final Action ===

    if analysis.action == "merge":
        # Already set in duplicate detection
        pass
    elif analysis.related_ids and not has_sufficient_edges:
        analysis.action = "link"
        analysis.reason = f"Found {len(analysis.related_ids)} related memories to connect"
    elif analysis.quality_score >= 7:
        analysis.action = "keep"
        analysis.reason = "High quality memory, well-connected"
    else:
        analysis.action = "keep"
        analysis.reason = "Acceptable quality, no issues"

    return analysis


def cmd_curate(args):
    """
    Intelligently curate all memories with phased workflow.

    Phases:
    - analyze: Run full analysis, save pending actions to file
    - review: Show pending actions for Claude to review
    - apply: Execute pending actions (with optional filters)

    This non-interactive design allows background execution and
    lets Claude decide what actions to take.
    """
    from pathlib import Path
    import time
    import json as json_lib

    pending_path = Path.home() / ".cache/helix-memory/pending_curate.json"
    pending_path.parent.mkdir(parents=True, exist_ok=True)

    # === REVIEW PHASE ===
    if args.action == 'review':
        if not pending_path.exists():
            print("No pending curation. Run 'curate analyze' first.")
            sys.exit(1)

        with open(pending_path) as f:
            pending = json_lib.load(f)

        print("=" * 60)
        print("PENDING CURATION ACTIONS")
        print("=" * 60)
        print(f"Analyzed:  {pending.get('timestamp', 'unknown')}")
        print(f"Delete:    {len(pending.get('delete', []))} memories")
        print(f"Merge:     {len(pending.get('merge', []))} duplicates")
        print(f"Link:      {len(pending.get('link', []))} memories")

        if pending.get('delete'):
            print(f"\n--- Delete Candidates ---")
            for d in pending['delete'][:5]:
                print(f"  {d['id'][:8]}: {d['reason'][:50]}")
            if len(pending['delete']) > 5:
                print(f"  ... and {len(pending['delete']) - 5} more")

        if pending.get('merge'):
            print(f"\n--- Merge Candidates ---")
            for m in pending['merge'][:5]:
                print(f"  {m['id'][:8]} → {m['into'][:8]} (sim={m['similarity']:.2f})")
            if len(pending['merge']) > 5:
                print(f"  ... and {len(pending['merge']) - 5} more")

        if pending.get('link'):
            print(f"\n--- Link Candidates ---")
            print(f"  {len(pending['link'])} memories need edges")

        print(f"\nTo apply: curate apply [--links-only|--deletes-only|--merges-only]")
        return

    # === APPLY PHASE ===
    if args.action == 'apply':
        if not check_helix_running():
            print("ERROR: HelixDB not running", file=sys.stderr)
            sys.exit(1)

        if not pending_path.exists():
            print("No pending curation. Run 'curate analyze' first.")
            sys.exit(1)

        with open(pending_path) as f:
            pending = json_lib.load(f)

        applied = {'deleted': 0, 'merged': 0, 'linked': 0}

        # Apply deletes
        if pending.get('delete') and not args.links_only and not args.merges_only:
            print(f"Deleting {len(pending['delete'])} garbage memories...")
            for d in pending['delete']:
                try:
                    if delete_memory(d['id']):
                        applied['deleted'] += 1
                except Exception:
                    pass

        # Apply merges (delete duplicates, keeping the referenced 'into')
        if pending.get('merge') and not args.links_only and not args.deletes_only:
            print(f"Merging {len(pending['merge'])} duplicates...")
            for m in pending['merge']:
                try:
                    if delete_memory(m['id']):
                        applied['merged'] += 1
                except Exception:
                    pass

        # Apply links
        if pending.get('link') and not args.deletes_only and not args.merges_only:
            from hooks.common import link_related_memories
            total_links = len(pending['link'])
            print(f"Linking {total_links} memories...", flush=True)
            for i, lnk in enumerate(pending['link']):
                for related_id, sim_score, rel_type in lnk.get('related', [])[:3]:
                    try:
                        strength = int(sim_score * 10)
                        if link_related_memories(lnk['id'], related_id, rel_type, strength):
                            applied['linked'] += 1
                        if link_related_memories(related_id, lnk['id'], rel_type, strength):
                            applied['linked'] += 1
                    except Exception:
                        pass
                # Progress every 100 memories
                if (i + 1) % 100 == 0:
                    print(f"  [{i+1}/{total_links}] {applied['linked']} edges created", flush=True)

        print(f"\nApplied: {applied['deleted']} deleted, {applied['merged']} merged, {applied['linked']} edges")

        # Clear pending if all applied
        if not args.links_only and not args.deletes_only and not args.merges_only:
            pending_path.unlink(missing_ok=True)
            print("Pending actions cleared.")
        return

    # === ANALYZE PHASE (default) ===
    if not check_helix_running():
        print("ERROR: HelixDB not running", file=sys.stderr)
        sys.exit(1)

    memories = get_all_memories()
    total = len(memories)
    cache = {'hashes': {}}
    results = {'keep': [], 'delete': [], 'merge': [], 'link': []}

    print(f"Analyzing {total} memories...", flush=True)
    print("=" * 60, flush=True)

    start_time = time.time()
    error_count = 0

    for i, m in enumerate(memories):
        try:
            analysis = analyze_memory(m, memories, cache)
            results[analysis.action].append(analysis)

            if analysis.action != 'keep':
                symbols = {'delete': '✗', 'merge': '⊕', 'link': '⟷'}
                print(f"[{i+1:4d}/{total}] {symbols[analysis.action]} {m.get('id', '')[:8]} {analysis.reason[:45]}", flush=True)
            elif args.verbose:
                print(f"[{i+1:4d}/{total}] ✓ {m.get('id', '')[:8]} quality={analysis.quality_score}", flush=True)
        except Exception as e:
            error_count += 1
            if args.verbose:
                print(f"[{i+1:4d}/{total}] ERROR: {e}", file=sys.stderr, flush=True)

        if (i + 1) % 25 == 0:
            time.sleep(0.3)

    elapsed = time.time() - start_time

    # Summary
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Analyzed:  {total} memories in {elapsed:.1f}s")
    print(f"Keep:      {len(results['keep']):4d}")
    print(f"Delete:    {len(results['delete']):4d}")
    print(f"Merge:     {len(results['merge']):4d}")
    print(f"Link:      {len(results['link']):4d}")
    print(f"Errors:    {error_count:4d}")

    # Save pending actions
    pending = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_analyzed': total,
        'elapsed_seconds': elapsed,
        'delete': [{'id': a.id, 'reason': a.reason, 'issues': a.issues} for a in results['delete']],
        'merge': [{'id': a.id, 'into': a.duplicate_of, 'similarity': a.similarity_score} for a in results['merge']],
        'link': [{'id': a.id, 'related': [(r[0], r[1], r[2]) for r in a.related_ids]} for a in results['link']],
    }

    with open(pending_path, 'w') as f:
        json_lib.dump(pending, f, indent=2)

    print(f"\nPending actions saved: {pending_path}")
    print(f"Next: 'curate review' to inspect, 'curate apply' to execute")


def cmd_projects(args):
    """List projects worked on based on memory tags, cross-referenced with p --list."""
    import subprocess
    from collections import Counter

    if not check_helix_running():
        print("ERROR: HelixDB not running", file=sys.stderr)
        sys.exit(1)

    # Get known projects from p --list
    try:
        result = subprocess.run(
            ['python3', os.path.expanduser('~/Tools/p'), '--list'],
            capture_output=True, text=True, timeout=5
        )
        known_projects = set()
        for line in result.stdout.strip().split('\n'):
            if ':' in line:
                project_name = line.split(':')[0].strip().lower()
                known_projects.add(project_name)
    except Exception as e:
        print(f"ERROR: Could not get project list: {e}", file=sys.stderr)
        sys.exit(1)

    # Parse date filters
    since_date = None
    exact_date = None

    if args.since:
        since_date = parse_date_arg(args.since)
        if not since_date:
            print(f"ERROR: Invalid date format: {args.since}", file=sys.stderr)
            sys.exit(1)
    elif args.date:
        exact_date = parse_date_arg(args.date)
        if not exact_date:
            print(f"ERROR: Invalid date format: {args.date}", file=sys.stderr)
            sys.exit(1)
    else:
        # Default to yesterday
        since_date = (datetime.now() - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)

    memories = get_all_memories()
    project_counts = Counter()
    project_memories = {}  # project -> list of memory snippets

    for m in memories:
        created = parse_memory_timestamp(m)
        if not created:
            continue

        # Check date filter
        if since_date and created < since_date:
            continue
        if exact_date:
            next_day = exact_date + timedelta(days=1)
            if not (exact_date <= created < next_day):
                continue

        # Check tags against known projects
        tags = [t.strip().lower() for t in m.get('tags', '').split(',') if t.strip()]
        for tag in tags:
            if tag in known_projects:
                project_counts[tag] += 1
                if tag not in project_memories:
                    project_memories[tag] = []
                if len(project_memories[tag]) < 3:  # Keep top 3 snippets
                    project_memories[tag].append(m.get('content', '')[:60])

    if not project_counts:
        date_desc = f"since {since_date.strftime('%Y-%m-%d')}" if since_date else f"on {exact_date.strftime('%Y-%m-%d')}"
        print(f"No project activity found {date_desc}")
        return

    # Display results
    date_desc = f"since {since_date.strftime('%Y-%m-%d')}" if since_date else f"on {exact_date.strftime('%Y-%m-%d')}"
    print(f"Projects worked on {date_desc}:\n")

    for project, count in project_counts.most_common():
        print(f"  {project}: {count} memories")
        if args.verbose and project in project_memories:
            for snippet in project_memories[project]:
                print(f"    - {snippet}...")
            print()


def cmd_show(args):
    """Show memory details and its edges."""
    if not check_helix_running():
        print("ERROR: HelixDB not running", file=sys.stderr)
        sys.exit(1)

    mem_id = args.id
    memories = get_all_memories()

    # Find by prefix
    matches = [m for m in memories if m.get('id', '').startswith(mem_id)]
    if not matches:
        print(f"No memory found with ID prefix: {mem_id}")
        sys.exit(1)

    mem = matches[0]
    full_id = mem.get('id')

    # Print memory details
    print("=" * 60)
    print(f"[{mem.get('category', '?').upper()}-{mem.get('importance', 0)}] {full_id[:8]}")
    print("=" * 60)
    print(f"Content:    {mem.get('content', '')}")
    print(f"Category:   {mem.get('category', '?')}")
    print(f"Importance: {mem.get('importance', 0)}")
    print(f"Tags:       {mem.get('tags', '')}")
    print(f"Created:    {mem.get('created_at', '?')}")
    print(f"Full ID:    {full_id}")

    # Get edges
    try:
        import requests
        r = requests.post(f"{HELIX_URL}/GetRelatedMemories", json={"memory_id": full_id}, timeout=5)
        data = r.json()
        related = data.get('related', [])

        # Dedupe by ID
        seen_ids = set()
        unique_related = []
        for rel in related:
            rid = rel.get('id')
            if rid and rid not in seen_ids:
                seen_ids.add(rid)
                unique_related.append(rel)

        if unique_related:
            print(f"\n--- Related ({len(unique_related)} edges) ---")
            for rel in unique_related[:10]:
                print(f"  → [{rel.get('category', '?').upper()}-{rel.get('importance', 0)}] {rel.get('id', '?')[:8]}")
                print(f"    {rel.get('content', '')[:60]}...")
            if len(unique_related) > 10:
                print(f"  ... and {len(unique_related) - 10} more")
        else:
            print(f"\n--- No edges (orphan) ---")
    except Exception as e:
        print(f"\n--- Could not fetch edges: {e} ---")


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
    list_parser.add_argument('--since', help='Show memories created after date (yesterday, 2026-01-10, "3 days ago")')
    list_parser.add_argument('--date', help='Show memories created on exact date (2026-01-10)')
    list_parser.set_defaults(func=cmd_list)

    # by-tag command
    tag_parser = subparsers.add_parser('by-tag', help='Get memories by tag')
    tag_parser.add_argument('tag', help='Tag to search for')
    tag_parser.set_defaults(func=cmd_by_tag)

    # delete command
    delete_parser = subparsers.add_parser('delete', help='Delete a memory')
    delete_parser.add_argument('id', help='Memory ID to delete')
    delete_parser.set_defaults(func=cmd_delete)

    # show command - display memory details and edges
    show_parser = subparsers.add_parser('show', help='Show memory details and edges')
    show_parser.add_argument('id', help='Memory ID (prefix OK)')
    show_parser.set_defaults(func=cmd_show)

    # status command
    status_parser = subparsers.add_parser('status', help='Check HelixDB status')
    status_parser.set_defaults(func=cmd_status)

    # projects command - list projects worked on
    projects_parser = subparsers.add_parser('projects', help='List projects worked on (cross-refs with p --list)')
    projects_parser.add_argument('--since', help='Show projects since DATE (default: yesterday)')
    projects_parser.add_argument('--date', help='Show projects on exact DATE')
    projects_parser.add_argument('--verbose', '-v', action='store_true', help='Show memory snippets')
    projects_parser.set_defaults(func=cmd_projects)

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

    # health command - memory health report
    health_parser = subparsers.add_parser('health', help='Show memory health report')
    health_parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed breakdown')
    health_parser.add_argument('--thorough', '-t', action='store_true', help='Use vector similarity (slower but more accurate)')
    health_parser.set_defaults(func=cmd_health)

    # garbage command - find/delete garbage
    garbage_parser = subparsers.add_parser('garbage', help='Find and delete garbage memories')
    garbage_parser.add_argument('--dry-run', '-n', action='store_true', help='Show garbage without deleting')
    garbage_parser.add_argument('--force', '-f', action='store_true', help='Delete without confirmation')
    garbage_parser.set_defaults(func=cmd_garbage)

    # link command - create edge between two memories
    link_cmd = subparsers.add_parser('link', help='Create edge between two memories')
    link_cmd.add_argument('from_id', help='Source memory ID (prefix OK)')
    link_cmd.add_argument('to_id', help='Target memory ID (prefix OK)')
    link_cmd.add_argument('--relationship', '-r', default='related',
                          choices=['related', 'supersedes', 'implies', 'contradicts', 'leads_to', 'supports'],
                          help='Relationship type (default: related)')
    link_cmd.add_argument('--strength', '-s', type=int, default=5, help='Edge strength 1-10')
    link_cmd.set_defaults(func=cmd_link)

    # merge command - merge duplicates
    merge_parser = subparsers.add_parser('merge', help='Merge duplicate memories (keep one, delete others)')
    merge_parser.add_argument('keep_id', help='Memory ID to keep (prefix OK)')
    merge_parser.add_argument('delete_ids', nargs='+', help='Memory IDs to delete (prefix OK)')
    merge_parser.add_argument('--dry-run', '-n', action='store_true', help='Show what would be merged')
    merge_parser.add_argument('--force', '-f', action='store_true', help='Merge without confirmation')
    merge_parser.set_defaults(func=cmd_merge)

    # curate command - intelligent full curation (phased: analyze → review → apply)
    curate_parser = subparsers.add_parser('curate', help='Intelligent memory curation with analysis')
    curate_parser.add_argument('action', nargs='?', default='analyze',
                               choices=['analyze', 'review', 'apply'],
                               help='Phase: analyze (default), review, or apply')
    curate_parser.add_argument('--verbose', '-v', action='store_true', help='Show all analysis results')
    curate_parser.add_argument('--links-only', action='store_true', help='Apply only link operations')
    curate_parser.add_argument('--deletes-only', action='store_true', help='Apply only delete operations')
    curate_parser.add_argument('--merges-only', action='store_true', help='Apply only merge operations')
    curate_parser.set_defaults(func=cmd_curate)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)

if __name__ == '__main__':
    main()
