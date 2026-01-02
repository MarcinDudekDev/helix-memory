#!/usr/bin/env python3
"""
UserPromptSubmit Hook: Load Relevant Memories

Executes BEFORE Claude processes user's message. Searches HelixDB for
relevant memories based on the prompt and injects them as additional context.

This allows Claude to automatically have access to:
- User preferences
- Project context
- Previous decisions
- Related facts

Without the user having to remind Claude or explicitly invoke memory.
"""

import sys
import json
from pathlib import Path

# Add hooks directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from common import (
    read_hook_input,
    ensure_helix_running,
    format_hook_output,
    print_hook_output
)

def extract_keywords(prompt: str) -> list:
    """
    Extract keywords from user prompt for memory retrieval.

    Simple keyword extraction - looks for important words to guide
    memory search.

    Args:
        prompt: User's message text

    Returns:
        List of keywords
    """
    # Common words to ignore
    stop_words = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
        "be", "have", "has", "had", "do", "does", "did", "will", "would", "should",
        "could", "can", "may", "might", "must", "i", "you", "he", "she", "it",
        "we", "they", "me", "him", "her", "us", "them", "my", "your", "his",
        "her", "its", "our", "their", "this", "that", "these", "those"
    }

    # Extract words
    words = prompt.lower().split()
    keywords = [w for w in words if w not in stop_words and len(w) > 2]

    return keywords[:10]  # Limit to top 10

def bm25_score(query_terms: list, doc_terms: list, avg_doc_len: float, doc_freq: dict, num_docs: int, k1: float = 1.5, b: float = 0.75) -> float:
    """
    Calculate BM25 score for a document.

    Args:
        query_terms: List of query terms
        doc_terms: List of terms in document
        avg_doc_len: Average document length
        doc_freq: Dict mapping term -> number of docs containing it
        num_docs: Total number of documents
        k1: Term frequency saturation parameter
        b: Length normalization parameter

    Returns:
        BM25 score
    """
    import math

    score = 0.0
    doc_len = len(doc_terms)

    for term in query_terms:
        if term not in doc_terms:
            continue

        # Term frequency in this doc
        tf = doc_terms.count(term)

        # Inverse document frequency
        df = doc_freq.get(term, 0)
        if df == 0:
            continue
        idf = math.log((num_docs - df + 0.5) / (df + 0.5) + 1)

        # BM25 formula
        tf_component = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avg_doc_len))
        score += idf * tf_component

    return score


def search_memories_by_keywords(keywords: list) -> list:
    """
    Search HelixDB for memories using BM25 ranking.

    BM25 provides better ranking than simple keyword matching by considering:
    - Term frequency (how often term appears)
    - Inverse document frequency (rarer terms score higher)
    - Document length normalization

    Args:
        keywords: List of search keywords

    Returns:
        List of matching memories (sorted by BM25 + importance)
    """
    from common import get_all_memories

    memories = get_all_memories()
    if not memories:
        return []

    # Tokenize all documents
    def tokenize(text):
        return [w.lower() for w in text.split() if len(w) > 2]

    docs = []
    for m in memories:
        content = m.get("content", "")
        tags = m.get("tags", "").replace(",", " ")
        terms = tokenize(content + " " + tags)
        docs.append((m, terms))

    # Calculate document frequencies
    doc_freq = {}
    for _, terms in docs:
        seen = set(terms)
        for term in seen:
            doc_freq[term] = doc_freq.get(term, 0) + 1

    # Calculate average document length
    total_len = sum(len(terms) for _, terms in docs)
    avg_doc_len = total_len / len(docs) if docs else 1

    # Score each document
    query_terms = [kw.lower() for kw in keywords]
    scored = []

    for memory, doc_terms in docs:
        bm25 = bm25_score(query_terms, doc_terms, avg_doc_len, doc_freq, len(docs))

        if bm25 > 0:
            importance = memory.get("importance", 5)
            # Combined score: BM25 * importance weight
            final_score = bm25 * (1 + importance / 10)
            scored.append((final_score, memory))

    # Sort by score descending, take top 5
    scored.sort(key=lambda x: x[0], reverse=True)
    results = [m for _, m in scored[:5]]

    # Record access for retrieved memories
    if results:
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from memory_lifecycle import record_access
            for m in results:
                record_access(m.get("id", ""))
        except:
            pass

    return results

def format_memories_as_context(memories: list) -> str:
    """
    Format memories as additional context string.

    Args:
        memories: List of memory dicts

    Returns:
        Formatted context string
    """
    if not memories:
        return ""

    context_parts = ["Relevant memories from previous sessions:"]

    for memory in memories:
        category = memory.get("category", "unknown")
        importance = memory.get("importance", 0)
        content = memory.get("content", "")

        context_parts.append(f"\n[{category.upper()} - importance: {importance}] {content}")

    return "\n".join(context_parts)

def check_search_command(prompt: str) -> str:
    """Check if user is explicitly searching memories."""
    prompt_lower = prompt.lower()

    search_triggers = [
        "search memories:", "search memory:",
        "find in memory:", "recall:",
        "what do you remember about",
        "check memories for"
    ]

    for trigger in search_triggers:
        if trigger in prompt_lower:
            # Extract search query after trigger
            idx = prompt_lower.find(trigger)
            query = prompt[idx + len(trigger):].strip()
            return query

    return ""


def get_project_context(cwd: str) -> list:
    """Get memories specific to current working directory."""
    from common import get_all_memories

    memories = get_all_memories()
    if not memories or not cwd:
        return []

    # Find memories tagged with this project path
    project_memories = []
    cwd_parts = cwd.lower().split('/')

    for m in memories:
        tags = m.get('tags', '').lower()
        content = m.get('content', '').lower()

        # Check if memory is related to this project
        for part in cwd_parts[-3:]:  # Check last 3 path components
            if len(part) > 3 and (part in tags or part in content):
                project_memories.append(m)
                break

    # Sort by importance
    project_memories.sort(key=lambda x: x.get('importance', 0), reverse=True)
    return project_memories[:3]


def get_high_importance_memories(min_importance: int = 8) -> list:
    """
    Get high-importance memories that should always be loaded.

    These are critical preferences and facts that should guide
    every interaction regardless of prompt content.
    """
    from common import get_all_memories

    memories = get_all_memories()
    if not memories:
        return []

    # Filter by importance
    high_imp = [m for m in memories if m.get('importance', 0) >= min_importance]

    # Sort by importance (highest first), then by category priority
    category_priority = {'preference': 0, 'decision': 1, 'fact': 2, 'context': 3, 'task': 4}
    high_imp.sort(key=lambda x: (-x.get('importance', 0), category_priority.get(x.get('category', ''), 5)))

    return high_imp[:5]  # Return top 5 most important


def should_load_memories(prompt: str) -> tuple[bool, str]:
    """
    LAZY LOADING: Determine if we should load memory context.

    Returns (should_load, reason) tuple.

    Triggers on:
    - Explicit search commands (recall:, remember:, etc.)
    - Trigger words (remember, recall, how do I, etc.)
    - Question patterns (what, how, where, why, which)
    - Problem/error keywords
    - Technical/project terms
    """
    prompt_lower = prompt.lower()

    # Explicit search triggers - highest priority
    search_triggers = [
        "search memories:", "search memory:",
        "find in memory:", "recall:",
        "what do you remember about",
        "check memories for", "/recall"
    ]
    for trigger in search_triggers:
        if trigger in prompt_lower:
            return True, "explicit_search"

    # Implicit recall triggers
    recall_words = [
        "remember", "recall", "pamiętasz", "wspomnienie",
        "czy wiesz", "do you know", "have we",
        "last time", "previously", "before", "earlier",
        "we discussed", "you mentioned", "as usual"
    ]
    for word in recall_words:
        if word in prompt_lower:
            return True, "recall_trigger"

    # Question patterns - likely need context
    question_patterns = [
        "how do i", "how to", "how can i", "how should",
        "what is the", "what's the", "what are",
        "where is", "where are", "where do",
        "why does", "why is", "why do",
        "which one", "which should",
        "can you", "could you", "would you",
        "jak ", "gdzie ", "dlaczego ", "co to",  # Polish
    ]
    for pattern in question_patterns:
        if pattern in prompt_lower:
            return True, "question_pattern"

    # Problem/error triggers - might need solutions
    problem_words = [
        "error", "bug", "problem", "issue", "broken",
        "doesn't work", "nie działa", "failed", "błąd",
        "not working", "fix", "debug", "crash", "exception"
    ]
    for word in problem_words:
        if word in prompt_lower:
            return True, "problem_search"

    # Technical terms that often have stored context
    tech_triggers = [
        "deploy", "mikrus", "wp-test", "wordpress", "docker",
        "datastar", "fastapi", "credentials", "password", "login",
        "api key", "config", "setup", "install"
    ]
    for term in tech_triggers:
        if term in prompt_lower:
            return True, "tech_context"

    # Default: DON'T load memories automatically
    return False, "no_trigger"


def main():
    """Main hook execution - LAZY LOADING VERSION.

    Key optimization: Only loads memories when triggered, not on every prompt.
    This reduces token overhead from ~500-1K per prompt to ~0 when not needed.
    """
    # Read hook input
    hook_data = read_hook_input()
    prompt = hook_data.get("prompt", "")
    cwd = hook_data.get("cwd", "")

    if not prompt:
        output = format_hook_output(suppress=True)
        print_hook_output(output)
        sys.exit(0)

    # Check if HelixDB is running
    if not ensure_helix_running():
        output = format_hook_output(suppress=True)
        print_hook_output(output)
        sys.exit(0)

    # LAZY LOADING CHECK
    should_load, trigger_reason = should_load_memories(prompt)

    if not should_load:
        # No trigger detected - skip memory loading entirely
        output = format_hook_output(suppress=True)
        print_hook_output(output)
        sys.exit(0)

    # Triggered! Load relevant memories
    memories = []
    seen_ids = set()

    # For explicit search, extract query after trigger
    search_query = check_search_command(prompt)
    if search_query:
        keywords = extract_keywords(search_query)
    else:
        keywords = extract_keywords(prompt)

    # Search by keywords (BM25 ranking)
    if keywords:
        keyword_memories = search_memories_by_keywords(keywords)
        for m in keyword_memories:
            if m.get('id') not in seen_ids:
                memories.append(m)
                seen_ids.add(m.get('id'))

    # For problem searches, also look for solutions
    if trigger_reason == "problem_search":
        from common import get_all_memories
        all_mems = get_all_memories()
        solutions = [m for m in all_mems if m.get('category') == 'solution']
        # Add top 3 relevant solutions
        for s in solutions[:3]:
            if s.get('id') not in seen_ids:
                memories.append(s)
                seen_ids.add(s.get('id'))

    # Add project context if in known project directory
    if cwd and any(p in cwd for p in ['/Sites/', '/Tools/', '/Projects/', 'PycharmProjects', '.wp-test']):
        project_memories = get_project_context(cwd)
        for pm in project_memories:
            if pm.get('id') not in seen_ids:
                memories.append(pm)
                seen_ids.add(pm.get('id'))

    if not memories:
        output = format_hook_output(suppress=True)
        print_hook_output(output)
        sys.exit(0)

    # Format and return context
    additional_context = format_memories_as_context(memories)
    output = {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": additional_context
        },
        "suppressOutput": True
    }
    print_hook_output(output)
    sys.exit(0)

if __name__ == "__main__":
    main()
