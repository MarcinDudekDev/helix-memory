#!/usr/bin/env python3
"""
Simple Scribe - Minimal Memory Evaluator

Takes user message + assistant response, asks LLM to extract important memories.
Uses Ollama (local) or Gemini (free) - NO Anthropic/Haiku.
Fire and forget - no complex parsing, no hash tracking.

Usage:
    python3 simple_scribe.py <user_msg_file> <assistant_msg_file> [--project NAME]
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import store_memory, check_helix_running, ensure_helix_running, llm_generate, extract_json_array


def extract_memories(user_msg: str, assistant_msg: str, project: str) -> list:
    """Use LLM (Ollama or Gemini) to extract important memories from exchange."""

    prompt = f'''Analyze this conversation exchange and extract ALL important memories worth storing long-term.

RULES:
1. Output in ENGLISH only (regardless of input language)
2. Store HIGH-LEVEL SUMMARIES - no code snippets, no raw outputs
3. Focus on: decisions made, problems solved, user preferences, important facts learned
4. Skip: routine operations, file reads, trivial exchanges

Categories:
- preference: User preferences, workflow choices (importance 8-10)
- decision: Technical/architectural decisions (importance 7-9)
- solution: Bug fixes, workarounds, completed tasks (importance 6-8)
- fact: Learned information about tools, APIs, systems (importance 5-7)

USER MESSAGE:
{user_msg[:1500]}

ASSISTANT RESPONSE:
{assistant_msg[:3000]}

Return ONLY a valid JSON array. Extract all genuinely important memories (not limited to 2).
Format: [{{"content": "...", "category": "...", "importance": N, "tags": "{project}"}}]
If nothing important, return: []'''

    output, provider = llm_generate(prompt, timeout=20)
    if output:
        print(f"Using LLM: {provider}", file=sys.stderr)
        return extract_json_array(output)

    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("user_file", help="File with user message")
    parser.add_argument("assistant_file", help="File with assistant message")
    parser.add_argument("--project", default="", help="Project name")
    args = parser.parse_args()
    
    # Read messages from temp files
    try:
        user_msg = Path(args.user_file).read_text()
        assistant_msg = Path(args.assistant_file).read_text()
    except Exception as e:
        print(f"Error reading files: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Cleanup temp files
    Path(args.user_file).unlink(missing_ok=True)
    Path(args.assistant_file).unlink(missing_ok=True)
    
    # Skip trivial exchanges
    if len(user_msg) < 10 or len(assistant_msg) < 50:
        sys.exit(0)
    
    # Ensure HelixDB running
    if not check_helix_running():
        if not ensure_helix_running():
            sys.exit(1)
    
    # Extract and store
    memories = extract_memories(user_msg, assistant_msg, args.project)

    # Flatten nested arrays from LLM
    while memories and isinstance(memories, list) and len(memories) > 0 and isinstance(memories[0], list):
        memories = memories[0]

    stored = 0
    for mem in memories:
        # Skip if mem is not a dict
        if not isinstance(mem, dict):
            continue
        content = mem.get("content", "")
        if content and len(content) > 15:
            result = store_memory(
                content,
                mem.get("category", "fact"),
                min(10, max(1, mem.get("importance", 5))),
                mem.get("tags", args.project),
                "scribe"
            )
            if result:
                stored += 1
                print(f"+ [{mem.get('category')}] {content[:60]}...")
    
    if stored:
        print(f"\n=== Stored {stored} memories ===")


if __name__ == "__main__":
    main()
