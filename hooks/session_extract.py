#!/usr/bin/env python3
"""
Stop hook: Extract memories from new session exchanges.
Called by Claude Code Stop event with transcript_path in stdin JSON.

Debugging:
    Log file: ~/.cache/helix/session_extract.log
    Status:   ~/.cache/helix/last_extract_status
    Test:     echo '{"transcript_path": "...", "cwd": "..."}' | python3 session_extract.py --test
"""

import json
import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path

# === RECURSION GUARD ===
if os.environ.get("HELIX_HOOK_RUNNING"):
    sys.exit(0)
os.environ["HELIX_HOOK_RUNNING"] = "1"

sys.path.insert(0, str(Path(__file__).parent))
from common import (store_memory, generate_embedding, store_memory_embedding,
                    check_helix_running, ensure_helix_running, detect_project)
from simple_scribe import extract_memories

TRANSCRIPT_EXTRACT = Path.home() / "Tools" / "transcript-extract.py"
LOG_FILE = Path.home() / ".cache" / "helix" / "session_extract.log"
STATUS_FILE = Path.home() / ".cache" / "helix" / "last_extract_status"
MAX_EXCHANGES = 5  # Limit per run to avoid rate limits/timeouts


def log(level: str, msg: str):
    """Append to log file for debugging."""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(f"{datetime.now().isoformat()} [{level}] {msg}\n")


def write_status(success: bool, memories: int = 0, error: str = ""):
    """Write status file for health monitoring."""
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATUS_FILE.write_text(json.dumps({
        "success": success,
        "memories_stored": memories,
        "error": error,
        "timestamp": datetime.now().isoformat()
    }))


def main():
    test_mode = "--test" in sys.argv
    stored_count = 0

    # 1. Read stdin JSON from Claude Code
    try:
        stdin_data = json.loads(sys.stdin.read())
    except json.JSONDecodeError as e:
        log("ERROR", f"Invalid JSON from stdin: {e}")
        write_status(False, error=f"JSON parse error: {e}")
        return

    transcript_path = stdin_data.get("transcript_path")
    cwd = stdin_data.get("cwd", "")

    if not transcript_path:
        log("INFO", "No transcript_path in stdin, skipping")
        write_status(True, error="no transcript_path")
        return

    if not Path(transcript_path).exists():
        log("WARN", f"Transcript not found: {transcript_path}")
        write_status(False, error=f"transcript not found: {transcript_path}")
        return

    log("INFO", f"Processing transcript: {transcript_path}")

    # 2. Detect project from working directory
    project = detect_project(cwd) or ""
    log("DEBUG", f"Detected project: {project or '(none)'}")

    # 3. Extract NEW exchanges only (incremental via offset tracking)
    result = subprocess.run(
        ["python3", str(TRANSCRIPT_EXTRACT), "-f", "json", transcript_path],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        log("ERROR", f"transcript-extract failed: {result.stderr}")
        write_status(False, error=f"transcript-extract: {result.stderr[:200]}")
        return

    try:
        exchanges = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        log("ERROR", f"Invalid JSON from transcript-extract: {e}")
        write_status(False, error=f"transcript output parse: {e}")
        return

    if not exchanges:
        log("INFO", "No new exchanges to process")
        write_status(True, memories=0)
        return

    log("INFO", f"Found {len(exchanges)} new exchange(s)")

    # Limit exchanges to process (avoid rate limits/timeouts)
    if len(exchanges) > MAX_EXCHANGES:
        log("INFO", f"Limiting to last {MAX_EXCHANGES} exchanges")
        exchanges = exchanges[-MAX_EXCHANGES:]

    # 4. Ensure HelixDB is running
    if not check_helix_running():
        log("WARN", "HelixDB not running, attempting start...")
        if not ensure_helix_running():
            log("ERROR", "Failed to start HelixDB")
            write_status(False, error="HelixDB unavailable")
            return

    # 5. For each exchange, run LLM extraction
    for i, exchange in enumerate(exchanges):
        user_msg = exchange.get("user", "")
        asst_msg = exchange.get("assistant", "")

        if not user_msg or len(user_msg) < 10:
            continue
        if not asst_msg or len(asst_msg) < 50:
            continue

        # 6. Extract memories via LLM (Ollama or Gemini - NO Anthropic)
        log("DEBUG", f"Extracting memories from exchange {i+1}/{len(exchanges)}")
        memories = extract_memories(user_msg, asst_msg, project)

        if not memories:
            log("DEBUG", f"No memories extracted from exchange {i+1}")
            continue

        # 7. Store each memory with embedding
        # Flatten nested arrays from LLM (e.g., [[{...}]] or [[[{...}]]])
        while memories and isinstance(memories, list) and len(memories) > 0 and isinstance(memories[0], list):
            memories = memories[0]

        for mem in memories:
            # Skip if mem is not a dict (malformed LLM output)
            if not isinstance(mem, dict):
                log("WARN", f"Skipping non-dict memory: {type(mem)}")
                continue
            content = mem.get("content", "")
            if not content or len(content) < 15:
                continue

            mem_id = store_memory(
                content=content,
                category=mem.get("category", "fact"),
                importance=min(10, max(1, mem.get("importance", 5))),
                tags=mem.get("tags", project),
                source="session_extract"
            )

            if mem_id:
                vector, model = generate_embedding(content)
                if vector:
                    store_memory_embedding(mem_id, vector, content, model)
                stored_count += 1
                log("INFO", f"Stored: [{mem.get('category')}] {content[:60]}...")

    log("INFO", f"=== Session extract complete: {stored_count} memories stored ===")
    write_status(True, memories=stored_count)

    if test_mode:
        print(f"TEST MODE: {stored_count} memories stored")
        print(f"Log file: {LOG_FILE}")


if __name__ == "__main__":
    main()
