#!/usr/bin/env python3
"""
Stop hook: Witty narration of what just happened.
Reads transcript, sends to small model, speaks result.

Enable per-project: touch ~/.narrator-<projectname>
"""

import sys
import json
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import llm_generate

PROMPT = """You are a witty narrator for a coding session. In ONE SHORT sentence (max 15 words),
comment on what the assistant just did. Be brief, clever, slightly sarcastic. No quotes.

What happened: {summary}

Your narration:"""


def should_narrate(project: str) -> bool:
    """Check if narration enabled for this project."""
    if not project:
        return False
    flag = Path.home() / f".narrator-{project.lower()}"
    return flag.exists()


def speak(text: str, rate: int = 185):
    """TTS using macOS say."""
    subprocess.Popen(["say", "-v", "Daniel", "-r", str(rate), text])


def summarize_exchange(assistant_msg: str) -> str:
    """Extract key action from assistant's response."""
    # Take first 200 chars or first paragraph
    lines = assistant_msg.strip().split('\n')
    summary = lines[0][:200] if lines else ""
    return summary


def main():
    try:
        data = json.load(sys.stdin)
    except:
        sys.exit(0)

    # Get project from cwd
    project = data.get("cwd", "").split("/")[-1]

    if not should_narrate(project):
        sys.exit(0)

    # Get last assistant message
    assistant_msg = ""
    for msg in reversed(data.get("messages", [])):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            if isinstance(content, list):
                assistant_msg = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
            else:
                assistant_msg = str(content)
            break

    if not assistant_msg:
        sys.exit(0)

    # Generate narration
    summary = summarize_exchange(assistant_msg)
    prompt = PROMPT.format(summary=summary)

    narration, _ = llm_generate(prompt, timeout=5)
    if narration:
        narration = narration.strip().strip('"').split('\n')[0]
        speak(narration)
        print(f"[narrator] {narration}", file=sys.stderr)


if __name__ == "__main__":
    main()
