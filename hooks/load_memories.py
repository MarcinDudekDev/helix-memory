#!/usr/bin/env python3
"""
UserPromptSubmit Hook: Tattoo Injection

Injects a persistent "tattoo" reminder on every prompt - tools and rules
that Claude should always have visible, like Lester's tattoos in Memento.

Memory loading is now explicit via /recall command only.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import read_hook_input, print_hook_output


def main():
    """Inject tattoo on every prompt."""
    read_hook_input()  # consume stdin even if unused

    # TATTOO: Always-visible reminders (Memento-style)
    tattoo = "TOOLS (--help): p --list | op | dev-browser | wp-test | memorize | recall | t --list (all tools) → direct, NEVER interactive, servers need nohup & | /recall if lost | REMEMBER TO MEMORIZE 💾!"

    output = {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": tattoo
        },
        "suppressOutput": True
    }
    print_hook_output(output)
    sys.exit(0)


if __name__ == "__main__":
    main()
