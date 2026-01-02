#!/usr/bin/env python3
"""
Daily Devlog Generator

Creates public-safe daily devlogs combining:
- Memory data from helix-memory (sanitized)
- AI-generated narrative (Claude Haiku)

Outputs to ~/Documents/Praca/_DZIENNIK_/devlog/ as Markdown
"""

import sys
import subprocess
import re
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent / "hooks"))

from common import check_helix_running, get_all_memories
from sanitizer import Sanitizer, sanitize_memory

# Configuration
DEVLOG_DIR = Path.home() / "Documents" / "Praca" / "_DZIENNIK_" / "devlog"


def get_known_projects() -> list:
    """Get known projects from p tool."""
    try:
        result = subprocess.run(
            [str(Path.home() / 'Tools' / 'p'), '--list'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            projects = []
            for line in result.stdout.split('\n'):
                if ':' in line:
                    name = line.split(':')[0].strip()
                    if name and name != 'cminds':
                        projects.append(name)
            return projects
    except:
        pass
    return []


def get_recent_memories(days: int = 1, target_date: str = None) -> list:
    """Get memories from a specific date or last N days.

    Args:
        days: Days to look back (default: 1)
        target_date: If specified (YYYY-MM-DD), get memories from that specific date
    """
    memories = get_all_memories()

    if target_date:
        # Get memories from specific date
        target = datetime.strptime(target_date, '%Y-%m-%d')
        start = target.replace(hour=0, minute=0, second=0)
        end = target.replace(hour=23, minute=59, second=59)
    else:
        # Get memories from last N days
        end = datetime.now()
        start = end - timedelta(days=days)

    recent = []
    for m in memories:
        created = m.get('created_at', '')
        if created:
            try:
                mem_date = datetime.fromisoformat(created.replace('Z', '+00:00'))
                mem_date = mem_date.replace(tzinfo=None)
                if start <= mem_date <= end:
                    recent.append(m)
            except:
                pass  # Skip unparseable dates

    return recent


def categorize_memories(memories: list) -> dict:
    """Group memories by category."""
    by_category = {}
    for m in memories:
        cat = m.get('category', 'other')
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(m)
    return by_category


def extract_insights(memories: list) -> list:
    """Extract technical insights from memories."""
    insights = []
    for m in memories:
        content = m.get('content', '')
        # Look for insight markers or solution/fact categories
        if m.get('category') in ['solution', 'fact', 'decision']:
            if len(content) > 30 and len(content) < 300:
                insights.append(content)
    return insights[:5]  # Top 5


def generate_devlog_narrative(sanitized_data: dict, insights: list, days: int) -> str:
    """Generate AI narrative using Claude Haiku."""
    stats = sanitized_data['stats']
    categories = sanitized_data['by_category']

    # Count unique projects from tags
    project_count = len(set(
        tag.strip()
        for m in sanitized_data.get('all_memories', [])
        for tag in m.get('tags', '').split(',')
        if tag.strip() and 'Project' in tag.strip()
    ))
    if project_count < 3:
        project_count = max(3, stats['categories'])  # Fallback estimate

    # Prepare VERY generic summaries - strip specifics
    def genericize(text: str) -> str:
        """Remove specific names, keep only generic descriptions."""
        # Remove anything that looks like a product/project name
        text = re.sub(r'\b[A-Z][a-z]+[A-Z]\w+\b', 'a tool', text)  # CamelCase
        text = re.sub(r'\b[a-z]+-[a-z]+(-[a-z]+)*\b', 'a component', text)  # kebab-case
        text = re.sub(r'\[\w+_\w+\]', '[shortcode]', text)  # [shortcode_name]
        return text

    solutions = [genericize(m.get('content', '')[:100]) for m in categories.get('solution', [])[:5]]
    facts = [genericize(m.get('content', '')[:100]) for m in categories.get('fact', [])[:5]]

    # Get variety of work types
    work_types = list(categories.keys())

    time_word = "today" if days == 1 else f"the last {days} days"

    prompt = f"""Write a public developer devlog in English. START DIRECTLY with content - no meta-commentary.

CONTEXT (private - do not mention specifics):
- Time frame: {time_word.upper()} (use "Today" not "This week")
- Worked on approximately {project_count}+ different projects/tasks
- {stats['total']} activities across {len(work_types)} categories: {', '.join(work_types)}

THEMES (generalize these, don't copy verbatim):
{chr(10).join(['- ' + s for s in solutions[:3]]) if solutions else '- Various development tasks'}
{chr(10).join(['- ' + f for f in facts[:3]]) if facts else ''}

CRITICAL RULES:
1. START IMMEDIATELY with the devlog content - NO "I'll write..." or "Here's..." preamble
2. Be EXTREMELY GENERIC - say "a client's WordPress site" not any specific name
3. Say "plugin consolidation" not "merged X into Y"
4. Mention you worked on MULTIPLE projects ({project_count}+), not just 1-2
5. Focus on TYPES of work (debugging, refactoring, learning) not specific features
6. Technical insights should be framework/language level, not project-specific

OUTPUT (start directly, no intro):
[2 paragraphs: variety of work done + general learnings]

### Highlights
- [4-5 generic bullets covering the breadth of work]

### Tomorrow's Focus
- [1-2 general items]"""

    try:
        result = subprocess.run(
            ['claude', '-p', prompt, '--model', 'haiku', '--output-format', 'text', '--max-turns', '1'],
            capture_output=True, text=True, timeout=90
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception as e:
        print(f"Claude error: {e}", file=sys.stderr)

    return "*(Failed to generate narrative)*"


def format_devlog(date_str: str, narrative: str, stats: dict) -> str:
    """Format complete devlog markdown."""
    title_date = datetime.strptime(date_str, '%Y-%m-%d').strftime('%B %d, %Y')

    lines = [
        f"## {title_date} - Daily Dev Notes",
        "",
        narrative,
        "",
        "---",
        "",
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | "
        f"Activities: {stats['total']} | Categories: {stats['categories']}*"
    ]

    return '\n'.join(lines)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate daily devlog")
    parser.add_argument("--days", type=int, default=1, help="Days to look back (default: 1)")
    parser.add_argument("--dry-run", action="store_true", help="Print to terminal only, don't save")
    parser.add_argument("--date", help="Generate for specific date (YYYY-MM-DD)")

    args = parser.parse_args()

    if not check_helix_running():
        print("ERROR: HelixDB not running. Start with: helix deploy", file=sys.stderr)
        return 1

    # Date handling
    if args.date:
        target_date = args.date
    else:
        target_date = datetime.now().strftime('%Y-%m-%d')

    print(f"Generating devlog for {target_date}...", file=sys.stderr)

    # Get data
    print(" Fetching memories...", file=sys.stderr)
    memories = get_recent_memories(args.days, args.date if args.date else None)
    known_projects = get_known_projects()

    if not memories:
        print("No memories found for this period.", file=sys.stderr)
        return 1

    print(f"  Found {len(memories)} memories", file=sys.stderr)

    # Sanitize
    print(" Sanitizing data...", file=sys.stderr)
    sanitizer = Sanitizer()
    sanitized = [sanitize_memory(m, sanitizer, known_projects) for m in memories]

    # Categorize
    by_category = categorize_memories(sanitized)
    stats = {
        'total': len(sanitized),
        'categories': len(by_category)
    }

    # Extract insights (from original, will be sanitized in narrative)
    insights = extract_insights(sanitized)

    # Generate narrative
    print(" Generating narrative (Haiku)...", file=sys.stderr)
    data = {'by_category': by_category, 'stats': stats, 'all_memories': sanitized}
    narrative = generate_devlog_narrative(data, insights, args.days)

    # Format
    devlog = format_devlog(target_date, narrative, stats)

    # Output
    if args.dry_run:
        print("\n" + "=" * 50)
        print(devlog)
        print("=" * 50)
        print("\n[DRY RUN - not saved]", file=sys.stderr)
    else:
        DEVLOG_DIR.mkdir(parents=True, exist_ok=True)
        output_file = DEVLOG_DIR / f"{target_date}.md"
        output_file.write_text(devlog)
        print(f"\n Saved: {output_file}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
