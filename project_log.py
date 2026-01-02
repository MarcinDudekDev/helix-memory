#!/usr/bin/env python3
"""
Project Log Generator

Generates TECHNICAL project/task logs from helix-memory data.
Opposite of daily_devlog.py - shows specific project names, tasks, and technical details.

Outputs grouped by project with tasks/solutions for each.
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent / "hooks"))
from common import check_helix_running, get_all_memories


def get_known_projects() -> dict:
    """Get known projects from p tool with paths."""
    try:
        import subprocess
        result = subprocess.run(
            [str(Path.home() / 'Tools' / 'p'), '--list'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            projects = {}
            for line in result.stdout.split('\n'):
                if ':' in line:
                    parts = line.split(':')
                    name = parts[0].strip()
                    path = parts[1].strip() if len(parts) > 1 else ''
                    if name and name.lower() not in ('users', 'home', 'root'):
                        projects[name.lower()] = name
            return projects
    except:
        pass
    return {}


def detect_project(memory: dict, known_projects: dict) -> str:
    """Detect project name from memory content/tags.

    Priority:
    1. First tag (if memories are tagged correctly, this IS the project)
    2. Pattern matching for legacy/untagged memories
    3. Known projects from registry
    4. Generic tag fallback
    """
    content = memory.get('content', '').lower()
    tags = memory.get('tags', '').lower()
    combined = f"{content} {tags}"

    # Priority 1: First tag is likely the project (new tagging system)
    if tags:
        first_tag = tags.split(',')[0].strip()
        # Validate it's a project-like name (not generic like 'bugfix', 'solution')
        generic_tags = {'bugfix', 'solution', 'error', 'fix', 'update', 'feature',
                       'config', 'setup', 'test', 'debug', 'refactor', 'cleanup'}
        if first_tag and first_tag not in generic_tags and len(first_tag) > 2:
            # Normalize
            if first_tag in ('marriagemarketal', 'marriagemark'):
                return 'marriagemarket'
            return first_tag

    # Priority 2: Pattern matching for legacy memories
    patterns = [
        ('marriagemarket', 'marriagemarket'),
        ('marriage market', 'marriagemarket'),
        ('marriagemark', 'marriagemarket'),  # partial match (includes typos like marriagemarketal)
        ('sites/marriagemarket', 'marriagemarket'),  # directory detection (no leading /)
        ('marriagemarket.loc', 'marriagemarket'),
        ('marriagemarket.local', 'marriagemarket'),
        ('payment-2.0', 'marriagemarket'),
        ('payment 2.0', 'marriagemarket'),
        ('payment_2_0', 'marriagemarket'),
        ('payment2', 'marriagemarket'),
        ('wizard-ajax', 'marriagemarket'),
        ('wizard.js', 'marriagemarket'),
        ('wizard_test', 'marriagemarket'),
        ('wizard,', 'marriagemarket'),  # tag pattern
        ('tier-config', 'marriagemarket'),
        ('adifier', 'marriagemarket'),
        ('single-advert', 'marriagemarket'),
        ('mm2', 'marriagemarket'),
        ('mm_', 'marriagemarket'),
        ('promo code', 'marriagemarket'),
        ('promo codes', 'marriagemarket'),
        ('listing_id', 'marriagemarket'),
        ('my listing', 'marriagemarket'),
        ('create listing', 'marriagemarket'),
        ('user listings', 'marriagemarket'),
        ('multiple listings', 'marriagemarket'),
        (',listing', 'marriagemarket'),  # tag pattern - comma before to avoid 'whitelisting'
        ('promote button', 'marriagemarket'),
        ('webhook_', 'marriagemarket'),
        ('wp-multitool', 'wp-multitool'),
        ('multitool', 'wp-multitool'),
        ('helix-memory', 'helix-memory'),
        ('helix memory', 'helix-memory'),
        ('helix_memory', 'helix-memory'),
        ('oek', 'oek'),
        ('oeksandal', 'oek'),
        ('scf', 'wp-multitool'),
        ('datastar', 'datastar'),
        ('dev-browser', 'dev-browser'),
        ('brand-kit', 'brand-kit-gen'),
        ('brandkit', 'brand-kit-gen'),
        ('supply.family', 'supply-family'),
    ]

    for pattern, project in patterns:
        if pattern in combined:
            return project

    # Check known projects from p tool
    for key, name in known_projects.items():
        if len(key) > 2 and key in combined:  # Skip short keys like 'y'
            return name

    # Check tags for project names
    tag_list = [t.strip() for t in tags.split(',') if t.strip()]
    for tag in tag_list:
        if tag and len(tag) > 2 and tag not in ['wp', 'api', 'php', 'js', 'css', 'database', 'pricing', 'columns']:
            return tag

    return 'other'


def get_memories_for_period(days: int = 1, target_date: str = None, project_filter: str = None) -> list:
    """Get memories from a specific date/period, optionally filtered by project."""
    memories = get_all_memories()

    if target_date:
        target = datetime.strptime(target_date, '%Y-%m-%d')
        start = target.replace(hour=0, minute=0, second=0)
        end = target.replace(hour=23, minute=59, second=59)
    else:
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
                pass

    return recent


def group_by_project(memories: list, known_projects: dict) -> dict:
    """Group memories by detected project."""
    by_project = defaultdict(list)

    for m in memories:
        project = detect_project(m, known_projects)
        by_project[project].append(m)

    return dict(by_project)


def format_memory(m: dict) -> str:
    """Format a single memory for output."""
    category = m.get('category', 'unknown').upper()
    content = m.get('content', '')
    importance = m.get('importance', '?')
    tags = m.get('tags', '')

    # Truncate long content
    if len(content) > 200:
        content = content[:200] + '...'

    line = f"  [{category}] {content}"
    if tags:
        line += f"\n    Tags: {tags}"
    return line


def generate_project_log(date_str: str, by_project: dict, stats: dict) -> str:
    """Generate the project log markdown."""
    title_date = datetime.strptime(date_str, '%Y-%m-%d').strftime('%B %d, %Y')

    lines = [
        f"# Project Log - {title_date}",
        "",
        f"**Activities:** {stats['total']} | **Projects:** {stats['projects']}",
        ""
    ]

    # Sort projects by activity count
    sorted_projects = sorted(by_project.items(), key=lambda x: -len(x[1]))

    for project, memories in sorted_projects:
        lines.append(f"## {project} ({len(memories)} activities)")
        lines.append("")

        # Group by category within project
        by_cat = defaultdict(list)
        for m in memories:
            cat = m.get('category', 'other')
            by_cat[cat].append(m)

        # Show solutions first, then decisions, then facts, then others
        cat_order = ['solution', 'decision', 'problem', 'fact', 'task', 'preference', 'context', 'other']

        for cat in cat_order:
            if cat in by_cat:
                mems = by_cat[cat]
                for m in mems[:10]:  # Limit per category
                    lines.append(format_memory(m))
                    lines.append("")

        lines.append("---")
        lines.append("")

    lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*")
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate technical project log from memories")
    parser.add_argument("--days", type=int, default=1, help="Days to look back (default: 1)")
    parser.add_argument("--date", help="Generate for specific date (YYYY-MM-DD)")
    parser.add_argument("--project", help="Filter by project name")
    parser.add_argument("--output", "-o", help="Output file (default: stdout)")
    parser.add_argument("--month", help="Generate for entire month (YYYY-MM)")

    args = parser.parse_args()

    if not check_helix_running():
        print("ERROR: HelixDB not running. Start with: memory start", file=sys.stderr)
        return 1

    known_projects = get_known_projects()

    if args.month:
        # Generate month summary
        year, month = args.month.split('-')
        year, month = int(year), int(month)

        # Get all days in month
        from calendar import monthrange
        _, last_day = monthrange(year, month)

        all_by_project = defaultdict(list)
        dates_with_work = defaultdict(set)

        for day in range(1, last_day + 1):
            date_str = f"{year}-{month:02d}-{day:02d}"
            memories = get_memories_for_period(target_date=date_str)

            if memories:
                for m in memories:
                    project = detect_project(m, known_projects)
                    # If filtering by project, require exact or canonical match
                    if args.project:
                        proj_lower = args.project.lower()
                        detected_lower = project.lower()
                        # Only match if:
                        # 1. Exact match
                        # 2. Detected starts with filter (marriagemark* -> marriagemarket)
                        # 3. Filter in detected (for typos like marriagemarketal)
                        if not (detected_lower == proj_lower or
                                detected_lower.startswith(proj_lower) or
                                proj_lower in detected_lower):
                            continue
                        # Normalize to canonical name
                        project = args.project
                    all_by_project[project].append(m)
                    dates_with_work[project].add(date_str)

        # Generate month summary
        lines = [
            f"# Project Log - {datetime.strptime(args.month + '-01', '%Y-%m-%d').strftime('%B %Y')}",
            ""
        ]

        total_activities = sum(len(v) for v in all_by_project.values())
        lines.append(f"**Total Activities:** {total_activities} | **Projects:** {len(all_by_project)}")
        lines.append("")

        for project, memories in sorted(all_by_project.items(), key=lambda x: -len(x[1])):
            dates = sorted(dates_with_work[project])
            lines.append(f"## {project}")
            lines.append(f"**Days:** {len(dates)} | **Activities:** {len(memories)}")
            lines.append(f"**Dates:** {', '.join(d[5:] for d in dates)}")  # Show MM-DD
            lines.append("")

            # Show top solutions/decisions
            solutions = [m for m in memories if m.get('category') in ['solution', 'decision']]
            for m in solutions[:10]:
                lines.append(format_memory(m))
                lines.append("")

            lines.append("---")
            lines.append("")

        output = '\n'.join(lines)

    else:
        # Single day
        target_date = args.date or datetime.now().strftime('%Y-%m-%d')

        print(f"Generating project log for {target_date}...", file=sys.stderr)

        memories = get_memories_for_period(args.days, args.date)

        if not memories:
            print("No memories found for this period.", file=sys.stderr)
            return 1

        print(f"  Found {len(memories)} memories", file=sys.stderr)

        by_project = group_by_project(memories, known_projects)

        if args.project:
            by_project = {k: v for k, v in by_project.items()
                         if k.lower() == args.project.lower()}

        stats = {
            'total': len(memories),
            'projects': len(by_project)
        }

        output = generate_project_log(target_date, by_project, stats)

    if args.output:
        Path(args.output).write_text(output)
        print(f"Saved: {args.output}", file=sys.stderr)
    else:
        print(output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
