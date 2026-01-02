#!/usr/bin/env python3
"""
Weekly Retrospective Generator

Creates structured weekly retrospectives combining:
- Time tracking data (h tool)
- Memory/project data (helix-memory)
- AI-generated narrative (Claude Haiku)

Outputs to ~/Documents/Praca/_DZIENNIK_/ as Markdown
Optional: HTML email
"""

import sys
import json
import subprocess
import re
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent / "hooks"))

from common import check_helix_running, get_all_memories

# Configuration
JOURNAL_DIR = Path.home() / "Documents" / "Praca" / "_DZIENNIK_"
ACCESS_FILE = Path(__file__).parent / ".memory_access.json"


def get_week_info(days: int = 7) -> dict:
    """Get week number and date range."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    week_num = end_date.isocalendar()[1]
    year = end_date.year

    return {
        'week_num': week_num,
        'year': year,
        'start': start_date.strftime('%Y-%m-%d'),
        'end': end_date.strftime('%Y-%m-%d'),
        'filename': f"{year}-W{week_num:02d}.md",
        'title': f"Retrospektywa: {start_date.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')} (W{week_num})"
    }


def get_hours_data(year_month: str = None) -> dict:
    """Get hours data from h tool."""
    if not year_month:
        year_month = datetime.now().strftime('%Y-%m')

    try:
        result = subprocess.run(
            [str(Path.home() / 'Tools' / 'h'), 'report', year_month],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode != 0:
            return {'raw': '', 'projects': [], 'total': '0h'}

        output = result.stdout

        # Strip ANSI color codes
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        output = ansi_escape.sub('', output)

        # Parse projects and hours
        projects = []
        total = '0h'

        for line in output.split('\n'):
            # Skip headers and separators
            if '---' in line or 'Projekt' in line:
                continue

            # Parse SUMA line
            if 'SUMA' in line.upper():
                match = re.search(r'(\d+\.?\d*h)', line)
                if match:
                    total = match.group(1)
                continue

            # Parse project lines: "zbm                  18h      3"
            match = re.match(r'^\s*([a-zA-Z0-9_-]+)\s+(\d+\.?\d*h)', line)
            if match:
                projects.append({
                    'name': match.group(1).strip(),
                    'hours': match.group(2)
                })

        return {
            'raw': output,
            'projects': projects,
            'total': total
        }
    except Exception as e:
        print(f"Error getting hours: {e}", file=sys.stderr)
        return {'raw': '', 'projects': [], 'total': '0h'}


def load_access_data() -> dict:
    """Load memory access data."""
    try:
        if ACCESS_FILE.exists():
            with open(ACCESS_FILE) as f:
                return json.load(f)
    except:
        pass
    return {"memories": {}, "last_decay": None}


def get_known_projects() -> set:
    """Get known projects from p tool."""
    try:
        result = subprocess.run(
            [str(Path.home() / 'Tools' / 'p'), '--list'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            projects = set()
            for line in result.stdout.split('\n'):
                if ':' in line:
                    name = line.split(':')[0].strip()
                    if name:
                        projects.add(name.lower())
                        # Also add variations (with .loc, .local, etc.)
                        projects.add(name.lower().replace('-', '.'))
            return projects
    except:
        pass
    return set()


def get_recent_memories(days: int = 7) -> dict:
    """Get memories grouped by category from recent activity."""
    memories = get_all_memories()
    access_data = load_access_data()
    known_projects = get_known_projects()

    by_category = defaultdict(list)
    by_project = defaultdict(list)
    accessed = []

    for m in memories:
        mid = m.get('id', '')
        category = m.get('category', 'unknown')
        tags = m.get('tags', '')

        by_category[category].append(m)

        # Extract project from tags - only count known projects
        for tag in tags.split(','):
            tag = tag.strip()
            if not tag:
                continue

            # Skip generic root directories (but keep 'tools' - important productivity work)
            # Skip usernames that might appear as project names
            if tag.lower() in ('users', 'home', 'root'):
                continue

            # Normalize tag for matching
            tag_lower = tag.lower().replace('.loc', '').replace('.local', '').replace('-', '')

            # Check if tag matches any known project
            matched = False
            for proj in known_projects:
                # Skip generic directories
                if proj.lower() in ('users', 'home', 'root'):
                    continue
                proj_norm = proj.replace('-', '').replace('.loc', '').replace('.local', '')
                if tag_lower == proj_norm or tag_lower in proj_norm or proj_norm in tag_lower:
                    by_project[tag].append(m)
                    matched = True
                    break

            if matched:
                break

        # Check access count
        access_info = access_data.get('memories', {}).get(mid, {})
        access_count = access_info.get('access_count', 0)
        if access_count > 0:
            accessed.append((access_count, m))

    accessed.sort(key=lambda x: x[0], reverse=True)

    return {
        'by_category': dict(by_category),
        'by_project': dict(by_project),
        'most_accessed': accessed[:10],
        'stats': {
            'total': len(memories),
            'categories': len(by_category),
            'projects': len(by_project),
            'accessed': len(accessed)
        }
    }


def generate_ai_narrative(data: dict, hours: dict, days: int) -> str:
    """Generate AI narrative using Claude Haiku."""
    projects = list(data['by_project'].keys())[:15]
    prefs = data['by_category'].get('preference', [])
    top_prefs = [p.get('content', '')[:100] for p in sorted(prefs, key=lambda x: -x.get('importance', 0))[:3]]
    solutions = data['by_category'].get('solution', [])
    top_solutions = [s.get('content', '')[:150] for s in solutions[:5]]
    contexts = data['by_category'].get('context', [])

    session_work = []
    for ctx in contexts[:20]:
        content = ctx.get('content', '')
        if 'Topic:' in content or 'Directory:' in content:
            session_work.append(content[:200])

    hours_info = f"Zalogowane godziny: {hours['total']}"
    if hours['projects']:
        proj_list = ', '.join([p['name'] + ':' + p['hours'] for p in hours['projects'][:5]])
        hours_info += f" ({proj_list})"

    prompt = f"""Napisz krótką, przyjazną retrospektywę tygodniową (max 250 słów) dla developera na podstawie tych danych z ostatnich {days} dni:

CZAS PRACY: {hours_info}

STATYSTYKI: {data['stats']['total']} memories, {data['stats']['projects']} projektów

PROJEKTY: {', '.join(projects[:10])}

SESJE PRACY:
{chr(10).join(session_work[:6])}

ROZWIĄZANIA:
{chr(10).join(top_solutions[:3])}

Napisz jako pomocny asystent reflektujący nad tygodniem. Zawrzyj:
1. Krótki przegląd głównych projektów/osiągnięć (2-3 zdania)
2. Zauważone wzorce (jaki typ pracy dominował?)
3. Jedna konkretna sugestia na następny tydzień

Pisz po polsku, konwersacyjnie. Bez nagłówków markdown, sam tekst z podziałem na akapity."""

    try:
        result = subprocess.run(
            ['claude', '-p', prompt, '--model', 'haiku', '--output-format', 'text', '--max-turns', '1'],
            capture_output=True,
            text=True,
            timeout=90
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception as e:
        print(f"Claude error: {e}", file=sys.stderr)

    return "*(Nie udało się wygenerować narracji AI)*"


def format_markdown_report(week: dict, hours: dict, data: dict, narrative: str) -> str:
    """Format complete markdown report."""
    lines = [
        f"# {week['title']}",
        "",
        "---",
        "",
        "## 📊 Czas pracy",
        "",
    ]

    # Hours table
    if hours['projects']:
        lines.extend([
            "| Projekt | Godziny |",
            "|---------|---------|",
        ])
        for p in hours['projects']:
            lines.append(f"| {p['name']} | {p['hours']} |")
        lines.append(f"| **SUMA** | **{hours['total']}** |")
    else:
        lines.append("*Brak zalogowanych godzin w tym okresie*")

    lines.extend(["", "---", ""])

    # AI Narrative
    lines.extend([
        "## 💭 Refleksja",
        "",
        narrative,
        "",
        "---",
        "",
    ])

    # Projects from memory
    lines.append("## 📁 Projekty (helix-memory)")
    lines.append("")

    top_projects = sorted(data['by_project'].items(), key=lambda x: -len(x[1]))[:10]
    for proj, mems in top_projects:
        lines.append(f"- **{proj}**: {len(mems)} memories")

    lines.extend(["", "---", ""])

    # Key solutions
    solutions = data['by_category'].get('solution', [])
    if solutions:
        lines.extend([
            "## 🔧 Kluczowe rozwiązania",
            "",
        ])
        for s in solutions[:5]:
            content = s.get('content', '')[:120].replace('\n', ' ')
            lines.append(f"- {content}...")
        lines.append("")

    # Stats footer
    lines.extend([
        "---",
        "",
        f"*Wygenerowano: {datetime.now().strftime('%Y-%m-%d %H:%M')} | "
        f"Memories: {data['stats']['total']} | "
        f"Projekty: {data['stats']['projects']}*"
    ])

    return '\n'.join(lines)


def format_html_email(week: dict, hours: dict, data: dict, narrative: str) -> str:
    """Format HTML email version."""
    hours_rows = ""
    if hours['projects']:
        for p in hours['projects']:
            hours_rows += f"<tr><td>{p['name']}</td><td>{p['hours']}</td></tr>"
        hours_rows += f"<tr style='font-weight:bold'><td>SUMA</td><td>{hours['total']}</td></tr>"
    else:
        hours_rows = "<tr><td colspan='2'>Brak zalogowanych godzin</td></tr>"

    projects_list = ""
    top_projects = sorted(data['by_project'].items(), key=lambda x: -len(x[1]))[:8]
    for proj, mems in top_projects:
        projects_list += f"<li><strong>{proj}</strong>: {len(mems)} memories</li>"

    solutions_list = ""
    solutions = data['by_category'].get('solution', [])
    for s in solutions[:4]:
        content = s.get('content', '')[:100].replace('\n', ' ')
        solutions_list += f"<li>{content}...</li>"

    narrative_html = narrative.replace('\n\n', '</p><p>').replace('\n', '<br>')

    return f"""<!DOCTYPE html>
<html>
<head>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; color: #333; }}
h1 {{ color: #2563eb; border-bottom: 2px solid #2563eb; padding-bottom: 10px; }}
h2 {{ color: #1e40af; margin-top: 25px; }}
table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
th {{ background: #f3f4f6; }}
.narrative {{ background: #f8fafc; padding: 15px; border-radius: 8px; border-left: 4px solid #2563eb; }}
.footer {{ color: #6b7280; font-size: 12px; margin-top: 30px; border-top: 1px solid #e5e7eb; padding-top: 10px; }}
ul {{ padding-left: 20px; }}
li {{ margin: 5px 0; }}
</style>
</head>
<body>
<h1>📋 {week['title']}</h1>

<h2>📊 Czas pracy</h2>
<table>
<tr><th>Projekt</th><th>Godziny</th></tr>
{hours_rows}
</table>

<h2>💭 Refleksja</h2>
<div class="narrative">
<p>{narrative_html}</p>
</div>

<h2>📁 Projekty</h2>
<ul>{projects_list}</ul>

{f'<h2>🔧 Rozwiązania</h2><ul>{solutions_list}</ul>' if solutions_list else ''}

<div class="footer">
Wygenerowano: {datetime.now().strftime('%Y-%m-%d %H:%M')} |
Memories: {data['stats']['total']} | Projekty: {data['stats']['projects']}
</div>
</body>
</html>"""


def send_html_email(to_addr: str, subject: str, html_body: str) -> bool:
    """Send HTML email via SMTP or Mail.app fallback."""
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart

    # SMTP config file: ~/.config/helix-memory/smtp.json
    smtp_config = Path.home() / '.config' / 'helix-memory' / 'smtp.json'

    if smtp_config.exists():
        try:
            config = json.loads(smtp_config.read_text())
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = config.get('from', config['user'])
            msg['To'] = to_addr

            # Plain text fallback
            plain = html_body.replace('<br>', '\n').replace('</p>', '\n\n')
            plain = re.sub(r'<[^>]+>', '', plain)
            msg.attach(MIMEText(plain, 'plain'))
            msg.attach(MIMEText(html_body, 'html'))

            with smtplib.SMTP_SSL(config['host'], config.get('port', 465)) as server:
                server.login(config['user'], config['password'])
                server.send_message(msg)
            return True
        except Exception as e:
            print(f"SMTP error: {e}", file=sys.stderr)
            return False

    # Fallback: macOS Mail.app
    temp_file = Path('/tmp/retrospektywa_email.html')
    temp_file.write_text(html_body)

    script = f'''
    set htmlContent to (read POSIX file "/tmp/retrospektywa_email.html")
    tell application "Mail"
        set newMessage to make new outgoing message with properties {{subject:"{subject}", visible:false}}
        tell newMessage
            make new to recipient at end of to recipients with properties {{address:"{to_addr}"}}
            set html content to htmlContent
        end tell
        send newMessage
    end tell
    '''
    result = subprocess.run(['osascript', '-e', script], capture_output=True, text=True)
    temp_file.unlink(missing_ok=True)
    return result.returncode == 0


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate weekly retrospective")
    parser.add_argument("--days", type=int, default=7, help="Days to look back")
    parser.add_argument("--email", help="Also send HTML email to this address")
    parser.add_argument("--no-file", action="store_true", help="Don't save MD file")
    parser.add_argument("--terminal", action="store_true", help="Print to terminal only")
    parser.add_argument("--force", action="store_true", help="Regenerate even if file exists")
    parser.add_argument("--send-existing", action="store_true", help="Send existing file via email without regenerating")

    args = parser.parse_args()

    if not check_helix_running():
        print("ERROR: HelixDB not running", file=sys.stderr)
        return 1

    # Check for existing file
    week = get_week_info(args.days)
    JOURNAL_DIR.mkdir(parents=True, exist_ok=True)
    output_file = JOURNAL_DIR / week['filename']

    # If file exists and not forcing regeneration
    if output_file.exists() and not args.force and not args.terminal:
        if args.send_existing and args.email:
            # Just send existing file
            print(f"📄 Używam istniejącego pliku: {output_file}", file=sys.stderr)
            md_report = output_file.read_text()
            # Parse existing report to get data for HTML
            hours = get_hours_data()
            data = get_recent_memories(args.days)
            # Extract narrative from existing file
            narrative = ""
            in_narrative = False
            for line in md_report.split('\n'):
                if '## 💭 Refleksja' in line:
                    in_narrative = True
                    continue
                if in_narrative and line.startswith('---'):
                    break
                if in_narrative and line.strip():
                    narrative += line + '\n'
            narrative = narrative.strip()

            print(f"📧 Wysyłanie email do {args.email}...", file=sys.stderr)
            html_report = format_html_email(week, hours, data, narrative)
            if send_html_email(args.email, week['title'], html_report):
                print(f"✅ Email wysłany!", file=sys.stderr)
            else:
                print("❌ Błąd wysyłania email", file=sys.stderr)
                return 1
            return 0

        # Interactive prompt
        print(f"\n⚠️  Plik już istnieje: {output_file}", file=sys.stderr)
        print("Opcje:", file=sys.stderr)
        print("  1. Wygeneruj nowy (--force)", file=sys.stderr)
        print("  2. Wyślij istniejący email (--send-existing --email X)", file=sys.stderr)
        print("  3. Pokaż istniejący (--terminal)", file=sys.stderr)
        return 0

    # Gather data
    print("📊 Zbieranie danych...", file=sys.stderr)
    hours = get_hours_data()
    data = get_recent_memories(args.days)

    # Generate AI narrative
    print("💭 Generowanie refleksji AI...", file=sys.stderr)
    narrative = generate_ai_narrative(data, hours, args.days)

    # Format markdown
    md_report = format_markdown_report(week, hours, data, narrative)

    # Terminal only mode
    if args.terminal:
        print(md_report)
        return 0

    # Save markdown file
    if not args.no_file:
        JOURNAL_DIR.mkdir(parents=True, exist_ok=True)
        output_file = JOURNAL_DIR / week['filename']
        output_file.write_text(md_report)
        print(f"✅ Zapisano: {output_file}", file=sys.stderr)

    # Send email if requested
    if args.email:
        print(f"📧 Wysyłanie email do {args.email}...", file=sys.stderr)
        html_report = format_html_email(week, hours, data, narrative)
        if send_html_email(args.email, week['title'], html_report):
            print(f"✅ Email wysłany!", file=sys.stderr)
        else:
            print("❌ Błąd wysyłania email", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
