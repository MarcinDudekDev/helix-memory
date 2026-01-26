#!/usr/bin/env python3
"""
HelixDB Watchdog - Auto-restarts HelixDB on crash and logs failures.

Usage:
  watchdog.py start       Start watchdog in foreground
  watchdog.py daemon      Start watchdog as background daemon
  watchdog.py stop        Stop watchdog daemon
  watchdog.py status      Check watchdog and HelixDB status
  watchdog.py logs        Show recent crash logs

The watchdog:
- Checks HelixDB health every 30 seconds
- Auto-restarts on failure
- Logs crash events with timestamps and reasons
- Provides crash history for debugging
"""

import os
import sys
import time
import json
import signal
import subprocess
import configparser
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict

# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR = Path(__file__).parent.parent
CONFIG_FILE = Path.home() / ".helix-memory.conf"
LOG_DIR = Path.home() / ".cache/helix-memory"
CRASH_LOG = LOG_DIR / "crashes.jsonl"
PID_FILE = LOG_DIR / "watchdog.pid"
CHECK_INTERVAL = 30  # seconds

# Load config
def _load_config():
    config = configparser.ConfigParser()
    if CONFIG_FILE.exists():
        config.read(CONFIG_FILE)
    return config

_CONFIG = _load_config()

def _get_config(section: str, key: str, default: str = "") -> str:
    try:
        value = _CONFIG.get(section, key)
        if value.startswith("~"):
            value = str(Path(value).expanduser())
        return value
    except (configparser.NoSectionError, configparser.NoOptionError):
        return default

HELIX_URL = _get_config("helix", "url", "http://localhost:6969")
HELIX_BIN = _get_config("paths", "helix_bin", str(Path.home() / ".local/bin/helix"))

# ============================================================================
# LOGGING
# ============================================================================

def ensure_log_dir():
    """Ensure log directory exists."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

def log_crash(reason: str, details: Optional[Dict] = None):
    """Log a crash event to the crash log."""
    ensure_log_dir()
    entry = {
        "timestamp": datetime.now().isoformat(),
        "reason": reason,
        "details": details or {}
    }
    with open(CRASH_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"[CRASH] {entry['timestamp']}: {reason}", file=sys.stderr)

def log_info(msg: str):
    """Log info message."""
    print(f"[INFO] {datetime.now().isoformat()}: {msg}")

def log_error(msg: str):
    """Log error message."""
    print(f"[ERROR] {datetime.now().isoformat()}: {msg}", file=sys.stderr)

def get_recent_crashes(limit: int = 10) -> list:
    """Get recent crash entries."""
    if not CRASH_LOG.exists():
        return []
    crashes = []
    with open(CRASH_LOG) as f:
        for line in f:
            if line.strip():
                try:
                    crashes.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return crashes[-limit:]

# ============================================================================
# HEALTH CHECK
# ============================================================================

def check_docker_running() -> bool:
    """Check if Docker is running."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=5
        )
        return result.returncode == 0
    except:
        return False

def check_helix_running() -> tuple:
    """
    Check if HelixDB is running and responsive.
    Returns (is_running, error_reason)
    """
    import requests
    try:
        response = requests.post(
            f"{HELIX_URL}/GetAllMemories",
            json={},
            headers={"Content-Type": "application/json"},
            timeout=5
        )
        if response.status_code == 200:
            return True, None
        else:
            return False, f"HTTP {response.status_code}"
    except requests.exceptions.ConnectionError:
        return False, "Connection refused"
    except requests.exceptions.Timeout:
        return False, "Request timeout"
    except Exception as e:
        return False, str(e)

def get_container_status() -> Optional[str]:
    """Get HelixDB container status."""
    try:
        result = subprocess.run(
            ["docker", "ps", "-a", "--filter", "name=helix", "--format", "{{.Status}}"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().split('\n')[0]
        return None
    except:
        return None

# ============================================================================
# RESTART LOGIC
# ============================================================================

def restart_helix() -> bool:
    """
    Attempt to restart HelixDB.
    Tries helix CLI first, falls back to docker.
    Returns True if restart successful.
    """
    log_info("Attempting to restart HelixDB...")

    # First check Docker
    if not check_docker_running():
        log_info("Docker not running, starting Docker Desktop...")
        subprocess.run(["open", "-a", "Docker Desktop"], capture_output=True)
        # Wait for Docker
        for _ in range(30):
            if check_docker_running():
                log_info("Docker started")
                break
            time.sleep(2)
        else:
            log_error("Docker failed to start")
            return False

    # Try helix CLI restart
    try:
        log_info("Trying helix CLI restart...")
        result = subprocess.run(
            [HELIX_BIN, "stop", "dev"],
            cwd=str(SCRIPT_DIR),
            capture_output=True,
            text=True,
            timeout=30
        )
        time.sleep(2)

        result = subprocess.run(
            [HELIX_BIN, "push", "dev"],
            cwd=str(SCRIPT_DIR),
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode == 0:
            time.sleep(3)
            running, _ = check_helix_running()
            if running:
                log_info("HelixDB restarted via helix CLI")
                return True
    except Exception as e:
        log_info(f"Helix CLI failed: {e}")

    # Fallback to docker
    try:
        log_info("Trying docker restart...")
        subprocess.run(
            ["docker", "rm", "-f", "helix-memory-dev"],
            capture_output=True,
            timeout=10
        )
        time.sleep(1)

        result = subprocess.run(
            ["docker", "run", "-d",
             "--name", "helix-memory-dev",
             "-p", "6969:6969",
             "-v", f"{SCRIPT_DIR}/.helix/.volumes/dev/user:/root/.helix/user",
             "helix-helix-memory-dev:debug"],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            time.sleep(3)
            running, _ = check_helix_running()
            if running:
                log_info("HelixDB restarted via docker")
                return True
    except Exception as e:
        log_error(f"Docker restart failed: {e}")

    return False

# ============================================================================
# WATCHDOG LOOP
# ============================================================================

def run_watchdog():
    """Main watchdog loop."""
    log_info("Watchdog started")
    ensure_log_dir()

    consecutive_failures = 0
    max_consecutive = 3

    while True:
        try:
            running, error = check_helix_running()

            if running:
                if consecutive_failures > 0:
                    log_info(f"HelixDB recovered after {consecutive_failures} failures")
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                container_status = get_container_status()

                log_crash(error or "Unknown failure", {
                    "consecutive_failures": consecutive_failures,
                    "container_status": container_status
                })

                if consecutive_failures <= max_consecutive:
                    log_info(f"Failure {consecutive_failures}/{max_consecutive}, attempting restart...")
                    if restart_helix():
                        log_info("Restart successful!")
                    else:
                        log_error("Restart failed")
                else:
                    log_error(f"Too many consecutive failures ({consecutive_failures}), waiting for intervention")
                    # Reset counter after logging, still try to recover
                    consecutive_failures = max_consecutive

            time.sleep(CHECK_INTERVAL)

        except KeyboardInterrupt:
            log_info("Watchdog stopped by user")
            break
        except Exception as e:
            log_error(f"Watchdog error: {e}")
            time.sleep(CHECK_INTERVAL)

# ============================================================================
# DAEMON MANAGEMENT
# ============================================================================

def start_daemon():
    """Start watchdog as background daemon."""
    ensure_log_dir()

    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text().strip())
            # Check if process exists
            os.kill(pid, 0)
            print(f"Watchdog already running (PID {pid})")
            return
        except (OSError, ValueError):
            # Process doesn't exist, clean up
            PID_FILE.unlink()

    # Fork process
    pid = os.fork()
    if pid > 0:
        print(f"Watchdog daemon started (PID {pid})")
        return

    # Child process - become daemon
    os.setsid()
    os.chdir("/")

    # Redirect stdout/stderr to log
    log_file = LOG_DIR / "watchdog.log"
    with open(log_file, "a") as f:
        os.dup2(f.fileno(), sys.stdout.fileno())
        os.dup2(f.fileno(), sys.stderr.fileno())

    # Write PID file
    PID_FILE.write_text(str(os.getpid()))

    # Run watchdog
    try:
        run_watchdog()
    finally:
        if PID_FILE.exists():
            PID_FILE.unlink()

def stop_daemon():
    """Stop watchdog daemon."""
    if not PID_FILE.exists():
        print("Watchdog not running")
        return

    try:
        pid = int(PID_FILE.read_text().strip())
        os.kill(pid, signal.SIGTERM)
        print(f"Watchdog stopped (PID {pid})")
        time.sleep(1)
        if PID_FILE.exists():
            PID_FILE.unlink()
    except (OSError, ValueError) as e:
        print(f"Error stopping watchdog: {e}")
        if PID_FILE.exists():
            PID_FILE.unlink()

def show_status():
    """Show watchdog and HelixDB status."""
    # Watchdog status
    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text().strip())
            os.kill(pid, 0)
            print(f"Watchdog: RUNNING (PID {pid})")
        except (OSError, ValueError):
            print("Watchdog: NOT RUNNING (stale PID file)")
    else:
        print("Watchdog: NOT RUNNING")

    # HelixDB status
    running, error = check_helix_running()
    if running:
        # Get memory count
        import requests
        try:
            response = requests.post(
                f"{HELIX_URL}/GetAllMemories",
                json={},
                headers={"Content-Type": "application/json"},
                timeout=5
            )
            count = len(response.json().get("memories", []))
            print(f"HelixDB: RUNNING ({count} memories)")
        except:
            print("HelixDB: RUNNING")
    else:
        print(f"HelixDB: DOWN ({error})")
        container_status = get_container_status()
        if container_status:
            print(f"Container: {container_status}")

    # Recent crashes
    crashes = get_recent_crashes(3)
    if crashes:
        print(f"\nRecent crashes ({len(crashes)}):")
        for c in crashes[-3:]:
            print(f"  {c['timestamp']}: {c['reason']}")

def show_logs():
    """Show recent crash logs."""
    crashes = get_recent_crashes(20)
    if not crashes:
        print("No crash logs found")
        return

    print(f"=== Crash Log ({len(crashes)} entries) ===\n")
    for c in crashes:
        print(f"[{c['timestamp']}]")
        print(f"  Reason: {c['reason']}")
        if c.get('details'):
            for k, v in c['details'].items():
                print(f"  {k}: {v}")
        print()

# ============================================================================
# MAIN
# ============================================================================

def print_help():
    print(__doc__)

def main():
    if len(sys.argv) < 2:
        print_help()
        sys.exit(1)

    cmd = sys.argv[1].lower()

    if cmd == "start":
        run_watchdog()
    elif cmd == "daemon":
        start_daemon()
    elif cmd == "stop":
        stop_daemon()
    elif cmd == "status":
        show_status()
    elif cmd == "logs":
        show_logs()
    elif cmd in ("help", "-h", "--help"):
        print_help()
    else:
        print(f"Unknown command: {cmd}")
        print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
