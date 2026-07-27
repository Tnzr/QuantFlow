#!/usr/bin/env python3
"""QuantFlow Launcher — terminal interface for selecting and launching configurations.

Usage:
    python launch.py

Features:
    - Select environment: Local (conda) or Docker
    - Select mode: Full platform, Server only, Client only
    - Toggle Alpaca paper trading (live data + paper trades)
    - Auto-detect GPU for ML engine
    - Health check all services after launch
"""

import os, subprocess, sys, time, json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# ── Color helpers ─────────────────────────────────────────────────────
C = {
    "reset": "\033[0m", "bold": "\033[1m", "dim": "\033[2m",
    "red": "\033[31m", "green": "\033[32m", "yellow": "\033[33m",
    "blue": "\033[34m", "magenta": "\033[35m", "cyan": "\033[36m",
    "bg_green": "\033[42m", "bg_red": "\033[41m",
}

def cprint(text, color="reset", bold=False, end="\n"):
    prefix = C["bold"] if bold else ""
    print(f"{prefix}{C.get(color, '')}{text}{C['reset']}", end=end)

def header(text):
    print(f"\n{C['bold']}{C['cyan']}═══ {text} {C['reset']}")

def menu(title, options):
    """Display numbered menu and return selection index."""
    print(f"\n{C['bold']}{C['yellow']}{title}{C['reset']}")
    for i, (label, desc, _) in enumerate(options):
        print(f"  {C['bold']}{i+1}.{C['reset']} {label} {C['dim']}— {desc}{C['reset']}")
    while True:
        try:
            choice = input(f"\n{C['bold']}Select [{C['reset']}1-{len(options)}{C['bold']}]: {C['reset']}")
            if choice.strip() == "":
                return 0  # default to first
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                return idx
        except ValueError:
            pass
        cprint(f"Enter 1-{len(options)}", "red")

def yes_no(question, default=True):
    default_str = "Y/n" if default else "y/N"
    resp = input(f"{C['bold']}{question} [{default_str}]: {C['reset']}").strip().lower()
    if resp == "":
        return default
    return resp in ("y", "yes")

# ── Main ────────────────────────────────────────────────────────────────

def main():
    print(f"\n{C['bold']}{C['cyan']}╔══════════════════════════════════════════╗{C['reset']}")
    print(f"{C['bold']}{C['cyan']}║      QuantFlow Platform Launcher         ║{C['reset']}")
    print(f"{C['bold']}{C['cyan']}╚══════════════════════════════════════════╝{C['reset']}")
    print(f"{C['dim']}  CascadeANP v1.0 · Alpaca Data · Phase 2 Backtest{C['reset']}")

    # 1. Environment
    env_opts = [
        ("Local (conda)", "Run services directly with conda env", "conda"),
        ("Docker", "Run services in containers", "docker"),
    ]
    env_idx = menu("Choose environment:", env_opts)
    env = env_opts[env_idx][2]

    # 2. Mode
    mode_opts = [
        ("Full Platform", "nginx + backend + ML engine + postgres + redis", "full"),
        ("Server Only", "backend + ML engine + postgres + redis (no frontend)", "server"),
        ("Client Only", "nginx + frontend (connect to remote server)", "client"),
    ]
    mode_idx = menu("Choose mode:", mode_opts)
    mode = mode_opts[mode_idx][2]

    # 3. Paper trading
    paper_trading = yes_no("Enable Alpaca paper trading? (live quotes + paper orders)", True)

    # 4. Check prerequisites
    header("Checking prerequisites")
    issues = []

    # Check .env
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        cprint("  ✗ .env file not found. Creating from .env.example...", "yellow")
        example = PROJECT_ROOT / ".env.example"
        if example.exists():
            env_file.write_text(example.read_text())
            cprint("  ✓ .env created. Edit with your API keys.", "green")
        issues.append("Edit .env with your Alpaca API keys before running paper trading.")
    else:
        has_alpaca = "ALPACA_API_KEY=PK" in env_file.read_text()
        if has_alpaca:
            cprint("  ✓ .env found with Alpaca keys", "green")
        else:
            cprint("  ⚠ .env found but Alpaca keys not set", "yellow")

    # Check Docker
    if env == "docker":
        try:
            subprocess.run(["docker", "info"], capture_output=True, timeout=5)
            cprint("  ✓ Docker daemon available", "green")
        except Exception:
            cprint("  ✗ Docker not accessible. Start Docker Desktop.", "red")
            issues.append("Docker must be running for Docker mode.")

    # 5. Summary and confirm
    header("Configuration Summary")
    print(f"  Environment:  {env}")
    print(f"  Mode:         {mode}")
    print(f"  Paper trade:  {paper_trading}")
    if issues:
        print(f"\n{C['yellow']}  Issues to resolve:{C['reset']}")
        for i in issues:
            print(f"    • {i}")

    if not yes_no("\nLaunch with this configuration?", True):
        cprint("Aborted.", "red")
        return

    # 6. Launch
    header("Launching")
    start_time = time.time()

    try:
        if env == "docker":
            _launch_docker(mode, paper_trading)
        else:
            _launch_local(mode, paper_trading)
    except KeyboardInterrupt:
        cprint("\n\nShutting down...", "yellow")
        if env == "docker":
            subprocess.run(["docker", "compose", "down"], cwd=PROJECT_ROOT, capture_output=True)
    except Exception as e:
        cprint(f"\nLaunch failed: {e}", "red")
        return

    # 7. Health check
    elapsed = time.time() - start_time
    if env == "docker":
        time.sleep(5)
        _health_check(mode)

    cprint(f"\n{C['bold']}{C['green']}✓ Platform launched in {elapsed:.0f}s{C['reset']}")
    _print_urls(mode)


def _launch_docker(mode, paper_trading):
    """Launch with Docker compose."""
    os.chdir(PROJECT_ROOT)
    env = os.environ.copy()
    if paper_trading:
        env["PAPER_TRADING"] = "true"

    services = []
    if mode == "full":
        services = ["postgres", "redis", "ml-engine", "backend", "nginx"]
    elif mode == "server":
        services = ["postgres", "redis", "ml-engine", "backend"]
    elif mode == "client":
        services = ["nginx"]

    cprint(f"  Building and starting: {', '.join(services)}...", "dim")
    cmd = ["docker", "compose", "up", "-d", "--build"] + services
    subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)


def _launch_local(mode, paper_trading):
    """Launch services locally using conda env."""
    cprint("  Local mode uses the quantflow conda environment.", "dim")
    cprint("  Starting services...", "dim")

    # Start backend
    if mode in ("full", "server"):
        subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "quantflow.api.server:app",
             "--host", "0.0.0.0", "--port", "3000"],
            cwd=PROJECT_ROOT,
        )
        cprint("  ✓ Backend started on :3000", "green")

    # Start ML engine
    if mode in ("full", "server"):
        subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "docker.ml.server:app",
             "--host", "0.0.0.0", "--port", "8000"],
            cwd=PROJECT_ROOT,
        )
        cprint("  ✓ ML Engine started on :8000", "green")

    # Start frontend (serve web export)
    frontend_dist = PROJECT_ROOT / "frontend-expo" / "dist"
    if mode in ("full", "client") and frontend_dist.is_dir():
        subprocess.Popen(
            [sys.executable, "-m", "http.server", "8080",
             "--directory", str(frontend_dist)],
            cwd=PROJECT_ROOT,
        )
        cprint("  ✓ Frontend started on :8080", "green")


def _health_check(mode):
    """Verify all services are responding."""
    import urllib.request

    checks = []
    if mode in ("full", "server"):
        checks.append(("Backend", "http://localhost:3000/health"))
        checks.append(("ML Engine", "http://localhost:8000/health"))
    if mode in ("full", "client"):
        checks.append(("Frontend", "http://localhost:8080/"))

    for name, url in checks:
        try:
            urllib.request.urlopen(url, timeout=5)
            cprint(f"  ✓ {name}: OK", "green")
        except Exception:
            cprint(f"  ✗ {name}: not responding", "red")


def _print_urls(mode):
    print(f"\n{C['bold']}Access your platform:{C['reset']}")
    if mode in ("full", "client"):
        print(f"  Frontend:  {C['cyan']}http://localhost:8080{C['reset']}")
    if mode in ("full", "server"):
        print(f"  API:       {C['cyan']}http://localhost:3000/docs{C['reset']}")
        print(f"  ML Engine: {C['cyan']}http://localhost:8000/health{C['reset']}")
    print(f"\n{C['dim']}Press Ctrl+C to stop all services.{C['reset']}\n")


if __name__ == "__main__":
    main()
