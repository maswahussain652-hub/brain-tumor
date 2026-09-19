"""
NeuroScan 3D: Unified Launcher and Dependency Manager.
Checks and installs required dependencies automatically, validates environment,
and launches the Streamlit 3D Medical AI application with a single command.

Usage:
    python main.py
"""

import sys
import os
import subprocess
import threading
import time
import webbrowser
from pathlib import Path

# Required package mapping: (import_name, pip_package_name)
REQUIRED_PACKAGES = [
    ("numpy", "numpy>=1.24.0"),
    ("scipy", "scipy>=1.11.0"),
    ("torch", "torch>=2.0.0"),
    ("skimage", "scikit-image>=0.22.0"),
    ("plotly", "plotly>=5.18.0"),
    ("streamlit", "streamlit>=1.30.0"),
]


def print_banner() -> None:
    """Prints medical AI CLI startup banner."""
    banner = r"""
================================================================================
   _  __                     ____                   _____ ____ 
  / |/ /___ __ __ ____ ___  / __/____ ___ _ ___    |_  // _  \
 /    // -_) // // __// _ \_\ \ / __// _ `// _ \  _/_ < / // /
/_/|_/ \__/\_,_//_/   \___/___/ \__/ \_,_//_//_/ /____//____/ 
                                                               
  3D BRAIN TUMOR DETECTION & VOLUMETRIC MESH RENDERING ENGINE
  PyTorch 3D UNet  |  Marching Cubes  |  Streamlit + Plotly
================================================================================
"""
    print(banner)


def check_and_install_dependencies() -> None:
    """Verifies each required package, automatically installing missing ones."""
    missing_packages = []
    print("[+] Inspecting Python runtime environment...")

    for import_name, package_spec in REQUIRED_PACKAGES:
        try:
            __import__(import_name)
            print(f"    [OK] Found dependency: {import_name}")
        except ImportError:
            print(f"    [!] Missing dependency: {import_name} -> needs '{package_spec}'")
            missing_packages.append(package_spec)

    if missing_packages:
        print("\n[+] Automatically installing missing packages via pip...")
        cmd = [sys.executable, "-m", "pip", "install", *missing_packages]
        try:
            subprocess.check_call(cmd)
            print("[+] All dependencies successfully installed!\n")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed to install dependencies: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print("[+] All dependencies satisfied.\n")


def _auto_open_browser(url: str, delay: float = 2.0) -> None:
    """Opens browser automatically after Streamlit server initializes."""
    time.sleep(delay)
    try:
        webbrowser.open(url)
    except Exception:
        pass


def launch_application() -> None:
    """Launches the Streamlit application and pops up the browser."""
    project_dir = Path(__file__).resolve().parent
    app_script = project_dir / "app.py"

    if not app_script.exists():
        print(f"[ERROR] Application file not found: {app_script}", file=sys.stderr)
        sys.exit(1)

    port = 8501
    url = f"http://localhost:{port}"

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_script),
        f"--server.port={port}",
        "--server.address=localhost",
        "--server.headless=false",
        "--theme.base=dark",
        "--theme.primaryColor=#06B6D4",
        "--theme.backgroundColor=#0B0F19",
        "--theme.secondaryBackgroundColor=#111827",
        "--theme.textColor=#F8FAFC",
    ]

    print("=" * 80)
    print("  [>] Launching NeuroScan 3D Web Cockpit...")
    print(f"  [>] Local URL: {url}")
    print("  [>] Opening browser automatically...")
    print("  [>] Press Ctrl + C in this terminal to terminate the server.")
    print("=" * 80 + "\n")

    # Start delayed background thread to open browser
    threading.Thread(target=_auto_open_browser, args=(url, 2.5), daemon=True).start()

    try:
        # Run streamlit as a child process
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n[+] NeuroScan 3D gracefully shut down. Goodbye!")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Streamlit process exited with error code {e.returncode}", file=sys.stderr)


if __name__ == "__main__":
    print_banner()
    check_and_install_dependencies()
    launch_application()
