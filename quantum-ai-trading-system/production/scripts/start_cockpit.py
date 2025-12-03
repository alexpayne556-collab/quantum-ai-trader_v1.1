#!/usr/bin/env python3
"""
Quantum AI Cockpit — Universal Cross-Platform Launcher
========================================================
Auto-detects OS, activates venv, launches backend + frontend, opens dashboard.
Protocol 22.4-INTELLECTIA-LAUNCHER
"""

import os
import sys
import platform
import subprocess
import time
import webbrowser
from pathlib import Path
from typing import Optional, Tuple

# Colors for terminal output
class Colors:
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_colored(text: str, color: str = Colors.CYAN):
    """Print colored text to terminal."""
    if platform.system() == "Windows":
        # Windows may not support ANSI colors, use basic print
        print(text)
    else:
        print(f"{color}{text}{Colors.RESET}")

def detect_os() -> str:
    """Detect operating system."""
    system = platform.system()
    if system == "Windows":
        return "windows"
    elif system == "Linux":
        return "linux"
    elif system == "Darwin":
        return "macos"
    else:
        return "unknown"

def get_project_root() -> Path:
    """Get project root directory, handling both Windows and Ubuntu paths."""
    current = Path(__file__).resolve().parent
    
    # Check for common mount points
    if current.parts[0] == "mnt" and "shared" in current.parts:
        # Ubuntu VM mount
        return current
    elif str(current).startswith("D:\\") or str(current).startswith("C:\\"):
        # Windows path
        return current
    else:
        # Default to current directory
        return current

def find_venv() -> Optional[Path]:
    """Find virtual environment directory."""
    project_root = get_project_root()
    
    # Common venv locations
    venv_paths = [
        project_root / "venv",
        project_root / ".venv",
        project_root / "quantum_venv",
        Path.home() / "quantum_venv",
    ]
    
    for venv_path in venv_paths:
        if venv_path.exists():
            # Check for activation script
            if platform.system() == "Windows":
                activate = venv_path / "Scripts" / "activate.bat"
            else:
                activate = venv_path / "bin" / "activate"
            
            if activate.exists():
                return venv_path
    
    return None

def get_python_cmd() -> str:
    """Get Python command based on OS."""
    if platform.system() == "Windows":
        return "python"
    else:
        return "python3"

def get_venv_python(venv_path: Path) -> str:
    """Get Python executable from venv."""
    if platform.system() == "Windows":
        return str(venv_path / "Scripts" / "python.exe")
    else:
        return str(venv_path / "bin" / "python3")

def check_dependencies() -> Tuple[bool, list]:
    """Check if required dependencies are installed."""
    missing = []
    
    try:
        import fastapi
        import uvicorn
        import pandas
        import numpy
    except ImportError as e:
        missing.append(str(e))
    
    return len(missing) == 0, missing

def run_backend(venv_python: str, project_root: Path) -> subprocess.Popen:
    """Launch FastAPI backend."""
    print_colored("🚀 Starting FastAPI backend...", Colors.CYAN)
    
    backend_dir = project_root / "backend"
    backend_main = backend_dir / "main.py"
    
    if not backend_main.exists():
        print_colored(f"❌ Backend main.py not found at {backend_main}", Colors.RED)
        sys.exit(1)
    
    # Change to backend directory
    os.chdir(backend_dir)
    
    # Launch uvicorn
    cmd = [
        venv_python, "-m", "uvicorn",
        "main:app",
        "--host", "0.0.0.0",
        "--port", "8090",
        "--reload"
    ]
    
    if platform.system() == "Windows":
        # Windows: use CREATE_NEW_CONSOLE to show output
        process = subprocess.Popen(
            cmd,
            creationflags=subprocess.CREATE_NEW_CONSOLE
        )
    else:
        # Linux/Mac: run in background
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
    
    return process

def run_frontend(project_root: Path) -> subprocess.Popen:
    """Launch Vite frontend."""
    print_colored("🎨 Starting Vite frontend...", Colors.CYAN)
    
    frontend_dir = project_root / "frontend"
    
    if not frontend_dir.exists():
        print_colored(f"❌ Frontend directory not found at {frontend_dir}", Colors.RED)
        sys.exit(1)
    
    # Check if node_modules exists
    if not (frontend_dir / "node_modules").exists():
        print_colored("📦 Installing frontend dependencies...", Colors.YELLOW)
        subprocess.run(["npm", "install"], cwd=frontend_dir, check=True)
    
    # Launch Vite
    if platform.system() == "Windows":
        cmd = ["npm", "run", "dev"]
        process = subprocess.Popen(
            cmd,
            cwd=frontend_dir,
            creationflags=subprocess.CREATE_NEW_CONSOLE
        )
    else:
        cmd = ["npm", "run", "dev"]
        process = subprocess.Popen(
            cmd,
            cwd=frontend_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
    
    return process

def wait_for_server(url: str, timeout: int = 30) -> bool:
    """Wait for server to be ready."""
    import urllib.request
    import urllib.error
    
    print_colored(f"⏳ Waiting for server at {url}...", Colors.YELLOW)
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            urllib.request.urlopen(url, timeout=2)
            print_colored(f"✅ Server ready at {url}", Colors.GREEN)
            return True
        except (urllib.error.URLError, OSError):
            time.sleep(1)
    
    print_colored(f"❌ Server did not start within {timeout} seconds", Colors.RED)
    return False

def open_browser(url: str):
    """Open browser to dashboard URL."""
    print_colored(f"🌐 Opening browser to {url}...", Colors.CYAN)
    time.sleep(2)  # Give server a moment to fully start
    webbrowser.open(url)

def run_validation(venv_python: str, project_root: Path) -> bool:
    """Run system validation."""
    print_colored("🔍 Running system validation...", Colors.CYAN)
    
    validation_script = project_root / "backend" / "system_validation.py"
    
    if not validation_script.exists():
        print_colored("⚠️  Validation script not found, skipping...", Colors.YELLOW)
        return True
    
    try:
        # Run validation with timeout
        result = subprocess.run(
            [venv_python, str(validation_script)],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=120  # Increased timeout for comprehensive validation
        )
        
        if result.returncode == 0:
            print_colored("✅ System validation passed", Colors.GREEN)
            # Print summary if available
            if "VALIDATION SUMMARY" in result.stdout:
                lines = result.stdout.split("\n")
                for line in lines:
                    if "Errors:" in line or "Warnings:" in line or "Status:" in line:
                        print_colored(f"  {line.strip()}", Colors.CYAN)
            return True
        else:
            print_colored("⚠️  System validation found issues (non-critical)", Colors.YELLOW)
            # Print only summary, not full output
            if "VALIDATION SUMMARY" in result.stdout:
                lines = result.stdout.split("\n")
                summary_start = False
                for line in lines:
                    if "VALIDATION SUMMARY" in line:
                        summary_start = True
                    if summary_start:
                        if line.strip():
                            print_colored(f"  {line.strip()}", Colors.YELLOW)
            return True  # Continue anyway
    except subprocess.TimeoutExpired:
        print_colored("⚠️  Validation timed out, continuing...", Colors.YELLOW)
        return True
    except Exception as e:
        print_colored(f"⚠️  Validation error: {e}, continuing...", Colors.YELLOW)
        return True

def main():
    """Main launcher function."""
    print_colored("=" * 80, Colors.BOLD)
    print_colored("⚡ Quantum AI Cockpit — Universal Launcher", Colors.BOLD)
    print_colored("=" * 80, Colors.BOLD)
    
    # Detect OS
    os_type = detect_os()
    print_colored(f"🖥️  Detected OS: {os_type}", Colors.CYAN)
    
    # Get project root
    project_root = get_project_root()
    print_colored(f"📁 Project root: {project_root}", Colors.CYAN)
    
    # Find venv
    venv_path = find_venv()
    if venv_path:
        print_colored(f"🐍 Found virtual environment: {venv_path}", Colors.GREEN)
        venv_python = get_venv_python(venv_path)
    else:
        print_colored("⚠️  No virtual environment found, using system Python", Colors.YELLOW)
        venv_python = get_python_cmd()
    
    # Check dependencies
    deps_ok, missing = check_dependencies()
    if not deps_ok:
        print_colored(f"❌ Missing dependencies: {missing}", Colors.RED)
        print_colored("💡 Run: pip install -r requirements_full.txt", Colors.YELLOW)
        sys.exit(1)
    
    # Run validation
    run_validation(venv_python, project_root)
    
    # Launch backend
    backend_process = run_backend(venv_python, project_root)
    time.sleep(3)  # Give backend time to start
    
    # Wait for backend
    if not wait_for_server("http://localhost:8090/docs", timeout=30):
        print_colored("❌ Backend failed to start", Colors.RED)
        backend_process.terminate()
        sys.exit(1)
    
    # Launch frontend
    frontend_process = run_frontend(project_root)
    time.sleep(5)  # Give frontend time to start
    
    # Wait for frontend
    if not wait_for_server("http://localhost:5173", timeout=30):
        print_colored("⚠️  Frontend may not be ready yet, but continuing...", Colors.YELLOW)
    
    # Open browser
    open_browser("http://localhost:5173")
    
    print_colored("=" * 80, Colors.BOLD)
    print_colored("✅ Quantum AI Cockpit is running!", Colors.GREEN)
    print_colored("=" * 80, Colors.BOLD)
    print_colored("📊 Dashboard: http://localhost:5173", Colors.CYAN)
    print_colored("🔌 API Docs: http://localhost:8090/docs", Colors.CYAN)
    print_colored("", Colors.RESET)
    print_colored("Press Ctrl+C to stop all services", Colors.YELLOW)
    
    try:
        # Keep script running
        while True:
            time.sleep(1)
            # Check if processes are still alive
            if backend_process.poll() is not None:
                print_colored("❌ Backend process died", Colors.RED)
                break
            if frontend_process.poll() is not None:
                print_colored("❌ Frontend process died", Colors.RED)
                break
    except KeyboardInterrupt:
        print_colored("\n🛑 Shutting down...", Colors.YELLOW)
        backend_process.terminate()
        frontend_process.terminate()
        print_colored("✅ All services stopped", Colors.GREEN)

if __name__ == "__main__":
    main()

