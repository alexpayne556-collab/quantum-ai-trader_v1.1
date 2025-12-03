#!/usr/bin/env python3
"""
Quantum AI Trading System - Universal Launcher
Cross-platform system startup script
"""

import os
import sys
import platform
import subprocess
import time
import webbrowser
from pathlib import Path
from typing import Optional

def detect_os() -> str:
    """Detect operating system."""
    return platform.system().lower()

def find_venv() -> Optional[Path]:
    """Find virtual environment."""
    project_root = Path(__file__).parent.parent
    venv_paths = [
        project_root / "venv",
        project_root / ".venv",
        project_root / "quantum_venv",
    ]
    
    for venv_path in venv_paths:
        if venv_path.exists():
            activate = venv_path / ("Scripts" if platform.system() == "Windows" else "bin") / "activate"
            if activate.exists():
                return venv_path
    return None

def get_python_cmd(venv_path: Optional[Path]) -> str:
    """Get Python command."""
    if venv_path:
        if platform.system() == "Windows":
            return str(venv_path / "Scripts" / "python.exe")
        else:
            return str(venv_path / "bin" / "python3")
    return "python"

def start_backend(python_cmd: str) -> subprocess.Popen:
    """Start FastAPI backend."""
    print("🚀 Starting FastAPI backend...")
    backend_dir = Path(__file__).parent.parent / "backend"
    
    cmd = [
        python_cmd, "-m", "uvicorn",
        "main:app",
        "--host", "0.0.0.0",
        "--port", "8090",
        "--reload"
    ]
    
    if platform.system() == "Windows":
        process = subprocess.Popen(cmd, cwd=backend_dir, creationflags=subprocess.CREATE_NEW_CONSOLE)
    else:
        process = subprocess.Popen(cmd, cwd=backend_dir)
    
    return process

def start_frontend() -> subprocess.Popen:
    """Start Vite frontend."""
    print("🎨 Starting Vite frontend...")
    frontend_dir = Path(__file__).parent.parent / "frontend"
    
    # Install dependencies if needed
    if not (frontend_dir / "node_modules").exists():
        print("📦 Installing frontend dependencies...")
        subprocess.run(["npm", "install"], cwd=frontend_dir, check=True)
    
    cmd = ["npm", "run", "dev"]
    if platform.system() == "Windows":
        process = subprocess.Popen(cmd, cwd=frontend_dir, creationflags=subprocess.CREATE_NEW_CONSOLE)
    else:
        process = subprocess.Popen(cmd, cwd=frontend_dir)
    
    return process

def wait_for_server(url: str, timeout: int = 30) -> bool:
    """Wait for server to be ready."""
    import urllib.request
    print(f"⏳ Waiting for server at {url}...")
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            urllib.request.urlopen(url, timeout=2)
            print(f"✅ Server ready at {url}")
            return True
        except Exception:
            time.sleep(1)
    
    print(f"❌ Server did not start within {timeout} seconds")
    return False

def main():
    """Main launcher function."""
    print("=" * 80)
    print("⚡ Quantum AI Trading System — Universal Launcher")
    print("=" * 80)
    
    # Detect OS and find Python
    os_type = detect_os()
    print(f"🖥️  Detected OS: {os_type}")
    
    venv_path = find_venv()
    python_cmd = get_python_cmd(venv_path)
    
    if venv_path:
        print(f"🐍 Found virtual environment: {venv_path}")
    else:
        print("⚠️  No virtual environment found, using system Python")
    
    # Start backend
    backend_process = start_backend(python_cmd)
    time.sleep(3)
    
    # Wait for backend
    if not wait_for_server("http://localhost:8090/docs"):
        print("❌ Backend failed to start")
        backend_process.terminate()
        sys.exit(1)
    
    # Start frontend
    frontend_process = start_frontend()
    time.sleep(5)
    
    # Wait for frontend
    wait_for_server("http://localhost:5173", timeout=30)
    
    # Open browser
    print("🌐 Opening browser...")
    webbrowser.open("http://localhost:5173")
    
    print("=" * 80)
    print("✅ Quantum AI Trading System is running!")
    print("=" * 80)
    print("📊 Dashboard: http://localhost:5173")
    print("🔌 API Docs: http://localhost:8090/docs")
    print("")
    print("Press Ctrl+C to stop all services")
    
    try:
        while True:
            time.sleep(1)
            if backend_process.poll() is not None:
                print("❌ Backend process died")
                break
            if frontend_process.poll() is not None:
                print("❌ Frontend process died")
                break
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        backend_process.terminate()
        frontend_process.terminate()
        print("✅ All services stopped")

if __name__ == "__main__":
    main()
