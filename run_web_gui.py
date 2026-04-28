#!/usr/bin/env python3
"""
ML Experiments Web GUI
──────────────────────
Run as a web server or as a desktop application using pywebview.

Usage:
    # Web server (default 0.0.0.0:3004)
    python run_web_gui.py
    python run_web_gui.py --host 127.0.0.1 --port 8080

    # Desktop app via pywebview
    python run_web_gui.py --desktop

    # Both (web server + pywebview window)
    python run_web_gui.py --desktop --host 0.0.0.0 --port 3004
"""

import argparse
import os
import sys
import threading

# Ensure project root is on sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Load environment variables from .env (best-effort)
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(BASE_DIR, ".env"), override=False)
except ImportError:
    pass


def parse_args():
    parser = argparse.ArgumentParser(description="ML Experiments Web GUI")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=3004, help="Port number (default: 3004)")
    parser.add_argument("--desktop", action="store_true", help="Launch as desktop app using pywebview")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode")
    return parser.parse_args()


def run_flask_server(host: str, port: int, debug: bool = False):
    """Start the Flask web server."""
    from web_gui.app import app
    app.run(host=host, port=port, debug=debug, threaded=True)


def run_desktop(host: str, port: int):
    """Launch pywebview desktop window."""
    import webview as pywebview

    url = f"http://{host}:{port}"

    def _start_server():
        run_flask_server(host, port, debug=False)

    # Start Flask in a background thread
    server_thread = threading.Thread(target=_start_server, daemon=True)
    server_thread.start()

    # Wait for server to be ready
    import time
    time.sleep(1)

    # Create pywebview window
    window = pywebview.create_window(
        title="ML Experiments",
        url=url,
        width=1400,
        height=900,
        resizable=True,
        text_select=True,
        zoomable=True,
    )
    pywebview.start(debug=False)


def main():
    args = parse_args()

    if args.desktop:
        run_desktop(args.host, args.port)
    else:
        print(f"🌐 Starting web server at http://{args.host}:{args.port}")
        print("   Press Ctrl+C to stop.")
        run_flask_server(args.host, args.port, debug=args.debug)


if __name__ == "__main__":
    main()
