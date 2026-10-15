#!/usr/bin/env python3
"""
MiniCPM Micro Web Server (Zero-Dependency)

Uses Python's standard library http.server to provide a low-overhead,
resource-friendly web interface with Server-Sent Events (SSE) streaming.
Consumes < 5MB RAM and requires no external web frameworks.
"""

import argparse
import json
import os
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from engine import MiniCPMEngine

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


class ChatRequestHandler(BaseHTTPRequestHandler):
    engine: MiniCPMEngine
    lock: threading.Lock

    def do_OPTIONS(self):
        """CORS preflight handling."""
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path in ["/", "/index.html"]:
            self._serve_file(os.path.join(STATIC_DIR, "index.html"), "text/html; charset=utf-8")
        elif path == "/api/status":
            self._send_json({
                "thinking_mode": self.engine.thinking_mode,
                "device": str(self.engine.model.device),
                "history_turns": len(self.engine.history) // 2,
            })
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/chat":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            try:
                data = json.loads(body) if body else {}
            except json.JSONDecodeError:
                self.send_error(HTTPStatus.BAD_REQUEST, "Invalid JSON")
                return

            prompt = data.get("prompt", "").strip()
            if not prompt:
                self.send_error(HTTPStatus.BAD_REQUEST, "Prompt is required")
                return

            # Stream via Server-Sent Events (SSE)
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            with self.lock:
                try:
                    for token in self.engine.generate_stream(prompt):
                        payload = f"data: {json.dumps({'text': token})}\n\n"
                        self.wfile.write(payload.encode("utf-8"))
                        self.wfile.flush()
                except Exception as e:
                    err_payload = f"data: {json.dumps({'error': str(e)})}\n\n"
                    self.wfile.write(err_payload.encode("utf-8"))
                    self.wfile.flush()

            done_payload = f"data: {json.dumps({'done': True})}\n\n"
            self.wfile.write(done_payload.encode("utf-8"))
            self.wfile.flush()

        elif path == "/api/clear":
            with self.lock:
                self.engine.clear_history()
            self._send_json({"status": "cleared"})

        elif path == "/api/mode":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            data = json.loads(body) if body else {}

            with self.lock:
                if "thinking_mode" in data:
                    self.engine.thinking_mode = bool(data["thinking_mode"])
                else:
                    self.engine.thinking_mode = not self.engine.thinking_mode
                current_mode = self.engine.thinking_mode

            self._send_json({"thinking_mode": current_mode})
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Endpoint not found")

    def _serve_file(self, filepath: str, content_type: str):
        if not os.path.exists(filepath):
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return
        with open(filepath, "rb") as f:
            content = f.read()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _send_json(self, data: dict):
        body = json.dumps(data).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        # Concise logging
        sys.stderr.write(f"[{self.log_date_time_string()}] {format % args}\n")


def run_web(host: str = "127.0.0.1", port: int = 8000, model_path: str = "openbmb/MiniCPM5-1B"):
    engine = MiniCPMEngine(model_path=model_path, thinking_mode=True)
    lock = threading.Lock()

    ChatRequestHandler.engine = engine
    ChatRequestHandler.lock = lock

    server = ThreadingHTTPServer((host, port), ChatRequestHandler)
    print(f"\n==================================================")
    print(f"  MiniCPM Ultra-Lightweight Web Server")
    print(f"  URL: http://{host}:{port}")
    print(f"  Overhead: Standard library only (no external RAM bloat)")
    print(f"==================================================\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniCPM Ultra-Lightweight Web Server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    parser.add_argument("--model", type=str, default="openbmb/MiniCPM5-1B", help="Model path/ID")
    args = parser.parse_args()

    run_web(host=args.host, port=args.port, model_path=args.model)
