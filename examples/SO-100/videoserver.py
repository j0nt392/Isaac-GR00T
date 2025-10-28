# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lightweight MJPEG server used to preview robot camera streams in a browser.

This module provides:
- A thread-safe publisher that stores the latest JPEG per camera key
- An HTTP server that exposes:
  - "/": a minimal HTML index that shows all cameras
  - "/stream/<camera_key>": an MJPEG endpoint per camera

The server is designed to run in-process with your eval loop. Your loop calls
`MJPEGServer.update_frame(key, frame)` when a new RGB frame (H, W, 3 uint8) is
available. Clients connect to the endpoints to view the live stream.
"""
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

import cv2
import numpy as np
import os
import shutil
import json
import shlex
import subprocess
import sys
from datetime import datetime


class _FramePublisher:
    """Stores and serves the latest JPEG-encoded frame per camera key.

    - `update_frame` encodes an RGB frame to JPEG and publishes it
    - `wait_latest_jpeg` lets stream handlers wait briefly for the newest JPEG
    """
    def __init__(self, camera_keys, jpeg_quality: int = 80):
        # Map camera key -> latest JPEG bytes
        self.latest_jpeg_by_key = {key: None for key in camera_keys}
        # Per-key condition variable to notify waiting HTTP handlers of updates
        self.conditions_by_key = {key: threading.Condition() for key in camera_keys}
        self.jpeg_quality = int(jpeg_quality)

    def update_frame(self, camera_key: str, frame: np.ndarray):
        """Publish a new frame for a camera key after JPEG encoding.

        Expects RGB/BGR uint8 frame of shape (H, W, 3). The encoded bytes are
        cached and made available to all connected clients.
        """
        if camera_key not in self.latest_jpeg_by_key:
            return
        if not isinstance(frame, np.ndarray):
            return
        try:
            ok, buf = cv2.imencode(
                ".jpg",
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
            )
            if not ok:
                return
            jpeg_bytes = buf.tobytes()
        except Exception:
            return
        cond = self.conditions_by_key[camera_key]
        with cond:
            # Swap in the newest JPEG and wake any waiting client streams
            self.latest_jpeg_by_key[camera_key] = jpeg_bytes
            cond.notify_all()

    def wait_latest_jpeg(self, camera_key: str, timeout: float = 1.0):
        """Block briefly for a new frame and return the latest JPEG (or None)."""
        if camera_key not in self.latest_jpeg_by_key:
            return None
        cond = self.conditions_by_key[camera_key]
        with cond:
            cond.wait(timeout=timeout)
            return self.latest_jpeg_by_key[camera_key]


class _ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Simple threaded HTTP server; each request handled in a daemon thread."""
    daemon_threads = True


class _ProcessManager:
    """Manage a background subprocess for eval_lerobot.py.

    Provides start/stop/status with a ring buffer of recent logs.
    """

    def __init__(self, eval_script_path: str):
        self.eval_script_path = eval_script_path
        self.proc: subprocess.Popen | None = None
        self.args_str: str | None = None
        self.started_at: str | None = None
        self.pid: int | None = None
        self._log = []
        self._log_max = 400
        self.log_file_path: str | None = None
        self._log_file = None

    def _reader(self, pipe, tag: str):
        try:
            for line in iter(pipe.readline, ""):
                entry = f"[{tag}] {line.rstrip()}"
                self._log.append(entry)
                if len(self._log) > self._log_max:
                    self._log = self._log[-self._log_max :]
                if self._log_file is not None:
                    try:
                        self._log_file.write(entry + "\n")
                        self._log_file.flush()
                    except Exception:
                        pass
        except Exception:
            pass

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, args_str: str):
        if self.is_running():
            return False, "already running"
        # Build command prefixed with conda activation for 'gr00t' env
        activate = (
            "conda run -n gr00t"
            if shutil.which("conda")
            else None
        )
        if activate is not None:
            cmd = shlex.split(activate) + [sys.executable, self.eval_script_path]
        else:
            cmd = [sys.executable, self.eval_script_path]
        if args_str:
            cmd += shlex.split(args_str)
        try:
            # Prepare logfile in the same folder as the eval script
            try:
                ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                base_dir = os.path.dirname(self.eval_script_path)
                self.log_file_path = os.path.join(base_dir, f"eval_{ts}.log")
                self._log_file = open(self.log_file_path, "a", encoding="utf-8")
                self._log_file.write("# Command: " + " ".join(cmd) + "\n")
                self._log_file.flush()
            except Exception:
                self.log_file_path = None
                self._log_file = None

            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )
            self.args_str = args_str
            self.started_at = datetime.utcnow().isoformat() + "Z"
            self.pid = self.proc.pid
            # Readers
            if self.proc.stdout is not None:
                threading.Thread(target=self._reader, args=(self.proc.stdout, "stdout"), daemon=True).start()
            if self.proc.stderr is not None:
                threading.Thread(target=self._reader, args=(self.proc.stderr, "stderr"), daemon=True).start()
            return True, None
        except Exception as e:
            self.proc = None
            if self._log_file is not None:
                try:
                    self._log_file.close()
                except Exception:
                    pass
                self._log_file = None
            return False, str(e)

    def stop(self):
        if not self.is_running():
            return False, "not running"
        try:
            assert self.proc is not None
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()
            self.proc = None
            if self._log_file is not None:
                try:
                    self._log_file.write("# Stopped\n")
                    self._log_file.flush()
                    self._log_file.close()
                except Exception:
                    pass
                self._log_file = None
            return True, None
        except Exception as e:
            return False, str(e)

    def status(self) -> dict:
        return {
            "running": self.is_running(),
            "pid": self.pid if self.is_running() else None,
            "started_at": self.started_at if self.is_running() else None,
            "args": self.args_str if self.is_running() else None,
            "last_logs": self._log[-20:],
            "log_file": self.log_file_path,
        }


class _MJPEGRequestHandler(BaseHTTPRequestHandler):
    """HTTP handler that serves index and MJPEG endpoints.

    The enclosing server sets:
      - `camera_keys`: list of camera keys to show on the index page
      - `publisher`: `_FramePublisher` that provides JPEG bytes per key
    """
    server_version = "MJPEGServer/0.1"

    def do_GET(self):
        # Route: index page or per-camera MJPEG stream
        if self.path in ("/", "/index.html"):
            return self._handle_index()
        if self.path.startswith("/stream/"):
            camera_key = self.path.split("/stream/", 1)[1]
            return self._handle_stream(camera_key)
        if self.path == "/api/status":
            return self._handle_status()
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        if self.path == "/api/start":
            return self._handle_start()
        if self.path == "/api/stop":
            return self._handle_stop()
        self.send_response(404)
        self.end_headers()

    def _handle_index(self):
        """Return a minimal, responsive HTML page listing all cameras."""
        keys = getattr(self.server, "camera_keys", [])
        cards = "\n".join(
            [
                (
                    f"""
                <div class="card">
                  <div class="cam-wrap">
                    <span class="badge">connecting</span>
                    <img class="cam-img" src="/stream/{k}" alt="{k}" />
                  </div>
                  <h3>{k}</h3>
                </div>
                """
                )
                for k in keys
            ]
        )
        # Avoid curly braces in f-string; build default args without --robot.cameras example
        default_args = (
            f"--robot.type=so100_follower --robot.port=/dev/ttyACM0 --robot.id=lil_guy "
            f"--serve_stream=true --stream_host=0.0.0.0 --stream_port={self.server.server_port} "
            f"--lang_instruction=\"Grab markers and place into pen holder.\""
        )

        page = f"""<!doctype html>
            <html lang="en">
            <head>
            <meta charset="utf-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1" />
            <title>Robot Camera Streams</title>
            <style>
                :root {{
                --bg: #0b1020;
                --card: #0e162b;
                --text: #e5e7eb;
                --muted: #9aa3b2;
                --accent: #60a5fa;
                --border: #24324a;
                }}
                * {{ box-sizing: border-box; }}
                body {{
                margin: 0;
                background: var(--bg);
                color: var(--text);
                font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, Apple Color Emoji, Segoe UI Emoji;
                }}
                header {{
                border-bottom: 1px solid var(--border);
                background: linear-gradient(180deg, rgba(96,165,250,0.06), rgba(96,165,250,0));
                }}
                header .inner {{
                max-width: 1400px; margin: 0 auto; padding: 16px;
                display: flex; align-items: center; justify-content: space-between;
                }}
                h1 {{ font-size: 18px; margin: 0; letter-spacing: 0.3px; }}
                .sub {{ color: var(--muted); font-size: 13px; }}
                .container {{ max-width: 1400px; margin: 0 auto; padding: 16px; }}
                .grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
                gap: 16px;
                }}
                .card {{
                background: var(--card);
                border: 1px solid var(--border);
                border-radius: 12px;
                box-shadow: 0 10px 25px rgba(0,0,0,0.25);
                overflow: hidden;
                padding: 10px 10px 12px 10px;
                }}
                .card h3 {{
                margin: 10px 4px 0 4px;
                font-weight: 600; letter-spacing: 0.2px;
                }}
                .cam-wrap {{ position: relative; border-radius: 10px; overflow: hidden; border: 1px solid var(--border); }}
                .cam-img {{ width: 100%; height: auto; display: block; background: #0b1020; aspect-ratio: 16 / 9; object-fit: contain; }}
                .badge {{
                position: absolute; top: 8px; right: 8px;
                background: #374151; color: #e5e7eb;
                padding: 4px 8px; font-size: 12px; border-radius: 9999px;
                border: 1px solid rgba(255,255,255,0.08);
                }}
                .card.connected .badge {{ background: #16a34a; color: #fff; border-color: rgba(0,0,0,0.2); }}
                footer {{ text-align: center; padding: 8px 16px 24px; color: var(--muted); font-size: 12px; }}
                a {{ color: var(--accent); text-decoration: none; }}
                /* Controls UI */
                .controls {{
                background: var(--card);
                border: 1px solid var(--border);
                border-radius: 12px;
                padding: 12px;
                margin-bottom: 16px;
                }}
                .controls h2 {{ margin: 0 0 8px 0; font-size: 16px; }}
                .row {{ display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }}
                .row input[type=text], .row textarea {{
                width: 100%; max-width: 940px; background: #0b1426; color: var(--text);
                border: 1px solid var(--border); border-radius: 8px; padding: 8px;
                font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
                }}
                .row button {{
                background: #2563eb; color: white; border: 1px solid #1d4ed8; border-radius: 8px;
                padding: 8px 12px; cursor: pointer;
                }}
                .row button.secondary {{ background: #374151; border-color: #374151; }}
                .status {{ color: var(--muted); font-size: 13px; }}
            </style>
            </head>
            <body>
            <header>
                <div class="inner">
                <h1>Robot Camera Streams</h1>
                <div class="sub">MJPEG preview · {len(keys)} camera(s)</div>
                </div>
            </header>
            <div class="container">
                <div class="controls">
                <h2>Eval control</h2>
                <div class="row">
                    <button id="start">Start eval</button>
                    <button id="stop" class="secondary">Stop eval</button>
                    <span class="status" id="status">status: unknown</span>
                </div>
                </div>
                <div class="grid">
                {cards}
                </div>
            </div>
            <footer>
                Tip: open an individual stream via <code>/stream/&lt;camera_key&gt;</code>
            </footer>
            <script>
                async function getStatus() {{
                try {{
                    const r = await fetch('/api/status');
                    const j = await r.json();
                    return j;
                }} catch (e) {{ return {{error: String(e)}}; }}
                }}
                async function setStatusText() {{
                const s = await getStatus();
                const el = document.getElementById('status');
                el.textContent = 'status: ' + (s.running ? ('running (pid ' + s.pid + ')') : 'stopped');
                if (s.started_at) el.textContent += ' · started_at: ' + s.started_at;
                if (s.args) el.textContent += ' · args: ' + s.args;
                }}
                async function startEval() {{
                const r = await fetch('/api/start', {{ method: 'POST', headers: {{'Content-Type':'application/json'}}, body: '{{}}' }});
                const j = await r.json();
                await setStatusText();
                alert(j.ok ? 'Started' : ('Failed: ' + (j.error||'unknown')));
                }}
                async function stopEval() {{
                const r = await fetch('/api/stop', {{ method: 'POST' }});
                const j = await r.json();
                await setStatusText();
                alert(j.ok ? 'Stopped' : ('Failed: ' + (j.error||'unknown')));
                }}
                document.getElementById('start').addEventListener('click', startEval);
                document.getElementById('stop').addEventListener('click', stopEval);
                setStatusText();
                document.querySelectorAll('.cam-img').forEach(function(img) {{
                var card = img.closest('.card');
                function markConnected() {{ card && card.classList.add('connected'); }}
                function markDisconnected() {{ card && card.classList.remove('connected'); }}
                img.addEventListener('load', markConnected);
                img.addEventListener('error', markDisconnected);
                // Some browsers only fire load once for MJPEG; best-effort indicator.
                setTimeout(markConnected, 1000);
                }});
            </script>
            </body>
            </html>
            """
        html_bytes = page.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(html_bytes)))
        self.end_headers()
        self.wfile.write(html_bytes)

    def _handle_stream(self, camera_key: str):
        """Send a continuous multipart/x-mixed-replace MJPEG stream."""
        publisher: _FramePublisher = getattr(self.server, "publisher", None)
        if publisher is None:
            self.send_response(503)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header(
            "Content-Type",
            "multipart/x-mixed-replace; boundary=frame",
        )
        self.end_headers()
        try:
            # Keep writing JPEG parts separated by the boundary token
            while True:
                jpeg = publisher.wait_latest_jpeg(camera_key, timeout=1.0)
                if not jpeg:
                    continue
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("utf-8"))
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
        except Exception:
            # Client disconnected or network error; end this request handler
            pass

    def _json(self, code: int, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict:
        try:
            length = int(self.headers.get("Content-Length", "0") or "0")
        except Exception:
            length = 0
        raw = self.rfile.read(length) if length > 0 else b""
        try:
            return json.loads(raw.decode("utf-8") or "{}")
        except Exception:
            return {}

    def _handle_status(self):
        pm: _ProcessManager | None = getattr(self.server, "process_manager", None)
        status = pm.status() if pm is not None else {"running": False}
        return self._json(200, status)

    def _handle_start(self):
        pm: _ProcessManager | None = getattr(self.server, "process_manager", None)
        if pm is None:
            return self._json(500, {"ok": False, "error": "process manager missing"})
        data = self._read_json_body()
        args_str = data.get("args", "")
        if not args_str or not str(args_str).strip():
            # Default arguments to run eval when clicking Start (no user input required)
            camera_spec = (
                '{ wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fourcc: "MJPG", fps: 30}, '
                'front: {type: opencv, index_or_path: 4, width: 640, height: 480, fourcc: "MJPG", fps: 30} }'
            )
            # Note: adjust robot.port, camera indices and policy_host as needed
            args_parts = [
                "--robot.type=so100_follower",
                "--robot.port=/dev/ttyACM0",
                "--robot.id=my_awesome_follower_arm",
                f"--robot.cameras=\"{camera_spec}\"",
                "--policy_host=13.61.26.221",
                "--policy_port=8080",
                "--lang_instruction=\"Pick X and place it in the bottom left box.\"",
                "--serve_stream=true",
                "--stream_host=0.0.0.0",
                f"--stream_port={self.server.server_port}",
            ]
            args_str = " ".join(args_parts)
        ok, err = pm.start(args_str)
        if ok:
            return self._json(200, {"ok": True})
        return self._json(400, {"ok": False, "error": err or "failed"})

    def _handle_stop(self):
        pm: _ProcessManager | None = getattr(self.server, "process_manager", None)
        if pm is None:
            return self._json(500, {"ok": False, "error": "process manager missing"})
        ok, err = pm.stop()
        if ok:
            return self._json(200, {"ok": True})
        return self._json(400, {"ok": False, "error": err or "failed"})


class MJPEGServer:
    """Facade class wiring publisher and HTTP server.

    Usage:
        server = MJPEGServer(host, port, camera_keys)
        server.update_frame("front", frame)  # frame is np.ndarray (H, W, 3) uint8
    """
    def __init__(self, host: str, port: int, camera_keys):
        # Shared publisher caches the latest JPEG per camera
        self.publisher = _FramePublisher(camera_keys)
        # Threaded HTTP server serving index and stream endpoints
        self.server = _ThreadedHTTPServer((host, int(port)), _MJPEGRequestHandler)
        self.server.camera_keys = list(camera_keys)
        self.server.publisher = self.publisher
        # Attach a process manager for controlling eval from the UI/API
        eval_path = os.path.join(os.path.dirname(__file__), "eval_lerobot.py")
        self.server.process_manager = _ProcessManager(eval_path)
        # Run the HTTP server in the background
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def update_frame(self, camera_key: str, frame: np.ndarray):
        """Publish or replace the latest frame for a camera key."""
        self.publisher.update_frame(camera_key, frame)

    def shutdown(self):
        """Shut down the HTTP server and release the socket."""
        try:
            self.server.shutdown()
        finally:
            self.server.server_close()


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Demo MJPEG server with synthetic frames")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8008, help="Bind port")
    parser.add_argument(
        "--cameras",
        type=str,
        default="front,wrist",
        help="Comma-separated camera keys",
    )
    parser.add_argument("--fps", type=float, default=15.0, help="Frames per second")
    parser.add_argument("--width", type=int, default=640, help="Frame width")
    parser.add_argument("--height", type=int, default=480, help="Frame height")
    parser.add_argument(
        "--pattern",
        type=str,
        default="text",
        choices=["text", "color", "noise"],
        help="Synthetic frame pattern",
    )
    args = parser.parse_args()

    camera_keys = [k.strip() for k in args.cameras.split(",") if k.strip()]
    server = MJPEGServer(args.host, args.port, camera_keys)
    print(f"Demo MJPEG server at http://{args.host}:{args.port}")

    def make_frame(t: float, key: str) -> np.ndarray:
        h, w = int(args.height), int(args.width)
        if args.pattern == "noise":
            frame = (np.random.rand(h, w, 3) * 255).astype(np.uint8)
        elif args.pattern == "color":
            # Moving color gradient
            x = np.linspace(0, 1, w, dtype=np.float32)
            y = np.linspace(0, 1, h, dtype=np.float32)
            xx, yy = np.meshgrid(x, y)
            r = ((xx + 0.3 * np.sin(t)) * 255) % 255
            g = ((yy + 0.3 * np.cos(t)) * 255) % 255
            b = (((xx + yy) / 2 + 0.3 * np.sin(0.5 * t)) * 255) % 255
            frame = np.stack([b, g, r], axis=-1).astype(np.uint8)
        else:
            frame = np.zeros((h, w, 3), dtype=np.uint8)
        # Overlay text
        ts = time.strftime("%H:%M:%S")
        cv2.putText(frame, f"{key}  {ts}", (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"{args.pattern} demo", (16, h - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 200, 0), 2, cv2.LINE_AA)
        return frame

    try:
        period = 1.0 / max(1e-6, args.fps)
        t0 = time.time()
        while True:
            t = time.time() - t0
            for k in camera_keys:
                frame = make_frame(t, k)
                server.update_frame(k, frame)
            time.sleep(period)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()


