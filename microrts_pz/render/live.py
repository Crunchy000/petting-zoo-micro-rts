# SPDX-License-Identifier: GPL-3.0-only
"""Live browser viewer: a tiny local HTTP server streaming frames over Server-Sent Events."""

from __future__ import annotations

import json
import os
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_VIEWER_HTML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "viewer.html")
_DATA_MARKER = "/*__MICRORTS_DATA__*/"


class LiveViewer:
    """Serve the canvas viewer on ``http://127.0.0.1:<port>/`` and push frames to it.

    Only binds to localhost. The browser keeps every frame of the current
    episode, so it can pause and scrub while the game keeps running.
    """

    def __init__(self, port: int = 0, open_browser: bool = True):
        self._cond = threading.Condition()
        self._meta: dict | None = None
        self._frames: list[str] = []
        self._episode = 0
        self._closed = False
        viewer = self

        with open(_VIEWER_HTML, encoding="utf-8") as f:
            page = f.read().replace(_DATA_MARKER, "window.MICRORTS_LIVE = true;", 1).encode("utf-8")

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # keep the training console quiet
                pass

            def do_GET(self):
                if self.path in ("/", "/index.html"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(page)))
                    self.end_headers()
                    self.wfile.write(page)
                elif self.path == "/events":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    viewer._stream(self.wfile)
                else:
                    self.send_error(404)

        self._server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}/"
        print(f"MicroRTS viewer: {self.url}")
        if open_browser and os.environ.get("MICRORTS_NO_BROWSER") is None:
            try:
                webbrowser.open(self.url)
            except Exception:
                pass

    def send(self, meta: dict, frame: dict | None) -> None:
        if frame is None:
            return
        with self._cond:
            if frame.get("t") == 0 or self._meta is None:
                self._meta = meta
                self._frames = []
                self._episode += 1
            self._frames.append(json.dumps(frame, separators=(",", ":")))
            self._cond.notify_all()

    def _stream(self, wfile) -> None:
        episode, sent = -1, 0
        try:
            while True:
                with self._cond:
                    while not self._closed and episode == self._episode and sent >= len(self._frames):
                        self._cond.wait(timeout=15)
                        if episode == self._episode and sent >= len(self._frames):
                            break  # heartbeat
                    if self._closed:
                        return
                    msgs = []
                    if episode != self._episode:
                        episode, sent = self._episode, 0
                        msgs.append("event: reset\ndata: " + json.dumps(self._meta, separators=(",", ":")) + "\n\n")
                    new = self._frames[sent:]
                    sent = len(self._frames)
                msgs.extend("data: " + f + "\n\n" for f in new)
                if not msgs:
                    msgs.append(": ping\n\n")
                wfile.write("".join(msgs).encode("utf-8"))
                wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            return

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()
        self._server.shutdown()
        self._server.server_close()
