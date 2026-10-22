"""Bounded localhost browser probe: pause one unchanged video stream.

All other GET/HEAD requests go to the existing local product server. Uploads and
other mutations are refused. Touch the run's release.flag to resume media; stop
this dedicated process after the browser check. This is not a product server.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, build_opener, ProxyHandler

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_ID = "match_9de7cb2d09eb46a0af0e34c3b0e6b818"
SOURCE = ROOT / "outputs/product/analyses" / ANALYSIS_ID / "source.mp4"
MEDIA_PATH = f"/api/analyses/{ANALYSIS_ID}/media/source"
PREFIX_BYTES = 1024 * 1024
UPSTREAM = "http://127.0.0.1:8000"
HOP_HEADERS = {"connection", "transfer-encoding", "keep-alive", "proxy-authenticate",
               "proxy-authorization", "te", "trailer", "upgrade"}


def mp4_boxes(path: Path) -> list[dict]:
    boxes = []
    size = path.stat().st_size
    with path.open("rb") as stream:
        offset = 0
        while offset < size:
            stream.seek(offset)
            header = stream.read(8)
            if len(header) != 8:
                raise ValueError("Truncated MP4 box header")
            length, kind = struct.unpack(">I4s", header)
            if length == 1:
                extended = stream.read(8)
                if len(extended) != 8:
                    raise ValueError("Truncated extended MP4 box header")
                length = struct.unpack(">Q", extended)[0]
            elif length == 0:
                length = size - offset
            if length < (16 if header[:4] == b"\x00\x00\x00\x01" else 8) or offset + length > size:
                raise ValueError("Invalid MP4 box length")
            boxes.append({"kind": kind.decode("ascii"), "offset": offset, "length": length})
            offset += length
    return boxes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8010)
    args = parser.parse_args()
    directory = args.directory.resolve()
    evidence_root = (ROOT / "artifacts/validation/product_execution").resolve()
    if not directory.is_relative_to(evidence_root) or directory == evidence_root:
        raise ValueError("Probe evidence must be in its own workspace validation directory")
    # Exclusive directory creation prevents accidental reuse of a released run.
    directory.mkdir(parents=True, exist_ok=False)
    release = directory / "release.flag"
    journal = directory / "stream.jsonl"
    lock = threading.Lock()
    deadline = time.monotonic() + 1800
    source_size = SOURCE.stat().st_size
    source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    boxes = mp4_boxes(SOURCE)
    moov = next(box for box in boxes if box["kind"] == "moov")
    mdat = next(box for box in boxes if box["kind"] == "mdat")
    if moov["offset"] >= mdat["offset"] or moov["offset"] + moov["length"] > PREFIX_BYTES:
        raise ValueError("Probe needs existing fast-start metadata inside the unpaused prefix")

    def record(event: str, **details: object) -> None:
        with lock, journal.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"at": datetime.now(timezone.utc).isoformat(),
                                     "event": event, **details}) + "\n")

    metadata = {"analysisId": ANALYSIS_ID, "source": str(SOURCE.relative_to(ROOT)),
                "sourceSha256": source_hash, "sourceBytes": source_size, "mp4Boxes": boxes,
                "prefixBytes": PREFIX_BYTES, "upstream": UPSTREAM,
                "listen": f"http://127.0.0.1:{args.port}", "lifetimeSeconds": 1800,
                "qualificationEvidence": False}
    (directory / "protocol.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    opener = build_opener(ProxyHandler({}))

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args: object) -> None:
            pass

        def do_POST(self) -> None:
            self.send_error(405, "This browser probe is read-only")

        do_PUT = do_POST
        do_PATCH = do_POST
        do_DELETE = do_POST

        def do_HEAD(self) -> None:
            self.respond(head=True)

        def do_GET(self) -> None:
            self.respond(head=False)

        def respond(self, *, head: bool) -> None:
            parsed = urlsplit(self.path)
            if parsed.scheme or parsed.netloc or not parsed.path.startswith("/"):
                self.send_error(400)
                return
            if parsed.path == MEDIA_PATH:
                self.media(head=head)
                return
            if parsed.path == "/_buffer_test/status":
                body = json.dumps({"released": release.exists(), "secondsRemaining":
                                   max(0, deadline - time.monotonic())}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                if not head:
                    self.wfile.write(body)
                return
            headers = {key: self.headers[key] for key in ("Range", "Accept", "If-None-Match")
                       if self.headers.get(key)}
            request = Request(UPSTREAM + self.path, headers=headers,
                              method="HEAD" if head else "GET")
            try:
                response = opener.open(request, timeout=30)
            except HTTPError as error:
                response = error
            except OSError:
                self.send_error(502, "Local product server unavailable")
                return
            with response:
                self.send_response(response.status)
                for key, value in response.headers.items():
                    if key.lower() not in HOP_HEADERS | {"cache-control"}:
                        self.send_header(key, value)
                self.send_header("Cache-Control", "no-store")
                # Force close also safely delimits upstream error responses that
                # have no Content-Length; transfer-encoding is not forwarded.
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True
                if not head:
                    try:
                        while chunk := response.read(65536):
                            self.wfile.write(chunk)
                    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                        pass

        def media(self, *, head: bool) -> None:
            raw_range = self.headers.get("Range")
            start, end = 0, source_size - 1
            if raw_range:
                match = re.fullmatch(r"bytes=(\d*)-(\d*)", raw_range)
                if not match or not any(match.groups()):
                    self.send_error(416)
                    return
                first, last = match.groups()
                if first:
                    start = int(first)
                    end = min(int(last), end) if last else end
                else:
                    start = max(0, source_size - int(last))
                if start > end or start >= source_size:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{source_size}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
            self.send_response(206 if raw_range else 200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            if raw_range:
                self.send_header("Content-Range", f"bytes {start}-{end}/{source_size}")
            self.end_headers()
            self.close_connection = True
            if head:
                return
            sent = 0
            held = False
            record("media_start", range=raw_range, start=start, end=end)
            try:
                with SOURCE.open("rb") as stream:
                    stream.seek(start)
                    position = start
                    while position <= end:
                        if position >= PREFIX_BYTES and not release.exists():
                            if not held:
                                record("stream_held", position=position, sent=sent)
                                held = True
                            if time.monotonic() >= deadline:
                                record("stream_expired", sent=sent)
                                return
                            time.sleep(0.1)
                            continue
                        if held:
                            record("stream_released", position=position, sent=sent)
                            held = False
                        limit = end + 1 if release.exists() else min(PREFIX_BYTES, end + 1)
                        chunk = stream.read(min(65536, limit - position))
                        if not chunk:
                            raise OSError("Unexpected source end")
                        self.wfile.write(chunk)
                        self.wfile.flush()
                        position += len(chunk)
                        sent += len(chunk)
                record("media_complete", start=start, end=end, sent=sent)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                record("media_client_closed", sent=sent)

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.daemon_threads = True
    server.timeout = 0.5
    print(json.dumps(metadata), flush=True)
    try:
        while time.monotonic() < deadline:
            server.handle_request()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        after = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
        record("server_stopped", sourceSha256After=after, sourceUnchanged=after == source_hash)
        if after != source_hash:
            raise RuntimeError("Source changed during read-only browser probe")


if __name__ == "__main__":
    main()
