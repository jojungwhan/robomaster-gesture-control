"""Non-blocking bridge to a low-priority, narration-only YOLOE worker."""

from __future__ import annotations

import base64
import json
from pathlib import Path
import queue
import subprocess
import threading
import time
from typing import Optional, Tuple


EXPANDED_LABEL_ALIASES = {
    "armchair": "chair",
    "bean bag chair": "chair",
    "chair": "chair",
    "computer chair": "chair",
    "folding chair": "chair",
    "office chair": "chair",
    "rocking chair": "chair",
    "swivel chair": "chair",
    "computer desk": "desk",
    "information desk": "desk",
    "office desk": "desk",
    "writing desk": "desk",
    "cocktail table": "table",
    "dinning table": "table",
    "glass table": "table",
    "kitchen table": "table",
    "picnic table": "table",
    "round table": "table",
    "side table": "table",
    "table": "table",
    "bookcase": "bookshelf",
    "bookshelf": "bookshelf",
    "shelf": "bookshelf",
    "bathroom cabinet": "cabinet",
    "cabinet": "cabinet",
    "cabinetry": "cabinet",
    "file cabinet": "cabinet",
    "kitchen cabinet": "cabinet",
    "side cabinet": "cabinet",
    "tv cabinet": "cabinet",
    "door": "door",
    "doorway": "door",
    "glass door": "door",
    "screen door": "door",
    "glass window": "window",
    "kitchen window": "window",
    "office window": "window",
    "window": "window",
    "computer monitor": "monitor",
    "monitor": "monitor",
    "couch": "couch",
    "bed": "bed",
    "nightstand": "nightstand",
    "stool": "stool",
    "drawer": "drawer",
    "lamp": "lamp",
    "mirror": "mirror",
}


def _dictionary_iou(first, second) -> float:
    first_box = first["box"]
    second_box = second["box"]
    left = max(float(first_box[0]), float(second_box[0]))
    top = max(float(first_box[1]), float(second_box[1]))
    right = min(float(first_box[2]), float(second_box[2]))
    bottom = min(float(first_box[3]), float(second_box[3]))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    if intersection <= 0.0:
        return 0.0
    first_area = max(0.0, float(first_box[2]) - float(first_box[0])) * max(
        0.0, float(first_box[3]) - float(first_box[1])
    )
    second_area = max(0.0, float(second_box[2]) - float(second_box[0])) * max(
        0.0, float(second_box[3]) - float(second_box[1])
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def confirmed_expanded_detections(previous, current, overlap_threshold=0.20):
    """Keep labels independently observed in two successive YOLOE scans."""
    if not 0.0 <= overlap_threshold <= 1.0:
        raise ValueError("overlap threshold must be between 0 and 1")
    confirmed = []
    used_previous = set()
    for item in current:
        matches = [
            (_dictionary_iou(before, item), index)
            for index, before in enumerate(previous)
            if index not in used_previous
            and str(item["label"]).casefold()
            == str(before["label"]).casefold()
        ]
        if not matches:
            continue
        overlap, index = max(matches)
        if overlap >= overlap_threshold:
            confirmed.append(item)
            used_previous.add(index)
    return tuple(confirmed)


class ExpandedSceneRecognizer:
    """Send occasional frames to YOLOE without blocking the control loop."""

    def __init__(
        self,
        python_executable: Path,
        worker: Path,
        model: Path,
        confidence: float = 0.30,
        image_size: int = 320,
        interval_s: float = 1.5,
    ):
        self.python_executable = Path(python_executable)
        self.worker = Path(worker)
        self.model = Path(model)
        self.confidence = confidence
        self.image_size = image_size
        self.interval_s = interval_s
        self._process = None
        self._messages = queue.Queue()
        self._frames = queue.Queue(maxsize=1)
        self._threads = []
        self._stderr = []
        self._error = None  # type: Optional[str]
        self._closed = False
        self._next_submit_at = float("-inf")
        self._sequence = 0
        self._latest = None
        self._latest_lock = threading.Lock()

    def start(self, timeout_s: float = 45.0) -> None:
        for path, description in (
            (self.python_executable, "YOLO Python executable"),
            (self.worker, "expanded-scene worker"),
            (self.model, "expanded-scene YOLOE model"),
        ):
            if not path.is_file():
                raise RuntimeError("{} not found: {}".format(description, path))
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self._process = subprocess.Popen(
            (
                str(self.python_executable),
                str(self.worker),
                "--model",
                str(self.model),
                "--confidence",
                str(self.confidence),
                "--image-size",
                str(self.image_size),
            ),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=creation_flags,
        )
        self._threads = [
            threading.Thread(target=self._read_stdout, daemon=True),
            threading.Thread(target=self._read_stderr, daemon=True),
        ]
        for thread in self._threads:
            thread.start()
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                message = self._messages.get(timeout=0.1)
            except queue.Empty:
                if self._process.poll() is not None:
                    break
                continue
            if message.get("event") == "ready":
                sender = threading.Thread(target=self._send_loop, daemon=True)
                sender.start()
                self._threads.append(sender)
                return
            if message.get("event") == "error":
                error = str(message.get("message") or "YOLOE startup failed")
                self.close()
                raise RuntimeError(error)
        error = self.error or "expanded-scene worker did not become ready"
        self.close()
        raise RuntimeError(error)

    def _read_stdout(self) -> None:
        assert self._process is not None and self._process.stdout is not None
        for line in self._process.stdout:
            try:
                message = json.loads(line)
            except (TypeError, ValueError):
                message = {"event": "error", "message": "Invalid YOLOE event"}
            if message.get("event") == "error":
                self._error = str(message.get("message") or "YOLOE worker error")
            elif message.get("event") == "detections":
                with self._latest_lock:
                    self._latest = (
                        time.monotonic(),
                        int(message.get("sequence", 0)),
                        tuple(message.get("detections") or ()),
                    )
            if message.get("event") in ("ready", "error"):
                self._messages.put(message)

    def _read_stderr(self) -> None:
        assert self._process is not None and self._process.stderr is not None
        for line in self._process.stderr:
            self._stderr.append(line.rstrip())

    def _send_loop(self) -> None:
        try:
            import cv2

            while True:
                item = self._frames.get()
                if item is None:
                    self._write({"event": "close"})
                    return
                sequence, frame = item
                success, encoded = cv2.imencode(
                    ".jpg", frame, (cv2.IMWRITE_JPEG_QUALITY, 78)
                )
                if not success:
                    raise RuntimeError("could not encode expanded-scene frame")
                self._write(
                    {
                        "event": "frame",
                        "sequence": sequence,
                        "jpeg": base64.b64encode(encoded.tobytes()).decode("ascii"),
                    }
                )
        except Exception as exc:
            if not self._closed:
                self._error = "expanded-scene sender failed: {}".format(exc)

    def _write(self, message) -> None:
        if self._process is None or self._process.stdin is None:
            raise RuntimeError("expanded-scene worker is not running")
        self._process.stdin.write(json.dumps(message) + "\n")
        self._process.stdin.flush()

    def submit(self, frame, now_s: Optional[float] = None) -> bool:
        now_s = time.monotonic() if now_s is None else now_s
        if self.error is not None or self._closed or self._process is None:
            return False
        if now_s < self._next_submit_at:
            return False
        self._next_submit_at = now_s + self.interval_s
        self._sequence += 1
        try:
            self._frames.get_nowait()
        except queue.Empty:
            pass
        try:
            self._frames.put_nowait((self._sequence, frame.copy()))
        except queue.Full:
            return False
        return True

    def detections(self, max_age_s: float = 4.0) -> Tuple[dict, ...]:
        with self._latest_lock:
            latest = self._latest
        if latest is None or time.monotonic() - latest[0] > max_age_s:
            return ()
        return latest[2]

    @property
    def has_confirmed_scan(self) -> bool:
        with self._latest_lock:
            latest = self._latest
        return latest is not None and latest[1] >= 2

    @property
    def error(self) -> Optional[str]:
        if self._error:
            return self._error
        if self._process is not None and self._process.poll() not in (None, 0):
            stderr = "\n".join(self._stderr).strip()
            return stderr or "expanded-scene worker exited"
        return None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._frames.get_nowait()
        except queue.Empty:
            pass
        try:
            self._frames.put_nowait(None)
        except queue.Full:
            pass
        if self._process is None:
            return
        try:
            self._process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            self._process.terminate()
            try:
                self._process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=2.0)
        for stream in (self._process.stdin, self._process.stdout, self._process.stderr):
            if stream is not None:
                stream.close()
        self._process = None
