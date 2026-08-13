"""English scene descriptions and a non-blocking Piper speech bridge."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import queue
import subprocess
import threading
import time
from typing import Optional, Sequence, Tuple


SCENE_LABEL_ALIASES = {
    "armchair": "chair",
    "bean bag chair": "chair",
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
    "dining table": "table",
    "dinning table": "table",
    "glass table": "table",
    "kitchen table": "table",
    "picnic table": "table",
    "round table": "table",
    "side table": "table",
    "bookcase": "bookshelf",
    "shelf": "bookshelf",
    "bathroom cabinet": "cabinet",
    "cabinetry": "cabinet",
    "file cabinet": "cabinet",
    "kitchen cabinet": "cabinet",
    "side cabinet": "cabinet",
    "tv cabinet": "cabinet",
    "doorway": "door",
    "glass door": "door",
    "screen door": "door",
    "glass window": "window",
    "kitchen window": "window",
    "office window": "window",
    "computer monitor": "monitor",
    "tv": "television",
    "potted plant": "plant",
}


@dataclass(frozen=True)
class SceneDescription:
    signature: Tuple[Tuple[str, str, int], ...]
    text: str


def canonical_scene_label(label: str) -> str:
    normalized = " ".join(str(label).strip().casefold().replace("_", " ").split())
    return SCENE_LABEL_ALIASES.get(normalized, normalized)


def _spoken_count(label: str, count: int) -> str:
    pair_objects = {"scissors", "skis"}
    if label in pair_objects:
        return (
            "a pair of {}".format(label)
            if count == 1
            else "{} pairs of {}".format(count, label)
        )
    if count == 1:
        article = "an" if label[:1].casefold() in "aeiou" else "a"
        return "{} {}".format(article, label)
    irregular = {
        "person": "people",
        "sheep": "sheep",
        "knife": "knives",
        "bookshelf": "bookshelves",
        "mouse": "mice",
    }
    words = label.split()
    plural = irregular.get(words[-1])
    if plural is None:
        plural = words[-1] + (
            "es" if words[-1].endswith(("s", "x", "z", "ch", "sh")) else "s"
        )
    return "{} {}".format(count, " ".join(words[:-1] + [plural]))


def _join_phrases(phrases) -> str:
    if len(phrases) == 1:
        return phrases[0]
    if len(phrases) == 2:
        return " and ".join(phrases)
    return ", ".join(phrases[:-1]) + ", and " + phrases[-1]


def _box_iou(first, second) -> float:
    left = max(float(first[0]), float(second[0]))
    top = max(float(first[1]), float(second[1]))
    right = min(float(first[2]), float(second[2]))
    bottom = min(float(first[3]), float(second[3]))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    if intersection <= 0.0:
        return 0.0
    first_area = max(0.0, float(first[2]) - float(first[0])) * max(
        0.0, float(first[3]) - float(first[1])
    )
    second_area = max(0.0, float(second[2]) - float(second[0])) * max(
        0.0, float(second[3]) - float(second[1])
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def merge_scene_detections(
    primary: Sequence,
    expanded: Sequence,
    overlap_threshold: float = 0.45,
) -> Tuple:
    """Add narration-only detections while suppressing overlapping aliases."""
    if not 0.0 <= overlap_threshold <= 1.0:
        raise ValueError("overlap threshold must be between 0 and 1")
    merged = list(primary)
    specificity = {("desk", "table"), ("monitor", "television")}
    for candidate in expanded:
        candidate_label = canonical_scene_label(candidate.label)
        overlaps = [
            index
            for index, existing in enumerate(merged)
            if _box_iou(existing.box, candidate.box) >= overlap_threshold
        ]
        if any(
            canonical_scene_label(merged[index].label) == candidate_label
            for index in overlaps
        ):
            continue
        broader = [
            index
            for index in overlaps
            if (candidate_label, canonical_scene_label(merged[index].label))
            in specificity
        ]
        if broader:
            for index in reversed(broader):
                del merged[index]
        elif any(
            (canonical_scene_label(merged[index].label), candidate_label)
            in specificity
            for index in overlaps
        ):
            continue
        merged.append(candidate)
    return tuple(merged)


def describe_scene(
    detections: Sequence,
    frame_width: int,
    confidence: float = 0.35,
    maximum_groups: int = 4,
) -> Optional[SceneDescription]:
    """Summarize confident detections by class and horizontal location."""
    if frame_width <= 0:
        raise ValueError("frame width must be positive")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("scene confidence must be between 0 and 1")
    if maximum_groups < 1:
        raise ValueError("maximum scene groups must be positive")

    groups = Counter()
    for item in detections:
        if float(item.confidence) < confidence:
            continue
        label = canonical_scene_label(item.label)
        if not label:
            continue
        center_x = float(item.center[0])
        if center_x < frame_width / 3.0:
            location = "on the left"
        elif center_x > frame_width * 2.0 / 3.0:
            location = "on the right"
        else:
            location = "ahead"
        groups[(label, location)] += 1

    if not groups:
        return None

    location_order = {"ahead": 0, "on the left": 1, "on the right": 2}
    ordered = sorted(
        groups.items(),
        key=lambda item: (
            location_order[item[0][1]],
            -item[1],
            item[0][0],
        ),
    )[:maximum_groups]
    phrases = [
        "{} {}".format(_spoken_count(label, count), location)
        for (label, location), count in ordered
    ]
    body = _join_phrases(phrases)
    signature = tuple((label, location, count) for (label, location), count in ordered)
    return SceneDescription(signature, "I see {}.".format(body))


class SceneNarrationPolicy:
    """Speak stable scene changes, with repeat and empty-scene rate limits."""

    def __init__(
        self,
        stable_frames: int = 3,
        repeat_seconds: float = 12.0,
        clear_seconds: float = 3.0,
    ):
        if stable_frames < 1:
            raise ValueError("speech stable frames must be positive")
        if repeat_seconds < 1.0:
            raise ValueError("speech repeat interval must be at least one second")
        if clear_seconds < 0.0:
            raise ValueError("speech clear interval cannot be negative")
        self.stable_frames = stable_frames
        self.repeat_seconds = repeat_seconds
        self.clear_seconds = clear_seconds
        self._candidate = None
        self._candidate_frames = 0
        self._spoken_signature = None
        self._spoken_at = float("-inf")
        self._empty_since = None
        self._empty_announced = False

    def update(
        self,
        scene: Optional[SceneDescription],
        now_s: Optional[float] = None,
    ) -> Optional[str]:
        now_s = time.monotonic() if now_s is None else now_s
        if scene is None:
            self._candidate = None
            self._candidate_frames = 0
            if self._empty_since is None:
                self._empty_since = now_s
            if (
                self._spoken_signature is not None
                and not self._empty_announced
                and now_s - self._empty_since >= self.clear_seconds
            ):
                self._spoken_signature = None
                self._spoken_at = now_s
                self._empty_announced = True
                return "I do not see any recognized objects now."
            return None

        self._empty_since = None
        self._empty_announced = False
        if scene.signature == self._candidate:
            self._candidate_frames += 1
        else:
            self._candidate = scene.signature
            self._candidate_frames = 1
        if self._candidate_frames < self.stable_frames:
            return None

        changed = scene.signature != self._spoken_signature
        repeat_due = now_s - self._spoken_at >= self.repeat_seconds
        if changed or repeat_due:
            self._spoken_signature = scene.signature
            self._spoken_at = now_s
            return scene.text
        return None


class PiperSceneSpeaker:
    """Persistent subprocess bridge with replace-latest, non-blocking speech."""

    def __init__(self, python_executable: Path, worker: Path, model: Path):
        self.python_executable = Path(python_executable)
        self.worker = Path(worker)
        self.model = Path(model)
        self._process = None
        self._messages = queue.Queue()
        self._stderr = []
        self._threads = []
        self._closed = False
        self._error = None  # type: Optional[str]

    def start(self, timeout_s: float = 45.0) -> None:
        for path, description in (
            (self.python_executable, "Piper Python executable"),
            (self.worker, "Piper speech worker"),
            (self.model, "Piper voice model"),
            (Path(str(self.model) + ".json"), "Piper voice configuration"),
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
                return
            if message.get("event") == "error":
                error = str(message.get("message") or "Piper startup failed")
                self.close()
                raise RuntimeError(error)
        self.close()
        raise RuntimeError(self._error or "Piper speech worker did not become ready")

    def _read_stdout(self) -> None:
        assert self._process is not None and self._process.stdout is not None
        for line in self._process.stdout:
            try:
                message = json.loads(line)
            except (TypeError, ValueError):
                message = {"event": "error", "message": "Invalid Piper event"}
            if message.get("event") == "error":
                self._error = str(message.get("message") or "Piper speech error")
            self._messages.put(message)

    def _read_stderr(self) -> None:
        assert self._process is not None and self._process.stderr is not None
        for line in self._process.stderr:
            self._stderr.append(line.rstrip())

    def speak(self, text: str) -> None:
        if self.error is not None:
            raise RuntimeError(self.error)
        if self._closed or self._process is None or self._process.stdin is None:
            raise RuntimeError("Piper speech worker is not running")
        clean = " ".join(text.strip().split())
        if not clean:
            return
        try:
            self._process.stdin.write(json.dumps({"text": clean}) + "\n")
            self._process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError("Piper speech worker closed unexpectedly") from exc

    @property
    def error(self) -> Optional[str]:
        if self._error:
            return self._error
        if self._process is not None and self._process.poll() not in (None, 0):
            stderr = "\n".join(self._stderr).strip()
            return stderr or "Piper speech worker exited"
        return None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._process is None:
            return
        if self._process.poll() is None and self._process.stdin is not None:
            try:
                self._process.stdin.write(json.dumps({"event": "close"}) + "\n")
                self._process.stdin.flush()
                self._process.stdin.close()
                self._process.wait(timeout=0.2)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                self._process.terminate()
                try:
                    self._process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=2.0)
        for stream in (self._process.stdout, self._process.stderr):
            if stream is not None:
                stream.close()
        self._process = None
