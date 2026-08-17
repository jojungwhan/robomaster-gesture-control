"""Discover the local Whisper models the Control Center can offer for voice.

Models are downloaded by ``setup_whisper.ps1`` into ``<workspace>/models`` as
CTranslate2 directories (each holding a ``model.bin``). This module finds the
installed ones and pairs them with a friendly label and a speed rank so the
Control Center can present a fastest-to-most-accurate picker without loading
anything. faster-whisper loads a model from its directory path, so the display
name is purely cosmetic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


# dir name -> (display name, one-line hint, speed rank: 1 fastest .. 5 slowest)
KNOWN_WHISPER_MODELS: Dict[str, Tuple[str, str, int]] = {
    "whisper-tiny-en-ct2": ("tiny.en", "Fastest, lowest latency (default)", 1),
    "whisper-base-en-ct2": ("base.en", "Slower, a bit more accurate", 2),
    "whisper-small-en-ct2": ("small.en", "Accurate but slow on CPU", 3),
    "whisper-medium-en-ct2": ("medium.en", "Very accurate, slow", 4),
    "whisper-large-v3-turbo-ct2": (
        "large-v3-turbo",
        "Most accurate, slowest",
        5,
    ),
}

# tiny.en is the default: on a typical CPU the larger models transcribe a short
# command too slowly, so it arrives after the stale-command window and is
# ignored. tiny.en keeps the delay between speech and movement lowest.
# Discovery falls back to the fastest installed model if this one is missing.
DEFAULT_MODEL_DIR_NAME = "whisper-tiny-en-ct2"


@dataclass(frozen=True)
class WhisperModelChoice:
    name: str          # display name, also passed as --whisper-model-name
    label: str         # short speed/accuracy hint
    path: Path         # directory that contains model.bin
    speed_rank: int    # 1 fastest .. 5 slowest
    is_default: bool   # the responsive base.en model shipped by default

    @property
    def menu_text(self) -> str:
        return "{}  -  {}".format(self.name, self.label)


def _derive_name(dir_name: str) -> str:
    """Best-effort display name for an unrecognized model directory."""
    stem = dir_name
    if stem.startswith("whisper-"):
        stem = stem[len("whisper-"):]
    if stem.endswith("-ct2"):
        stem = stem[: -len("-ct2")]
    # "base-en" -> "base.en" while leaving names like "large-v3-turbo" intact.
    if stem.endswith("-en"):
        stem = stem[: -len("-en")] + ".en"
    return stem or dir_name


def discover_whisper_models(models_dir: Path) -> List[WhisperModelChoice]:
    """Return installed Whisper models, fastest first.

    A directory qualifies only if it contains a ``model.bin`` so the picker can
    never offer a model that faster-whisper cannot load.
    """
    models_dir = Path(models_dir)
    choices: List[WhisperModelChoice] = []
    try:
        entries = sorted(models_dir.iterdir())
    except (FileNotFoundError, NotADirectoryError, OSError):
        return choices

    for entry in entries:
        if not entry.is_dir() or not (entry / "model.bin").is_file():
            continue
        known = KNOWN_WHISPER_MODELS.get(entry.name)
        if known is not None:
            name, label, rank = known
        else:
            name, label, rank = _derive_name(entry.name), "Installed model", 3
        choices.append(
            WhisperModelChoice(
                name=name,
                label=label,
                path=entry,
                speed_rank=rank,
                is_default=entry.name == DEFAULT_MODEL_DIR_NAME,
            )
        )

    choices.sort(key=lambda choice: (choice.speed_rank, choice.name))
    return choices


def default_choice(choices: List[WhisperModelChoice]) -> WhisperModelChoice:
    """Pick the model the Control Center should start on."""
    for choice in choices:
        if choice.is_default:
            return choice
    return choices[0]
