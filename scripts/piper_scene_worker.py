"""Persistent Piper synthesizer/player controlled by JSON lines on stdin."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import queue
import sys
import tempfile
import threading
import wave

from piper import PiperVoice, SynthesisConfig


def emit(event: str, **fields) -> None:
    fields["event"] = event
    print(json.dumps(fields, ensure_ascii=False), flush=True)


def play_wave(path: Path) -> None:
    if sys.platform != "win32":
        raise RuntimeError("Piper scene playback currently requires Windows")
    import winsound

    winsound.PlaySound(str(path), winsound.SND_FILENAME)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--length-scale", type=float, default=0.95)
    parser.add_argument("--volume", type=float, default=1.0)
    return parser


def run(args) -> int:
    voice = PiperVoice.load(args.model)
    synthesis = SynthesisConfig(
        length_scale=args.length_scale,
        volume=args.volume,
    )
    pending = queue.Queue(maxsize=1)

    def synthesize_loop() -> None:
        while True:
            text = pending.get()
            if text is None:
                return
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as item:
                    output_path = Path(item.name)
                try:
                    with wave.open(str(output_path), "wb") as wav_file:
                        voice.synthesize_wav(text, wav_file, syn_config=synthesis)
                    play_wave(output_path)
                    emit("spoken", text=text)
                finally:
                    try:
                        output_path.unlink()
                    except FileNotFoundError:
                        pass
            except Exception as exc:
                emit("error", message=str(exc))

    worker = threading.Thread(target=synthesize_loop, name="PiperPlayback", daemon=False)
    worker.start()
    emit("ready", model=str(args.model))

    for line in sys.stdin:
        try:
            message = json.loads(line)
            if message.get("event") == "close":
                break
            text = " ".join(str(message.get("text", "")).strip().split())
            if not text:
                continue
            try:
                while True:
                    pending.get_nowait()
            except queue.Empty:
                pass
            pending.put_nowait(text)
        except Exception as exc:
            emit("error", message=str(exc))

    try:
        while True:
            pending.get_nowait()
    except queue.Empty:
        pass
    pending.put_nowait(None)
    worker.join()
    return 0


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
