"""Local free-form microphone transcription with Faster Whisper.

This module runs in a dedicated modern Python environment and communicates with
the RoboMaster voice controller using one JSON object per stdout line.  Audio is
never sent to a service.  An energy gate prevents silent-room Whisper
hallucinations, while command authorization remains in ``voice_control``.
"""

from __future__ import annotations

import argparse
from collections import deque
import json
import math
from pathlib import Path
import queue
import sys
import threading
import time
from typing import Optional, Sequence


SAMPLE_RATE = 16000
BLOCK_DURATION_S = 0.10
DEFAULT_MODEL_NAME = "base.en"
_OUTPUT_LOCK = threading.Lock()


def audio_level_from_rms(rms: float) -> int:
    """Map digital full-scale RMS energy to a useful 0-100 UI value."""
    try:
        rms = float(rms)
    except (TypeError, ValueError):
        return 0
    if not math.isfinite(rms) or rms <= 0.0:
        return 0
    dbfs = 20.0 * math.log10(rms)
    return max(0, min(100, int(round((dbfs + 60.0) * 2.0))))


def confidence_from_log_probabilities(segments) -> float:
    """Convert Whisper segment log probabilities to an approximate score."""
    weighted_total = 0.0
    total_duration = 0.0
    for segment in segments:
        duration = max(0.05, float(segment.end) - float(segment.start))
        log_probability = float(segment.avg_logprob)
        if not math.isfinite(log_probability):
            continue
        weighted_total += log_probability * duration
        total_duration += duration
    if total_duration <= 0.0:
        return 0.0
    return max(0.0, min(1.0, math.exp(weighted_total / total_duration)))


def _write_event(data) -> None:
    with _OUTPUT_LOCK:
        sys.stdout.write(json.dumps(data, ensure_ascii=True, separators=(",", ":")))
        sys.stdout.write("\n")
        sys.stdout.flush()


def _clean_text(segments) -> str:
    return " ".join(
        " ".join(str(segment.text).split())
        for segment in segments
        if str(segment.text).strip()
    ).strip()


# Bias Whisper toward the small driving vocabulary. Short isolated words such as
# "forward" are otherwise easily misheard (for example as "oh wait"); listing the
# expected words as hotwords raises both their recognition rate and confidence.
COMMAND_HOTWORDS = "forward back left right stop turn go straight reverse"


def _transcribe(model, audio, captured_at_s: float) -> None:
    segments, _information = model.transcribe(
        audio,
        language="en",
        task="transcribe",
        # Greedy decoding keeps latency low so a command is not rejected as stale
        # before it can drive the robot. The hotwords below recover most of the
        # accuracy a wider beam would add, at a fraction of the cost.
        beam_size=1,
        best_of=1,
        temperature=0.0,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 300},
        condition_on_previous_text=False,
        without_timestamps=True,
        hotwords=COMMAND_HOTWORDS,
    )
    materialized = list(segments)
    text = _clean_text(materialized)
    if not text or not any(character.isalnum() for character in text):
        return
    _write_event(
        {
            "event": "recognized",
            "text": text,
            "confidence": round(
                confidence_from_log_probabilities(materialized),
                6,
            ),
            "age_seconds": round(
                max(0.0, time.monotonic() - captured_at_s),
                6,
            ),
        }
    )


def _load_model(args):
    from faster_whisper import WhisperModel

    model_path = Path(args.model)
    if not model_path.is_dir():
        raise RuntimeError(
            "Local Whisper model was not found: {}. Run setup_whisper.ps1."
            .format(model_path)
        )
    _write_event(
        {
            "event": "status",
            "message": "Loading local Whisper {} on CPU...".format(
                args.model_name
            ),
        }
    )
    return WhisperModel(
        str(model_path),
        device="cpu",
        compute_type="int8",
        cpu_threads=max(1, int(args.cpu_threads)),
        num_workers=1,
        local_files_only=True,
    )


def _run_audio_file(args, model) -> int:
    audio_path = Path(args.audio_file).resolve()
    _write_event(
        {
            "event": "ready",
            "message": "Local Whisper {}; audio file {}".format(
                args.model_name,
                audio_path,
            ),
        }
    )
    _transcribe(model, str(audio_path), time.monotonic())
    _write_event({"event": "completed"})
    return 0


def _replace_pending_utterance(utterance_queue, utterance) -> None:
    try:
        utterance_queue.put_nowait(utterance)
        return
    except queue.Full:
        pass
    try:
        utterance_queue.get_nowait()
    except queue.Empty:
        pass
    try:
        utterance_queue.put_nowait(utterance)
    except queue.Full:
        pass


def _run_microphone(args, model) -> int:
    import numpy as np
    import sounddevice as sd

    device = args.input_device
    if isinstance(device, str) and device.isdigit():
        device = int(device)
    device_info = sd.query_devices(device, "input")
    sd.check_input_settings(
        device=device,
        channels=1,
        dtype="float32",
        samplerate=SAMPLE_RATE,
    )

    audio_queue = queue.Queue(maxsize=40)
    utterance_queue = queue.Queue(maxsize=1)
    def audio_callback(indata, _frames, _time_info, status):
        if status:
            _write_event({"event": "status", "message": "Microphone: {}".format(status)})
        block = indata[:, 0].copy()
        try:
            audio_queue.put_nowait(block)
        except queue.Full:
            try:
                audio_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                audio_queue.put_nowait(block)
            except queue.Full:
                pass

    def inference_worker():
        while True:
            item = utterance_queue.get()
            try:
                if item is None:
                    return
                samples, captured_at_s = item
                _transcribe(model, samples, captured_at_s)
            except Exception as exc:
                _write_event(
                    {
                        "event": "status",
                        "message": (
                            "Whisper could not transcribe audio: {}".format(exc)
                        ),
                    }
                )
            finally:
                utterance_queue.task_done()

    worker = threading.Thread(
        target=inference_worker,
        name="LocalWhisperInference",
        daemon=True,
    )
    worker.start()

    block_size = int(round(SAMPLE_RATE * BLOCK_DURATION_S))
    # 0.6 s of pre-roll so the onset of a word is never clipped when speech
    # detection trips slightly late.
    pre_roll = deque(maxlen=6)
    speech_blocks = []
    speech_active = False
    silence_blocks = 0
    noise_floor = 0.0005
    started_at_s = time.monotonic()
    # Trailing silence (in 0.10 s blocks) that ends an utterance and triggers
    # transcription. This is the main knob for how quickly a spoken command is
    # acted on; keep it short enough to feel responsive but long enough not to
    # split a two-word phrase like "nova forward" on its internal pause.
    end_silence_blocks = max(1, int(args.end_silence_blocks))

    try:
        with sd.InputStream(
            device=device,
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=block_size,
            callback=audio_callback,
        ):
            _write_event(
                {
                    "event": "ready",
                    "message": "Local Whisper {}; microphone {}".format(
                        args.model_name,
                        device_info["name"],
                    ),
                }
            )
            while args.duration <= 0.0 or time.monotonic() - started_at_s < args.duration:
                try:
                    block = audio_queue.get(timeout=0.25)
                except queue.Empty:
                    continue
                rms = float(np.sqrt(np.mean(np.square(block), dtype=np.float64)))
                _write_event(
                    {
                        "event": "audio-level",
                        "level": audio_level_from_rms(rms),
                    }
                )

                if not speech_active:
                    noise_floor = 0.98 * noise_floor + 0.02 * rms
                threshold = max(float(args.speech_threshold), noise_floor * 3.0)
                is_speech = rms >= threshold

                if not speech_active:
                    pre_roll.append(block)
                    if is_speech:
                        speech_active = True
                        speech_blocks = list(pre_roll)
                        silence_blocks = 0
                    continue

                speech_blocks.append(block)
                if is_speech:
                    silence_blocks = 0
                else:
                    silence_blocks += 1

                reached_silence = silence_blocks >= end_silence_blocks
                reached_limit = len(speech_blocks) * BLOCK_DURATION_S >= 8.0
                if not reached_silence and not reached_limit:
                    continue

                utterance = np.concatenate(speech_blocks).astype("float32", copy=False)
                if utterance.size >= int(SAMPLE_RATE * 0.5):
                    _replace_pending_utterance(
                        utterance_queue,
                        (utterance, time.monotonic()),
                    )
                speech_active = False
                speech_blocks = []
                silence_blocks = 0
                pre_roll.clear()
    finally:
        if speech_active and speech_blocks:
            utterance = np.concatenate(speech_blocks).astype("float32", copy=False)
            if utterance.size >= int(SAMPLE_RATE * 0.5):
                _replace_pending_utterance(
                    utterance_queue,
                    (utterance, time.monotonic()),
                )
        utterance_queue.put(None)
        worker.join()

    _write_event({"event": "completed"})
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local Faster Whisper JSON-line helper")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--audio-file")
    parser.add_argument("--input-device")
    parser.add_argument("--cpu-threads", type=int, default=12)
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--speech-threshold", type=float, default=0.003)
    # 3 blocks = 0.30 s of trailing silence before a command is transcribed.
    parser.add_argument("--end-silence-blocks", type=int, default=3)
    return parser


def run(args) -> int:
    model = _load_model(args)
    if args.audio_file:
        return _run_audio_file(args, model)
    return _run_microphone(args, model)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except Exception as exc:
        _write_event({"event": "error", "message": str(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
