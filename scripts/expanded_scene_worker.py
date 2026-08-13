"""Low-priority prompt-free YOLOE inference worker for scene narration."""

from __future__ import annotations

import argparse
import base64
import ctypes
import json
from pathlib import Path
import sys

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robomaster_gesture.expanded_scene import (
    EXPANDED_LABEL_ALIASES,
    confirmed_expanded_detections,
)


def emit(event: str, **fields) -> None:
    fields["event"] = event
    print(json.dumps(fields, ensure_ascii=False), flush=True)


def lower_process_priority() -> None:
    if sys.platform == "win32":
        below_normal_priority_class = 0x00004000
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.argtypes = ()
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.SetPriorityClass.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
        kernel32.SetPriorityClass.restype = ctypes.c_int
        if not kernel32.SetPriorityClass(
            kernel32.GetCurrentProcess(), below_normal_priority_class
        ):
            raise ctypes.WinError(ctypes.get_last_error())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.30)
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--device", default="cpu")
    return parser


def run(args) -> int:
    import torch
    from ultralytics import YOLOE

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    model = YOLOE(str(args.model), verbose=False)
    names = dict(model.names)
    class_ids = [
        class_id
        for class_id, name in names.items()
        if str(name).casefold() in EXPANDED_LABEL_ALIASES
    ]
    if not class_ids:
        raise RuntimeError("expanded-scene model has none of the requested labels")
    # Some native ML runtimes reset process priority while initializing.
    lower_process_priority()
    emit(
        "ready",
        model=str(args.model),
        labels=sorted(set(EXPANDED_LABEL_ALIASES.values())),
    )
    previous_detections = ()

    for line in sys.stdin:
        try:
            message = json.loads(line)
            if message.get("event") == "close":
                break
            if message.get("event") != "frame":
                continue
            encoded = base64.b64decode(message["jpeg"], validate=True)
            frame = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError("could not decode expanded-scene frame")
            results = model.predict(
                frame,
                conf=args.confidence,
                imgsz=args.image_size,
                device=args.device,
                classes=class_ids,
                max_det=40,
                verbose=False,
            )
            detections = []
            if results and results[0].boxes is not None:
                boxes = results[0].boxes
                for coordinates, confidence, class_id in zip(
                    boxes.xyxy.cpu().tolist(),
                    boxes.conf.cpu().tolist(),
                    boxes.cls.int().cpu().tolist(),
                ):
                    source_label = str(names[int(class_id)]).casefold()
                    detections.append(
                        {
                            "label": EXPANDED_LABEL_ALIASES[source_label],
                            "confidence": float(confidence),
                            "box": [float(value) for value in coordinates],
                        }
                    )
            confirmed = confirmed_expanded_detections(
                previous_detections, detections
            )
            previous_detections = tuple(detections)
            emit(
                "detections",
                sequence=int(message.get("sequence", 0)),
                detections=confirmed,
            )
        except Exception as exc:
            emit("error", message=str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
