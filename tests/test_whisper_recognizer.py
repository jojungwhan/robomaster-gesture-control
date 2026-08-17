import math
from types import SimpleNamespace
import unittest
from unittest import mock

from robomaster_gesture.whisper_recognizer import (
    audio_level_from_rms,
    build_parser,
    confidence_from_log_probabilities,
    _resolve_whisper_device,
)


class WhisperEndpointTests(unittest.TestCase):
    def test_default_endpoint_silence_is_responsive(self):
        # 3 blocks x 0.10 s = 0.30 s of trailing silence before transcription,
        # down from the older 0.60 s, so a spoken command is acted on sooner.
        args = build_parser().parse_args(("--model", r"C:\model"))
        self.assertEqual(3, args.end_silence_blocks)


class WhisperSignalTests(unittest.TestCase):
    def test_audio_level_maps_silence_and_speech_into_ui_range(self):
        self.assertEqual(0, audio_level_from_rms(0.0))
        self.assertEqual(0, audio_level_from_rms(float("nan")))
        self.assertEqual(0, audio_level_from_rms(0.001))
        self.assertEqual(40, audio_level_from_rms(0.01))
        self.assertEqual(100, audio_level_from_rms(1.0))

    def test_segment_log_probability_becomes_bounded_confidence(self):
        segments = (
            SimpleNamespace(start=0.0, end=1.0, avg_logprob=math.log(0.8)),
            SimpleNamespace(start=1.0, end=2.0, avg_logprob=math.log(0.6)),
        )

        confidence = confidence_from_log_probabilities(segments)

        self.assertAlmostEqual(math.sqrt(0.8 * 0.6), confidence)
        self.assertEqual(0.0, confidence_from_log_probabilities(()))


class WhisperDeviceTests(unittest.TestCase):
    def test_device_flag_defaults_to_auto(self):
        args = build_parser().parse_args(("--model", r"C:\model"))
        self.assertEqual("auto", args.device)

    def test_explicit_devices_pick_matching_compute_type(self):
        self.assertEqual(("cpu", "int8"), _resolve_whisper_device("cpu"))
        self.assertEqual(("cuda", "float16"), _resolve_whisper_device("cuda"))

    def test_auto_uses_gpu_only_when_cuda_is_visible(self):
        with mock.patch(
            "robomaster_gesture.whisper_recognizer._cuda_is_available",
            return_value=True,
        ):
            self.assertEqual(("cuda", "float16"), _resolve_whisper_device("auto"))
        with mock.patch(
            "robomaster_gesture.whisper_recognizer._cuda_is_available",
            return_value=False,
        ):
            self.assertEqual(("cpu", "int8"), _resolve_whisper_device("auto"))


if __name__ == "__main__":
    unittest.main()
