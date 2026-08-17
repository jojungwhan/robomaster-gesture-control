import tempfile
import unittest
from pathlib import Path

from robomaster_gesture.whisper_models import (
    WhisperModelChoice,
    default_choice,
    discover_whisper_models,
)


def _make_model_dir(root: Path, name: str, with_binary: bool = True) -> None:
    directory = root / name
    directory.mkdir(parents=True)
    if with_binary:
        (directory / "model.bin").write_bytes(b"0")


class WhisperModelDiscoveryTests(unittest.TestCase):
    def test_installed_models_are_sorted_fastest_first_with_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _make_model_dir(root, "whisper-large-v3-turbo-ct2")
            _make_model_dir(root, "whisper-base-en-ct2")
            _make_model_dir(root, "whisper-small-en-ct2")
            _make_model_dir(root, "whisper-tiny-en-ct2")

            choices = discover_whisper_models(root)

        self.assertEqual(
            ["tiny.en", "base.en", "small.en", "large-v3-turbo"],
            [choice.name for choice in choices],
        )
        # tiny.en is the default: it is fast enough to beat the stale window.
        by_name = {choice.name: choice for choice in choices}
        self.assertTrue(by_name["tiny.en"].is_default)
        self.assertIn("default", by_name["tiny.en"].label)
        self.assertFalse(by_name["small.en"].is_default)

    def test_directories_without_a_model_binary_are_ignored(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _make_model_dir(root, "whisper-base-en-ct2")
            _make_model_dir(root, "whisper-tiny-en-ct2", with_binary=False)
            (root / "not-a-model.txt").write_text("x")

            names = [choice.name for choice in discover_whisper_models(root)]

        self.assertEqual(["base.en"], names)

    def test_unknown_model_directory_gets_a_derived_name(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _make_model_dir(root, "whisper-small-en-ct2")  # known
            _make_model_dir(root, "whisper-distil-large-ct2")  # unknown

            by_name = {c.name: c for c in discover_whisper_models(root)}

        self.assertIn("small.en", by_name)
        self.assertIn("distil-large", by_name)

    def test_missing_directory_returns_no_models(self):
        self.assertEqual([], discover_whisper_models(Path(r"C:\does\not\exist")))

    def test_default_choice_prefers_the_default_then_falls_back_to_first(self):
        tiny = WhisperModelChoice("tiny.en", "", Path("t"), 1, True)
        base = WhisperModelChoice("base.en", "", Path("b"), 2, False)
        self.assertIs(tiny, default_choice([tiny, base]))
        # When the preferred model is not installed, the fastest one is used.
        self.assertIs(base, default_choice([base]))


if __name__ == "__main__":
    unittest.main()
