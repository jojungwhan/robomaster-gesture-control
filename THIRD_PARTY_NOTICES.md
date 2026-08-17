# Third-party notices

## Ultraleap LeapC Python visualiser

The hand-skeleton topology and desktop x/z projection in
`robomaster_gesture/leap_visualizer.py` are adapted from Ultraleap's
[`leapc-python-bindings/examples/visualiser.py`](https://github.com/ultraleap/leapc-python-bindings/blob/main/examples/visualiser.py).
The implementation in this project has been rewritten to use Tk, the installed
LeapC CFFI module, and a Windows non-activating click-through overlay.

Copyright 2023 Leap V5 Platform. Licensed under the Apache License, Version 2.0.
The license text is included at `LICENSES/Apache-2.0.txt`; the upstream copy is
available at
https://github.com/ultraleap/leapc-python-bindings/blob/main/LICENSE.md.

## Ultralytics YOLO

Optional object detection and tracking imports the separately installed
[`ultralytics`](https://github.com/ultralytics/ultralytics) package and downloads
Ultralytics pretrained model weights during setup. This includes the default
YOLO11 follow model and the prompt-free YOLOE narration model. The package and
pretrained models are offered under AGPL-3.0 by default; commercial or
closed-source use requires an appropriate Ultralytics Enterprise license.
Neither the package nor model weights are stored in this repository. See
https://www.ultralytics.com/license before deployment.

## Piper neural text-to-speech

Optional English scene narration launches the separately installed
[`piper-tts`](https://github.com/OHF-Voice/piper1-gpl) package as a local worker
process. Piper 1.6.0 is licensed GPL-3.0-or-later. The package is installed into
a sibling virtual environment and is not stored in this repository.

The default `en_US-kristin-medium` US English female voice is downloaded from the
[`rhasspy/piper-voices`](https://huggingface.co/rhasspy/piper-voices) model
repository and is not stored here. Its model card identifies its source as
public-domain LibriVox recordings. Review the model card before redistribution:
https://huggingface.co/rhasspy/piper-voices/blob/main/en/en_US/kristin/medium/MODEL_CARD.

## Whisper speech recognition

Optional free-form transcription launches the separately installed
[`faster-whisper`](https://github.com/SYSTRAN/faster-whisper) package and its
CTranslate2 runtime from a sibling virtual environment. Both are MIT licensed
and are not stored in this repository.

The setup script downloads a CTranslate2 conversion of OpenAI's MIT-licensed
Whisper `base.en` model for responsive English control. It can optionally
download OpenAI's MIT-licensed `large-v3-turbo` model for higher-accuracy offline
transcription. Model weights are stored outside this repository under the
sibling `models` directory.
