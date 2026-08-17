# RoboMaster gesture control

This bridge turns one tracked hand into a fail-safe virtual joystick for a DJI RoboMaster chassis. It uses Ultraleap LeapC for tracking. EP/EP Core use DJI's official Python SDK; a stock S1 can use DJI's Windows app as a guarded W/A/S/D transport.

## Gesture mapping

| Gesture or motion | Result |
| --- | --- |
| Hold one open hand for 0.25 seconds | Enter READY state; robot remains stopped |
| Lightly pinch thumb and index finger for 0.20 seconds | Capture the current hand position and enter DRIVING; no grip is required |
| Move pinched hand away from you | Drive forward |
| Move pinched hand toward you | Drive backward |
| Move pinched hand left or right | Strafe left or right |
| Point the pinched hand left or right at the wrist | Rotate left or right |
| Release pinch | Immediate stop and disarm |
| Make a fist | Emergency stop and disarm |
| Lose tracking, show two hands, change tracked hand, or encounter an error | Immediate stop and disarm |

The defaults cap translation at 0.35 m/s and rotation at 35 degrees/s. Position and wrist rotation have dead zones, hysteresis, smoothing, and a 200 ms stale-input watchdog. Robot control is always dry-run unless the Live switch is supplied.

In stock-S1 app mode, DJI exposes only W/A/S/D chassis movement. Forward/back/strafe are therefore discrete at the app's normal (non-Shift) speed, and wrist-yaw rotation is disabled. After the two rising DRIVING beeps, move the whole pinched hand about 2 cm from the captured anchor. One beep means READY and a low beep means STOP. The adapter releases all movement keys on stop, tracking timeout, focus loss, controller error, and exit, and has its own 250 ms key watchdog.

## Why this gesture set

- Pinch strength and thumb-to-index distance are both exposed by LeapC. Either can engage the dead-man control, so a natural pinch works even if one signal is noisy. Overall grab strength is ignored during a pinch; only a nearly closed fist is treated as an emergency stop. Opening the pinch has an unambiguous meaning: stop.
- Relative hand displacement behaves like a spring-centered joystick and provides proportional speed without depending on hand size or a fixed absolute sensor position.
- Wrist yaw provides proportional turning while keeping the broad face of the hand visible to the camera. Rolling the hand edge-on is less trackable.
- An open-hand hold is a deliberate readiness and re-centering step. A fist is reserved exclusively for emergency stop.
- Swipes were not used because they are brief events and do not naturally define when continuous robot motion must stop.
- Finger-count poses were not used because curled or occluded fingers are less reliable and discrete counts do not provide proportional speed.
- Two-hand control was not used by default because another person's hand entering the camera should stop the robot, not transfer control.

## Build and sensor-only test

From this directory:

    .\build.ps1
    .\run_gesture_control.ps1

The second command prints derived chassis speeds but does not contact or move a robot. Press Ctrl+C to stop.

## Robot connection

The supported DJI connection modes are:

- ap: connect this PC to the RoboMaster's Wi-Fi network.
- sta: put the robot and PC on the same Wi-Fi network; automatic discovery is used unless RobotIp is supplied.
- rndis: connect the RoboMaster intelligent controller by USB/RNDIS.

Verify a robot connection without permitting movement:

    .\run_gesture_control.ps1 -Live -ConnectOnly -Connection sta

Start guarded live control:

    .\run_gesture_control.ps1 -Live -Connection sta

For a stock S1, start DJI's Windows app through the project launcher, connect the
robot, and open the Solo live-drive view:

    .\launch_robomaster_standard.ps1

The launcher keeps a writable runtime copy under
`%LOCALAPPDATA%\RoboMasterGesture\RoboMaster`. This is necessary because the
2019 DJI build writes runtime state beside its executable; running it directly
from `Program Files` at standard integrity can interrupt Solo-mode initialization
and produce a gray/noise camera view. The signed installation is left unchanged.
The first launch copies about 1.1 GB and may show a Windows Firewall prompt;
allow private-network access. Use `-RefreshRuntime` after a DJI app update.

With the live-drive view open, use:

    .\run_gesture_control.ps1 -Live -ConnectOnly -Transport s1-app
    .\run_gesture_control.ps1 -Live -Transport s1-app

Keep the RoboMaster live drive view in the foreground while driving. Changing focus immediately releases W/A/S/D and stops the gesture controller. This mode never sends Shift, mouse input, gimbal input, or blaster input.

## RoboMaster Control Center

Start the Control Center before or during a control session:

    .\run_control_center.ps1

It draws both tracked hands, all finger bones, palm and arm joints, live
pinch/grab meters, coordinates, tracking state, and frame rate. It also provides
one-place controls for `HAND CONTROL`, `VOICE CONTROL`, `STOP CONTROL`, and a
15-second microphone dry test. Confirm the on-screen safety checkbox before a
live input button is enabled. Selecting another input first stops the current
controller and releases W/A/S/D; the machine-local controller lease remains an
independent guard against concurrent live controllers.

`VOICE CONTROL` is a low-latency, no-wake-word mode for the least delay between
speech and movement: say just `forward`, `back`, `left`, `right` (or a diagonal
like `forward left`), and `stop`. It uses a tight 0.20 s end-of-speech window
and sends the command automatically. Because any recognized direction moves the
robot, use it on a clear floor or with `DRY RUN`; the 40% confidence gate and
the strict command-word filter (which rejects ordinary prose) still apply.

The status/log line and the recognized-speech transcript are selectable read-only
text: drag to highlight and press `Ctrl+C`, or right-click for `Copy`. The `COPY`
button in the header copies a full diagnostics snapshot — tracking state, frame
rate, the derived command, the last transcript, and the current status/error — to
the clipboard in one click, which is convenient for pasting an error into a bug
report or chat.

`HAND CONTROL` and `VOICE CONTROL` drive a stock S1 through DJI's RoboMaster
Windows app. If that app is not already open when you start one of these modes,
the Control Center launches it for you through `launch_robomaster_standard.ps1`
(the first launch may copy about 1.1 GB) and asks you to connect the S1 and open
its live drive view, then press the button again. If the app is already open, it
is used as-is and never relaunched.

The Control Center is always on top and opens without taking keyboard focus from
the RoboMaster live-drive view. A deliberate button click can briefly activate
the Control Center; starting Leap or voice control then returns focus to the
RoboMaster app before it can move. Any later focus loss remains a fail-safe stop.
The Leap display uses a separate LeapC client and continues visualizing hands
when robot control is off or voice mode is selected.

The view labels the four translation axes directly: hand away from you is robot
forward, hand toward you is robot back, and hand left/right is robot left/right.
When the gesture controller is running, the line below the skeleton reports the
derived command as `FORWARD`, `BACK`, `LEFT`, `RIGHT`, or a diagonal combination.
It also states whether that command reaches the robot: `SENDS TO S1` for a live
controller versus `NOT SENT (DRY RUN)`, and changes to `STOP` immediately when
the controller disarms.

Tick `DRY RUN (no S1)` to test Leap and Voice without a robot. Dry run computes
and displays commands but never connects to the RoboMaster app, so it needs
neither the safety checkbox nor an S1: use it to confirm gestures and speech are
recognized and to read `NOT SENT (DRY RUN)` before driving for real. Untick it
and confirm the safety checkbox to send `SENDS TO S1`.

Voice mode listens to this laptop's default microphone, not the RoboMaster
camera or the Leap sensor. Say `forward`, `back`, `left`, or `right` (no wake
word); `stop` works too. If the dry microphone test
shows no input, choose `MIC BLOCKED?` in the Control Center. The segmented meter
is active only during Voice or `TEST MIC`: it displays the speech recognizer's
live input energy from 0 to 100, reports `NO SIGNAL` at zero, and warns when the
input is unusually loud. It is an activity level, not the Windows microphone
volume setting. Recognized phrases appear on the same panel as a local Whisper
transcription with an approximate confidence score; the last phrase remains
visible after the dry test finishes. Free-form English such as `there is a chair
beside the table` is transcribed, but only explicit direction phrases (`forward`,
`back`, `left`, `right`, a diagonal, or `stop`) move the chassis; ordinary prose
is rejected by the command-word filter. The guided panel
links to Samsung Settings, Windows microphone privacy, and Windows sound input.
On this Samsung laptop, press `Fn+F10` until the toast says Block Recording is
off. Then turn on Windows **Privacy & security > Microphone > Let desktop apps
access your microphone** and retry the dry test. PowerShell may not appear as an
individual app in that list because Windows applies one global permission to
desktop apps. If the test remains silent, use **Sound > Input > Microphone Array
> Start test** and check whether its meter moves.

Stop the Control Center with its `CLOSE` button or:

    .\run_control_center.ps1 -Stop

For a timed diagnostic, use `-Duration`, for example:

    .\run_control_center.ps1 -Duration 30

The previous `run_leap_visualizer.ps1` launcher remains as a compatible alias.

The implementation is adapted from Ultraleap's Apache-2.0 licensed
[`leapc-python-bindings` visualizer](https://github.com/ultraleap/leapc-python-bindings/blob/main/examples/visualiser.py).
See `THIRD_PARTY_NOTICES.md` for attribution.

## Offline voice control

Install the free local transcription runtime once:

    .\setup_whisper.ps1

This creates an isolated Python 3.9+ environment, installs pinned
`faster-whisper` and `sounddevice` packages, and downloads the compact official
English Whisper model used by the Control Center. Microphone audio remains on
this PC and no account, cloud service, or API key is needed.

Additional models can be installed and the Control Center will offer them in a
`MODEL` menu on its VOICE panel (fastest first):

    .\setup_whisper.ps1 -Model tiny.en          # fastest, least accurate
    .\setup_whisper.ps1                          # base.en - fast, balanced (default)
    .\setup_whisper.ps1 -Model small.en          # more accurate, slower
    .\setup_whisper.ps1 -Model large-v3-turbo    # most accurate, slowest

The `MODEL` menu lists only models that are already downloaded and switches the
recognizer to the chosen one (any running listener stops so the next VOICE or
`TEST MIC` press loads it). `base.en` is the responsive default; `tiny.en` gives
the lowest latency; `small.en` and `large-v3-turbo` are more accurate but slower
and, on a modest CPU, may lag live robot steering.

Movement needs no wake word by default: say the direction (`forward`, `back`,
`left`, `right`) and it drives; `stop`, `halt`, `freeze`, and `emergency stop`
also work. Pass `--wake-word <word>` to require a spoken prefix (for example in a
noisy room). Each accepted movement is a short pulse (0.60 seconds by default)
and then stops automatically.
In live mode, microphone or WAV recognition starts only after the robot
connection and independent stale-command watchdog are active, so speech cannot
queue while the controller is still connecting. Non-stop transcriptions more
than four seconds old are rejected rather than executed.

Test the default PC microphone without contacting the robot:

    .\run_voice_control.ps1 -Duration 20

Say `forward`, `back`, `left`, `right`, or a diagonal such as `forward left`.
Test a recorded command from a PCM WAV file with:

    .\run_voice_control.ps1 -AudioFile .\command.wav

To diagnose the older constrained Windows recognizer, select it explicitly or
list the installed Windows recognizers with:

    .\run_voice_control.ps1 -SpeechEngine windows -Duration 20
    .\run_voice_control.ps1 -ListRecognizers

Whisper cannot bypass a hardware or Windows privacy mute. If `TEST MIC` stays at
`NO SIGNAL 0%`, unblock the laptop microphone first using the Control Center's
`MIC BLOCKED?` guide; transcription begins once that meter receives real audio.

When dry-run recognition is correct, connect the S1 in its foreground live-drive
view and verify input access without starting the microphone or movement:

    .\run_voice_control.ps1 -Live -ConnectOnly -Transport s1-app

Then explicitly enable voice movement:

    .\run_voice_control.ps1 -Live -Transport s1-app

For an EP/EP Core using the DJI SDK, use `-Transport sdk`. SDK voice mode also
supports `turn left/right`; the stock S1 app exposes only W/A/S/D, so a
turn phrase stops instead of synthesizing unsupported input.

## YOLO object following

Install the optional pinned computer-vision runtime once:

    .\setup_yolo.ps1

YOLO mode detects and tracks one named model class, draws boxes, track IDs,
state, FPS, and the intended robot direction, and remains a dry run unless
`-Live` is supplied. For a stock S1, first connect its camera in the RoboMaster
live-drive view. Move an object in front of the camera and verify that the
stream is genuinely changing without loading YOLO or enabling motion:

    .\run_yolo_follow.ps1 -CameraCheck -Source robomaster-app

Then test detection without motion:

    .\run_yolo_follow.ps1 -Target bottle

Once the preview reliably follows the intended object, put the robot on a clear
floor and explicitly enable movement:

    .\run_yolo_follow.ps1 -Live -Transport s1-app -Source robomaster-app -Target bottle

For an EP/EP Core with SDK camera access:

    .\run_yolo_follow.ps1 -Live -Transport sdk -Source sdk -Target bottle

For SDK STA discovery, close the RoboMaster desktop app first because both use
UDP port 45678. Alternatively, pass the robot's current address with
`-RobotIp`; this skips broadcast discovery. The stock-S1 app-camera workflow
above does not use the SDK and should keep the desktop app open.

Live mode permits only the robot-mounted camera source and refuses to follow a
person. It requires three consecutive target frames before movement; target
loss, a changed track ID, stale inference, a frozen camera frame, camera loss,
app focus loss, or any
person detected alongside a non-person target causes a stop. The robot strafes
left/right until the target is centered, approaches it slowly, and stops when
the target box reaches the configured size. A machine-local lease prevents
gesture, voice, and YOLO live controllers from running simultaneously.

Setup downloads the default `yolo11n.pt` follow model and compact
`yoloe-26n-seg-pf.pt` narration model; neither is committed. Ultralytics code
and pretrained models use AGPL-3.0 by default; review
`THIRD_PARTY_NOTICES.md` and Ultralytics licensing before deployment.

### Natural English scene speech

Install the optional local Piper neural voice once (Python 3.9 or newer is
required for Piper; the script creates a separate environment):

    .\setup_scene_speech.ps1 -TestVoice

Then add `-Speak` to a YOLO run:

    .\run_yolo_follow.ps1 -Speak -Target bottle

The detector describes objects beyond the follow target, using counts and
positions such as `I see a desk ahead, a chair on the left, and a table on the
right.` A compact prompt-free YOLOE model expands narration with furniture such
as chairs, desks, tables, bookshelves, cabinets, doors, windows, monitors,
lamps, mirrors, beds, and couches. It runs only in a low-priority background
worker, confirms additions across two separate scans, and never supplies
detections to movement or person-stop logic. Use `-BasicSceneOnly` to disable
it. Expanded labels are shown in purple as `SCENE <label>` and are for
narration, not follow targets. A scene must remain stable for three frames
before it is announced; unchanged scenes are repeated no more than once every
12 seconds. Speech and expanded recognition use replace-latest workers, so
neither blocks camera inference or the robot's motion watchdog.

With the stock S1 app transport, speech comes from the PC's default audio
output because that transport exposes only W/A/S/D movement. The selected
`en_US-kristin-medium` voice is a local Piper neural voice with a natural US
English female voice, trained from public-domain LibriVox recordings. No
microphone audio, camera frame, or scene description is sent to a cloud
service. Pass `-Voice <PiperVoiceName>` during setup and `-PiperModel <path>`
while running if you want to use a different installed voice.

The app and controller must run at the same Windows privilege level. Normally both can run without elevation. If the DJI app was launched as administrator, either restart it normally or launch the controller from an administrator PowerShell.

For AP mode:

    .\run_gesture_control.ps1 -Live -ConnectOnly -Connection ap
    .\run_gesture_control.ps1 -Live -Connection ap

For a known STA address:

    .\run_gesture_control.ps1 -Live -Connection sta -RobotIp 192.168.1.50

Before live control, put the robot on a clear floor or raise the wheels, keep the power switch reachable, and use ConnectOnly first. Do not use this around stairs, traffic, people, or fragile objects.

## API references

- Ultraleap LEAP_HAND: https://docs.ultraleap.com/api-reference/tracking-api/struct/struct_l_e_a_p___h_a_n_d.html
- Ultraleap Python visualizer: https://github.com/ultraleap/leapc-python-bindings/blob/main/examples/visualiser.py
- DJI RoboMaster SDK: https://github.com/dji-sdk/RoboMaster-SDK
- DJI chassis control example: https://robomaster-dev.readthedocs.io/en/latest/python_sdk/beginner_ep.html
- DJI S1 keyboard controls (user manual): https://dl.djicdn.com/downloads/robomaster-s1/20200324/RoboMaster_S1_User_Manual_v1.8_EN.pdf
- Microsoft offline WAV speech input: https://learn.microsoft.com/en-us/dotnet/api/system.speech.recognition.speechrecognitionengine.setinputtowavefile
- Ultralytics multi-object tracking: https://docs.ultralytics.com/modes/track
- Ultralytics YOLOE open-vocabulary detection: https://docs.ultralytics.com/models/yoloe
- Piper local neural speech: https://github.com/OHF-Voice/piper1-gpl
- Piper `en_US-kristin-medium` voice card: https://huggingface.co/rhasspy/piper-voices/blob/main/en/en_US/kristin/medium/MODEL_CARD
