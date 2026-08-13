param(
    [switch]$Live,
    [ValidateSet('sdk', 's1-app')]
    [string]$Transport = 's1-app',
    [ValidateSet('robomaster-app', 'sdk', 'webcam', 'file')]
    [string]$Source = 'robomaster-app',
    [ValidateSet('ap', 'sta', 'rndis')]
    [string]$Connection = 'sta',
    [ValidateSet('tcp', 'udp')]
    [string]$Protocol = 'tcp',
    [string]$InputFile,
    [int]$WebcamIndex = 0,
    [string]$Target = 'bottle',
    [string]$Model = 'yolo11n.pt',
    [double]$Confidence = 0.35,
    [int]$ImageSize = 416,
    [string]$Device = 'cpu',
    [int]$MinimumLockFrames = 3,
    [double]$CenterDeadzone = 0.12,
    [double]$StopHeight = 0.38,
    [double]$ResumeHeight = 0.31,
    [double]$ForwardSpeed = 0.15,
    [double]$StrafeSpeed = 0.12,
    [double]$PersonStopConfidence = 0.30,
    [double]$MaxInferenceSeconds = 1.50,
    [switch]$Speak,
    [double]$SpeechConfidence = 0.45,
    [int]$SpeechStableFrames = 3,
    [double]$SpeechRepeatSeconds = 12.0,
    [double]$SpeechClearSeconds = 3.0,
    [int]$SpeechMaxGroups = 4,
    [switch]$BasicSceneOnly,
    [double]$ExpandedSceneConfidence = 0.35,
    [int]$ExpandedSceneImageSize = 320,
    [double]$ExpandedSceneInterval = 1.5,
    [string]$ExpandedSceneModel,
    [string]$PiperPython,
    [string]$PiperModel,
    [double]$Duration = 0,
    [switch]$NoPreview,
    [string]$RobotIp,
    [string]$LocalIp
)

$ErrorActionPreference = 'Stop'
$workspaceRoot = Split-Path -Parent $PSScriptRoot
$pythonCommand = Join-Path $workspaceRoot '.venv-robomaster\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonCommand)) {
    throw "Python environment not found: $pythonCommand"
}

$arguments = @(
    '-m', 'robomaster_gesture.yolo_follow',
    '--transport', $Transport,
    '--source', $Source,
    '--connection', $Connection,
    '--protocol', $Protocol,
    '--webcam-index', $WebcamIndex,
    '--target', $Target,
    '--model', $Model,
    '--confidence', $Confidence,
    '--image-size', $ImageSize,
    '--device', $Device,
    '--minimum-lock-frames', $MinimumLockFrames,
    '--center-deadzone', $CenterDeadzone,
    '--stop-height', $StopHeight,
    '--resume-height', $ResumeHeight,
    '--forward-speed', $ForwardSpeed,
    '--strafe-speed', $StrafeSpeed,
    '--person-stop-confidence', $PersonStopConfidence,
    '--max-inference-seconds', $MaxInferenceSeconds,
    '--speech-confidence', $SpeechConfidence,
    '--speech-stable-frames', $SpeechStableFrames,
    '--speech-repeat-seconds', $SpeechRepeatSeconds,
    '--speech-clear-seconds', $SpeechClearSeconds,
    '--speech-max-groups', $SpeechMaxGroups,
    '--expanded-scene-confidence', $ExpandedSceneConfidence,
    '--expanded-scene-image-size', $ExpandedSceneImageSize,
    '--expanded-scene-interval', $ExpandedSceneInterval
)
if ($Live) { $arguments += '--live' }
if ($Speak) { $arguments += '--speak' }
if ($BasicSceneOnly) { $arguments += '--basic-scene-only' }
if ($ExpandedSceneModel) { $arguments += @('--expanded-scene-model', $ExpandedSceneModel) }
if ($PiperPython) { $arguments += @('--piper-python', $PiperPython) }
if ($PiperModel) { $arguments += @('--piper-model', $PiperModel) }
if ($InputFile) { $arguments += @('--input-file', $InputFile) }
if ($Duration -gt 0) { $arguments += @('--duration', $Duration) }
if ($NoPreview) { $arguments += '--no-preview' }
if ($RobotIp) { $arguments += @('--robot-ip', $RobotIp) }
if ($LocalIp) { $arguments += @('--local-ip', $LocalIp) }

Push-Location -LiteralPath $PSScriptRoot
try {
    & $pythonCommand @arguments
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
