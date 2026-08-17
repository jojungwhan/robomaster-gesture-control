param(
    [switch]$Live,
    [switch]$ConnectOnly,
    [ValidateSet('sdk', 's1-app')]
    [string]$Transport = 's1-app',
    [ValidateSet('ap', 'sta', 'rndis')]
    [string]$Connection = 'sta',
    [ValidateSet('tcp', 'udp')]
    [string]$Protocol = 'tcp',
    [string]$AudioFile,
    [switch]$ListRecognizers,
    [ValidateSet('whisper', 'windows')]
    [string]$SpeechEngine = 'whisper',
    [string]$Culture = 'en-US',
    [string]$WakeWord = 'robot',
    [switch]$NoWakeWord,
    [double]$MinConfidence = 0.55,
    [double]$CommandDuration = 0.60,
    [double]$Speed = 0.20,
    [double]$YawSpeed = 25.0,
    [double]$Duration = 0,
    [double]$MaxCommandAge = 4.0,
    [string]$WhisperInputDevice,
    [string]$WhisperModel,
    [string]$WhisperModelName,
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
    '-m', 'robomaster_gesture.voice_control',
    '--transport', $Transport,
    '--connection', $Connection,
    '--protocol', $Protocol,
    '--speech-engine', $SpeechEngine,
    '--culture', $Culture,
    '--wake-word', $WakeWord,
    '--min-confidence', $MinConfidence,
    '--command-duration', $CommandDuration,
    '--speed', $Speed,
    '--yaw-speed', $YawSpeed,
    '--max-command-age', $MaxCommandAge
)
if ($Live) { $arguments += '--live' }
if ($ConnectOnly) { $arguments += '--connect-only' }
if ($ListRecognizers) { $arguments += '--list-recognizers' }
if ($NoWakeWord) { $arguments += '--no-wake-word' }
if ($AudioFile) { $arguments += @('--audio-file', $AudioFile) }
if ($Duration -gt 0) { $arguments += @('--duration', $Duration) }
if ($RobotIp) { $arguments += @('--robot-ip', $RobotIp) }
if ($LocalIp) { $arguments += @('--local-ip', $LocalIp) }
if ($WhisperInputDevice) { $arguments += @('--whisper-input-device', $WhisperInputDevice) }
if ($WhisperModel) { $arguments += @('--whisper-model', $WhisperModel) }
if ($WhisperModelName) { $arguments += @('--whisper-model-name', $WhisperModelName) }

Push-Location -LiteralPath $PSScriptRoot
try {
    & $pythonCommand @arguments
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
