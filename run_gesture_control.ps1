param(
    [switch]$Live,
    [switch]$ConnectOnly,
    [ValidateSet('sdk', 's1-app')]
    [string]$Transport = 'sdk',
    [ValidateSet('ap', 'sta', 'rndis')]
    [string]$Connection = 'sta',
    [ValidateSet('tcp', 'udp')]
    [string]$Protocol = 'tcp',
    [ValidateSet('right', 'left', 'any')]
    [string]$Hand = 'right',
    [string]$RobotIp,
    [string]$LocalIp,
    [double]$Duration = 0
)

$ErrorActionPreference = 'Stop'
$workspaceRoot = Split-Path -Parent $PSScriptRoot
$pythonCommand = Join-Path $workspaceRoot '.venv-robomaster\Scripts\python.exe'
$bridgeLibrary = Join-Path $PSScriptRoot 'build\leap_hand_bridge.dll'

if (-not (Test-Path -LiteralPath $pythonCommand)) {
    throw "Python environment not found: $pythonCommand"
}
if (-not (Test-Path -LiteralPath $bridgeLibrary) -and -not $ConnectOnly) {
    throw "Leap bridge not built. Run .\build.ps1 first."
}

$launchArguments = @(
    '-m', 'robomaster_gesture',
    '--transport', $Transport,
    '--connection', $Connection,
    '--protocol', $Protocol,
    '--hand', $Hand,
    '--bridge-dll', $bridgeLibrary
)
if ($Live) { $launchArguments += '--live' }
if ($ConnectOnly) { $launchArguments += '--connect-only' }
if ($RobotIp) { $launchArguments += @('--robot-ip', $RobotIp) }
if ($LocalIp) { $launchArguments += @('--local-ip', $LocalIp) }
if ($Duration -gt 0) { $launchArguments += @('--duration', $Duration) }

Push-Location -LiteralPath $PSScriptRoot
try {
    & $pythonCommand @launchArguments
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
