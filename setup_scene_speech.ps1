param(
    [string]$Python,
    [string]$Voice = 'en_US-john-medium',
    [switch]$Recreate,
    [switch]$TestVoice
)

$ErrorActionPreference = 'Stop'
$workspaceRoot = Split-Path -Parent $PSScriptRoot
$environment = Join-Path $workspaceRoot '.venv-piper'
$pythonCommand = Join-Path $environment 'Scripts\python.exe'
$voiceDirectory = Join-Path $workspaceRoot 'piper-voices'
$model = Join-Path $voiceDirectory "$Voice.onnx"

if ($Recreate -and (Test-Path -LiteralPath $environment)) {
    $resolvedEnvironment = [IO.Path]::GetFullPath($environment)
    $resolvedWorkspace = [IO.Path]::GetFullPath($workspaceRoot)
    if (-not $resolvedEnvironment.StartsWith($resolvedWorkspace, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove environment outside workspace: $resolvedEnvironment"
    }
    Remove-Item -LiteralPath $resolvedEnvironment -Recurse -Force
}
if (-not (Test-Path -LiteralPath $pythonCommand)) {
    $bootstrap = $null
    $bootstrapArguments = @()
    if ($Python) {
        if (-not (Test-Path -LiteralPath $Python)) {
            throw "Python was not found at $Python."
        }
        $bootstrap = $Python
    } else {
        $command = Get-Command python.exe -ErrorAction SilentlyContinue
        if ($command) {
            & $command.Source -c 'import sys; assert sys.version_info >= (3, 9)' 2>$null
            if ($LASTEXITCODE -eq 0) { $bootstrap = $command.Source }
        }
        if (-not $bootstrap) {
            $launcher = Get-Command py.exe -ErrorAction SilentlyContinue
            if ($launcher) {
                foreach ($selector in @('-3.13', '-3.12', '-3.11', '-3.10', '-3.9')) {
                    & $launcher.Source $selector -c 'import sys; assert sys.version_info >= (3, 9)' 2>$null
                    if ($LASTEXITCODE -eq 0) {
                        $bootstrap = $launcher.Source
                        $bootstrapArguments = @($selector)
                        break
                    }
                }
            }
        }
    }
    if (-not $bootstrap) {
        throw 'Python 3.9+ was not found. Supply -Python with a valid executable.'
    }
    & $bootstrap @bootstrapArguments -c 'import sys; assert sys.version_info >= (3, 9)'
    if ($LASTEXITCODE -ne 0) {
        throw 'Piper requires Python 3.9 or newer.'
    }
    & $bootstrap @bootstrapArguments -m venv $environment
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the Piper environment.' }
}

& $pythonCommand -m pip install -r (Join-Path $PSScriptRoot 'requirements-speech.txt')
if ($LASTEXITCODE -ne 0) { throw 'Piper installation failed.' }
New-Item -ItemType Directory -Force -Path $voiceDirectory | Out-Null
& $pythonCommand -m piper.download_voices --download-dir $voiceDirectory $Voice
if ($LASTEXITCODE -ne 0) { throw 'Piper voice download failed.' }

if (-not (Test-Path -LiteralPath $model) -or -not (Test-Path -LiteralPath "$model.json")) {
    throw "Piper voice files are incomplete: $model"
}
Write-Host "Scene speech ready with $Voice."

if ($TestVoice) {
    $testFile = Join-Path ([IO.Path]::GetTempPath()) 'robomaster_scene_speech_test.wav'
    try {
        'Scene speech is ready. I can describe what the robot sees.' |
            & $pythonCommand -m piper -m $model -f $testFile
        if ($LASTEXITCODE -ne 0) { throw 'Piper voice test synthesis failed.' }
        $player = New-Object System.Media.SoundPlayer($testFile)
        $player.PlaySync()
    } finally {
        if (Test-Path -LiteralPath $testFile) {
            Remove-Item -LiteralPath $testFile -Force
        }
    }
}
