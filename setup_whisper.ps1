param(
    [string]$Python,
    # tiny.en  = fastest, least accurate
    # base.en  = fast, balanced (Control Center default)
    # small.en = more accurate, slower
    # large-v3-turbo = most accurate, slowest (heavy on CPU for live use)
    [ValidateSet('tiny.en', 'base.en', 'small.en', 'large-v3-turbo')]
    [string]$Model = 'base.en',
    [switch]$Recreate,
    [switch]$TestModel
)

$ErrorActionPreference = 'Stop'
$workspaceRoot = Split-Path -Parent $PSScriptRoot
$environment = Join-Path $workspaceRoot '.venv-whisper'
$pythonCommand = Join-Path $environment 'Scripts\python.exe'
switch ($Model) {
    'tiny.en' {
        $modelDirectory = Join-Path $workspaceRoot 'models\whisper-tiny-en-ct2'
        $modelBaseUrl = 'https://huggingface.co/Systran/faster-whisper-tiny.en/resolve/main'
        $modelFiles = @(
            @('config.json', 1000),
            @('model.bin', 70000000),
            @('tokenizer.json', 2000000),
            @('vocabulary.txt', 400000)
        )
    }
    'small.en' {
        $modelDirectory = Join-Path $workspaceRoot 'models\whisper-small-en-ct2'
        $modelBaseUrl = 'https://huggingface.co/Systran/faster-whisper-small.en/resolve/main'
        $modelFiles = @(
            @('config.json', 1000),
            @('model.bin', 460000000),
            @('tokenizer.json', 2000000),
            @('vocabulary.txt', 400000)
        )
    }
    'large-v3-turbo' {
        $modelDirectory = Join-Path $workspaceRoot 'models\whisper-large-v3-turbo-ct2'
        $modelBaseUrl = 'https://huggingface.co/dropbox-dash/faster-whisper-large-v3-turbo/resolve/main'
        $modelFiles = @(
            @('config.json', 1000),
            @('model.bin', 1500000000),
            @('preprocessor_config.json', 100),
            @('tokenizer.json', 2000000),
            @('vocabulary.json', 1000000)
        )
    }
    default {
        $modelDirectory = Join-Path $workspaceRoot 'models\whisper-base-en-ct2'
        $modelBaseUrl = 'https://huggingface.co/Systran/faster-whisper-base.en/resolve/main'
        $modelFiles = @(
            @('config.json', 1000),
            @('model.bin', 140000000),
            @('tokenizer.json', 2000000),
            @('vocabulary.txt', 400000)
        )
    }
}

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
    & $bootstrap @bootstrapArguments -m venv $environment
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the Whisper environment.' }
}

& $pythonCommand -m pip install -r (Join-Path $PSScriptRoot 'requirements-whisper.txt')
if ($LASTEXITCODE -ne 0) { throw 'Local Whisper installation failed.' }

New-Item -ItemType Directory -Force -Path $modelDirectory | Out-Null

function Install-WhisperModelFile {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][long]$MinimumBytes
    )

    $target = Join-Path $modelDirectory $Name
    if ((Test-Path -LiteralPath $target) -and (Get-Item -LiteralPath $target).Length -ge $MinimumBytes) {
        Write-Host "Already downloaded: $Name"
        return
    }

    $partial = "$target.download"
    $url = "$modelBaseUrl/$Name`?download=true"
    Write-Host "Downloading $Name..."
    & curl.exe --fail --location --retry 3 --continue-at - --output $partial $url
    if ($LASTEXITCODE -ne 0) { throw "Download failed: $Name" }
    if (-not (Test-Path -LiteralPath $partial) -or (Get-Item -LiteralPath $partial).Length -lt $MinimumBytes) {
        throw "Downloaded model file is incomplete: $partial"
    }
    Move-Item -LiteralPath $partial -Destination $target -Force
}

foreach ($modelFile in $modelFiles) {
    Install-WhisperModelFile -Name $modelFile[0] -MinimumBytes $modelFile[1]
}

& $pythonCommand -c "import faster_whisper, sounddevice; print('Whisper packages ready.')"
if ($LASTEXITCODE -ne 0) { throw 'Could not import the local Whisper packages.' }

if ($TestModel) {
    $testCode = "from faster_whisper import WhisperModel; import sys; WhisperModel(sys.argv[1], device='cpu', compute_type='int8', local_files_only=True); print('Whisper model loaded successfully.')"
    & $pythonCommand -c $testCode $modelDirectory
    if ($LASTEXITCODE -ne 0) { throw 'The local Whisper model could not be loaded.' }
}

Write-Host "Local Whisper $Model is ready. Choose it in the Control Center's VOICE model menu."
switch ($Model) {
    'tiny.en' { Write-Host 'Fastest model: lowest latency, least accurate.' }
    'base.en' { Write-Host 'The Control Center uses this responsive English model by default.' }
    'small.en' { Write-Host 'More accurate but slower; usable live on faster CPUs.' }
    default { Write-Host 'Most accurate; too delayed for live movement on most CPUs.' }
}
