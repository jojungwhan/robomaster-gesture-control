[CmdletBinding()]
param(
    [switch]$RefreshRuntime
)

$ErrorActionPreference = 'Stop'

$installedRoot = 'C:\Program Files\DJI Product\RoboMaster'
$installedExecutable = Join-Path $installedRoot 'RoboMaster.exe'
$runtimeRoot = Join-Path `
    ([Environment]::GetFolderPath([Environment+SpecialFolder]::LocalApplicationData)) `
    'RoboMasterGesture\RoboMaster'
$runtimeExecutable = Join-Path $runtimeRoot 'RoboMaster.exe'
$runtimeMarker = Join-Path $runtimeRoot '.source-version'

if (-not (Test-Path -LiteralPath $installedExecutable -PathType Leaf)) {
    throw "RoboMaster executable not found: $installedExecutable"
}

# This 2019 app writes Statistic.xml and firmware bookkeeping beside its EXE.
# Program Files is read-only for the standard-integrity process required by the
# gesture controller, and the resulting exception interrupts Solo-mode setup.
$installedFile = Get-Item -LiteralPath $installedExecutable
$installedVersion = $installedFile.VersionInfo.FileVersion
$sourceFingerprint = '{0}|{1}|{2}' -f `
    $installedVersion, $installedFile.Length, $installedFile.LastWriteTimeUtc.Ticks
$runtimeFingerprint = if (Test-Path -LiteralPath $runtimeMarker -PathType Leaf) {
    [System.IO.File]::ReadAllText($runtimeMarker).Trim()
} else {
    $null
}
$runtimeNeedsCopy = $RefreshRuntime -or `
    -not (Test-Path -LiteralPath $runtimeExecutable -PathType Leaf) -or `
    ($runtimeFingerprint -ne $sourceFingerprint)

$runningApps = @(
    Get-CimInstance Win32_Process -Filter "Name = 'RoboMaster.exe'" -ErrorAction SilentlyContinue
)
if ($runningApps.Count -gt 0) {
    $runtimeProcess = $runningApps | Where-Object {
        $_.ExecutablePath -and $_.ExecutablePath.Equals(
            $runtimeExecutable,
            [StringComparison]::OrdinalIgnoreCase
        )
    } | Select-Object -First 1

    if ($runtimeProcess -and -not $runtimeNeedsCopy) {
        [pscustomobject]@{
            ProcessId = [int]$runtimeProcess.ProcessId
            StartedAt = $null
            RuntimePath = $runtimeExecutable
            RuntimeInitialized = $false
            AlreadyRunning = $true
            CompatibilityLayer = 'RunAsInvoker'
        }
        return
    }

    $paths = ($runningApps | ForEach-Object {
        if ($_.ExecutablePath) { $_.ExecutablePath } else { "PID $($_.ProcessId)" }
    }) -join ', '
    throw "Close the running RoboMaster app before launching the writable runtime. Running: $paths"
}

if ($runtimeNeedsCopy) {
    [System.IO.Directory]::CreateDirectory($runtimeRoot) | Out-Null
    # A launch interrupted during the copy must retry instead of accepting a
    # directory that merely happens to contain RoboMaster.exe already.
    [System.IO.File]::WriteAllText($runtimeMarker, 'copy-in-progress')
    Copy-Item -Path (Join-Path $installedRoot '*') `
        -Destination $runtimeRoot -Recurse -Force

    $runtimeVersion = (Get-Item -LiteralPath $runtimeExecutable).VersionInfo.FileVersion
    if ($runtimeVersion -ne $installedVersion) {
        throw "RoboMaster runtime copy failed version validation: $runtimeExecutable"
    }
    [System.IO.File]::WriteAllText($runtimeMarker, $sourceFingerprint)
}

# Fail early with a useful message if policy changes make the runtime read-only.
$writeProbe = Join-Path $runtimeRoot '.robomaster-write-test'
try {
    [System.IO.File]::WriteAllText($writeProbe, 'ok')
} finally {
    if (Test-Path -LiteralPath $writeProbe) {
        [System.IO.File]::Delete($writeProbe)
    }
}

# The app and gesture controller must share the user's standard integrity level
# for Windows to permit guarded W/A/S/D input.
$env:__COMPAT_LAYER = 'RunAsInvoker'
$process = Start-Process -FilePath $runtimeExecutable `
    -WorkingDirectory $runtimeRoot -PassThru

[pscustomobject]@{
    ProcessId = $process.Id
    StartedAt = (Get-Date).ToString('o')
    RuntimePath = $runtimeExecutable
    RuntimeInitialized = $runtimeNeedsCopy
    AlreadyRunning = $false
    CompatibilityLayer = $env:__COMPAT_LAYER
}
