$ErrorActionPreference = 'Stop'

$projectRoot = $PSScriptRoot
$buildRoot = Join-Path $projectRoot 'build'
$cmakeCommand = Get-Command cmake.exe -ErrorAction SilentlyContinue
if (-not $cmakeCommand) {
    $wingetPackages = Join-Path $env:LOCALAPPDATA 'Microsoft\WinGet\Packages'
    $cmakeExecutable = Get-ChildItem -LiteralPath $wingetPackages -Recurse -File -Filter 'cmake.exe' -ErrorAction SilentlyContinue |
        Where-Object FullName -Like '*Kitware.CMake*' |
        Select-Object -First 1
    if (-not $cmakeExecutable) {
        throw 'CMake was not found on PATH or in the per-user WinGet packages.'
    }
    $cmakePath = $cmakeExecutable.FullName
} else {
    $cmakePath = $cmakeCommand.Source
}

$vsWhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$visualStudioRoot = $null
if (Test-Path -LiteralPath $vsWhere) {
    $visualStudioRoot = & $vsWhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
}
if (-not $visualStudioRoot) {
    $fallback = 'C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools'
    if (Test-Path -LiteralPath $fallback) {
        $visualStudioRoot = $fallback
    }
}
if (-not $visualStudioRoot) {
    throw 'Visual Studio C++ Build Tools were not found.'
}

$developerCommand = Join-Path $visualStudioRoot 'Common7\Tools\VsDevCmd.bat'
if (-not (Test-Path -LiteralPath $developerCommand)) {
    throw "VsDevCmd.bat was not found at $developerCommand"
}

$commandLine = '"{0}" -arch=amd64 -host_arch=amd64 && "{1}" -S "{2}" -B "{3}" -G "NMake Makefiles" && "{1}" --build "{3}"' -f $developerCommand, $cmakePath, $projectRoot, $buildRoot
& $env:ComSpec /d /s /c $commandLine
if ($LASTEXITCODE -ne 0) {
    throw "Native build failed with exit code $LASTEXITCODE"
}

Write-Host "Built: $(Join-Path $buildRoot 'leap_hand_bridge.dll')"
