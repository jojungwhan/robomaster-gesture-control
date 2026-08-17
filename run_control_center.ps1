param(
    [switch]$Stop,
    [double]$Duration = 0,
    [int]$X = -1,
    [int]$Y = 72,
    [int]$Width = 620,
    [int]$Height = 680,
    [double]$Opacity = 0.90
)

# Preferred name for the backward-compatible Leap visualizer launcher.
& (Join-Path $PSScriptRoot 'run_leap_visualizer.ps1') @PSBoundParameters
exit $LASTEXITCODE
