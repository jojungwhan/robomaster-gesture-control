$ErrorActionPreference = 'Stop'

# Launch elevation from a detached helper so the consent dialog is not tied to
# the lifetime of the calling terminal command.
Start-Sleep -Seconds 2
$recoveryScript = Join-Path $PSScriptRoot 'recover_leap_tracking.ps1'
$arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$recoveryScript`""
Start-Process `
    -FilePath "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" `
    -Verb RunAs `
    -ArgumentList $arguments `
    -WindowStyle Normal `
    -Wait
