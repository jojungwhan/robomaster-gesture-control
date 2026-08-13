$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$logDirectory = Join-Path $projectRoot 'logs'
[System.IO.Directory]::CreateDirectory($logDirectory) | Out-Null
$timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$stdoutPath = Join-Path $logDirectory "s1_live_$timestamp.stdout.log"
$stderrPath = Join-Path $logDirectory "s1_live_$timestamp.stderr.log"
$screenPath = Join-Path $logDirectory "s1_app_state_$timestamp.png"
$statePath = Join-Path $logDirectory 'current_s1_live.json'
$workspaceRoot = Split-Path -Parent $projectRoot
$pythonPath = Join-Path $workspaceRoot '.venv-robomaster\Scripts\python.exe'
$bridgePath = Join-Path $projectRoot 'build\leap_hand_bridge.dll'

if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Python environment not found: $pythonPath"
}
if (-not (Test-Path -LiteralPath $bridgePath)) {
    throw "Leap bridge not found: $bridgePath"
}

Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class RoboMasterRestartWin32 {
    [DllImport("user32.dll")] public static extern bool IsWindow(IntPtr handle);
    [DllImport("user32.dll")] public static extern bool ShowWindowAsync(IntPtr handle, int command);
    [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr handle);
    [DllImport("user32.dll")] public static extern void SwitchToThisWindow(IntPtr handle, bool altTab);
    [DllImport("user32.dll")] public static extern void keybd_event(byte virtualKey, byte scanCode, uint flags, UIntPtr extraInfo);
}
'@

function Send-MovementKeyUps {
    foreach ($virtualKey in @(0x57, 0x41, 0x53, 0x44)) {
        [RoboMasterRestartWin32]::keybd_event([byte]$virtualKey, 0, 0x0002, [UIntPtr]::Zero)
    }
}

# Always clear every supported movement key before replacing a controller.
Send-MovementKeyUps

$oldControllers = @(
    Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object {
            $_.Name -eq 'python.exe' -and
            $_.CommandLine -match 'python(.exe)?\s+-m\s+robomaster_gesture'
        }
)
$oldParents = @($oldControllers.ParentProcessId | Sort-Object -Unique)
foreach ($controller in $oldControllers) {
    Stop-Process -Id $controller.ProcessId -Force -ErrorAction SilentlyContinue
}
Start-Sleep -Milliseconds 500
foreach ($parentId in $oldParents) {
    $parent = Get-CimInstance Win32_Process -Filter "ProcessId=$parentId" -ErrorAction SilentlyContinue
    if ($parent -and $parent.Name -eq 'powershell.exe') {
        Stop-Process -Id $parentId -Force -ErrorAction SilentlyContinue
    }
}
Send-MovementKeyUps

$launchArguments = @(
    '-u',
    '-m', 'robomaster_gesture',
    '--live',
    '--transport', 's1-app',
    '--connection', 'sta',
    '--protocol', 'tcp',
    '--hand', 'right',
    '--bridge-dll', $bridgePath
)
$child = Start-Process -FilePath $pythonPath -ArgumentList $launchArguments `
    -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath

Start-Sleep -Seconds 3
$child.Refresh()
$status = 'running'
$exitCode = $null
if ($child.HasExited) {
    $status = 'exited'
    $exitCode = $child.ExitCode
}

$app = Get-Process -Name RoboMaster -ErrorAction Stop | Select-Object -First 1
$window = [IntPtr]$app.MainWindowHandle
if (-not [RoboMasterRestartWin32]::IsWindow($window)) {
    throw 'The RoboMaster app window is not available.'
}
[void][RoboMasterRestartWin32]::ShowWindowAsync($window, 9)
[void][RoboMasterRestartWin32]::SetForegroundWindow($window)
[RoboMasterRestartWin32]::SwitchToThisWindow($window, $true)
Start-Sleep -Milliseconds 800

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
$bounds = [System.Windows.Forms.SystemInformation]::VirtualScreen
$bitmap = New-Object System.Drawing.Bitmap $bounds.Width, $bounds.Height
$graphics = [System.Drawing.Graphics]::FromImage($bitmap)
try {
    $graphics.CopyFromScreen($bounds.Left, $bounds.Top, 0, 0, $bitmap.Size)
    $bitmap.Save($screenPath, [System.Drawing.Imaging.ImageFormat]::Png)
} finally {
    $graphics.Dispose()
    $bitmap.Dispose()
}

$stateJson = [pscustomobject]@{
    controller_process_id = $child.Id
    status = $status
    exit_code = $exitCode
    stdout_path = $stdoutPath
    stderr_path = $stderrPath
    screenshot_path = $screenPath
    started_at = (Get-Date).ToString('o')
} | ConvertTo-Json
[System.IO.File]::WriteAllText($statePath, $stateJson)

if ($status -ne 'running') {
    $stderrText = if (Test-Path -LiteralPath $stderrPath) {
        [System.IO.File]::ReadAllText($stderrPath)
    } else {
        ''
    }
    throw "Gesture controller exited with code $exitCode. $stderrText"
}

Write-Output "Gesture controller PID $($child.Id) is running."
Write-Output "Output: $stdoutPath"
Write-Output "Errors: $stderrPath"
