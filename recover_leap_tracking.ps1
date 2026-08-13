param(
    [string]$DeviceInstanceId = 'USB\VID_F182&PID_0003\5&1EA4C09&0&3'
)

$ErrorActionPreference = 'Stop'

$logDirectory = Join-Path $PSScriptRoot 'logs'
$statusPath = Join-Path $logDirectory 'leap_recovery_status.json'
New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null

function Write-RecoveryStatus {
    param(
        [string]$Status,
        [string]$Message,
        [int]$PreviousServicePid = 0,
        [int]$CurrentServicePid = 0
    )

    [ordered]@{
        status = $Status
        message = $Message
        device_instance_id = $DeviceInstanceId
        previous_service_pid = $PreviousServicePid
        current_service_pid = $CurrentServicePid
        updated_at = (Get-Date).ToString('o')
    } | ConvertTo-Json | Set-Content -LiteralPath $statusPath -Encoding UTF8
}

try {
    $service = Get-CimInstance Win32_Service -Filter "Name='UltraleapTracking'"
    $previousPid = [int]$service.ProcessId
    Write-RecoveryStatus -Status 'stopping' -Message 'Stopping the wedged tracking service.' -PreviousServicePid $previousPid

    # These are controllers launched by this project during the current session.
    # Retiring them here avoids multiple LeapC clients after recovery.
    foreach ($processId in 34636, 13148) {
        Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
    }

    Stop-Service -Name UltraleapTracking -Force -ErrorAction SilentlyContinue
    $serviceController = Get-Service -Name UltraleapTracking
    $serviceController.WaitForStatus('Stopped', [TimeSpan]::FromSeconds(15))

    Write-RecoveryStatus -Status 'cycling_usb' -Message 'Restarting the Leap USB composite device.' -PreviousServicePid $previousPid
    $pnpOutput = & "$env:SystemRoot\System32\pnputil.exe" /restart-device $DeviceInstanceId 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "pnputil failed with exit code $LASTEXITCODE`: $($pnpOutput -join ' ')"
    }

    Start-Sleep -Seconds 2
    Start-Service -Name UltraleapTracking
    $serviceController = Get-Service -Name UltraleapTracking
    $serviceController.WaitForStatus('Running', [TimeSpan]::FromSeconds(20))
    Start-Sleep -Seconds 3

    $service = Get-CimInstance Win32_Service -Filter "Name='UltraleapTracking'"
    $device = Get-PnpDevice -InstanceId $DeviceInstanceId -ErrorAction Stop
    if ($device.Status -ne 'OK') {
        throw "Leap USB device returned status '$($device.Status)' after restart."
    }

    Write-RecoveryStatus `
        -Status 'recovered' `
        -Message 'Leap USB device and Ultraleap tracking service restarted successfully.' `
        -PreviousServicePid $previousPid `
        -CurrentServicePid ([int]$service.ProcessId)
    exit 0
} catch {
    $currentPid = 0
    try {
        $currentPid = [int](Get-CimInstance Win32_Service -Filter "Name='UltraleapTracking'").ProcessId
    } catch {
    }
    Write-RecoveryStatus `
        -Status 'failed' `
        -Message $_.Exception.Message `
        -PreviousServicePid $previousPid `
        -CurrentServicePid $currentPid
    exit 1
}
