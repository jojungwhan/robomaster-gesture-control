param(
    [string]$AudioFile = '',
    [string]$Culture = 'en-US',
    [string]$WakeWord = 'robot',
    [double]$InitialSilenceSeconds = 1.0
)

$ErrorActionPreference = 'Stop'

function Write-SpeechEvent {
    param([hashtable]$Data)
    [Console]::Out.WriteLine(($Data | ConvertTo-Json -Compress))
    [Console]::Out.Flush()
}

try {
    Add-Type -AssemblyName System.Speech
    $recognizerInfo = [System.Speech.Recognition.SpeechRecognitionEngine]::InstalledRecognizers() |
        Where-Object { $_.Culture.Name -eq $Culture } |
        Select-Object -First 1
    if (-not $recognizerInfo) {
        throw "No installed Windows speech recognizer for culture $Culture."
    }

    $recognizer = New-Object System.Speech.Recognition.SpeechRecognitionEngine($recognizerInfo)
    $recognizedEventId = "RoboMasterSpeech.$PID.Recognized"
    $completedEventId = "RoboMasterSpeech.$PID.Completed"
    $audioLevelEventId = "RoboMasterSpeech.$PID.AudioLevel"
    $recognitionStarted = $false
    try {
        $prefix = $WakeWord.Trim().ToLowerInvariant()
        $commands = @(
            'forward', 'go forward', 'move forward', 'ahead',
            'back', 'backward', 'go backward', 'move backward', 'reverse',
            'left', 'move left', 'go left',
            'right', 'move right', 'go right',
            'forward left', 'forward right', 'back left', 'back right',
            'turn left', 'turn right', 'rotate left', 'rotate right'
        )
        $phrases = @('stop', 'halt', 'freeze', 'emergency stop')
        foreach ($command in $commands) {
            if ($prefix) {
                $phrases += "$prefix $command"
            } else {
                $phrases += $command
            }
        }
        if ($prefix) {
            $phrases += "$prefix stop"
            $phrases += "$prefix emergency stop"
        }

        $choices = New-Object System.Speech.Recognition.Choices
        $choices.Add([string[]]$phrases)
        $builder = New-Object System.Speech.Recognition.GrammarBuilder
        $builder.Culture = $recognizerInfo.Culture
        $builder.Append($choices)
        $grammar = New-Object System.Speech.Recognition.Grammar($builder)
        $recognizer.LoadGrammar($grammar)
        $recognizer.InitialSilenceTimeout = [TimeSpan]::FromSeconds($InitialSilenceSeconds)
        $recognizer.BabbleTimeout = [TimeSpan]::FromSeconds(2)
        $recognizer.EndSilenceTimeout = [TimeSpan]::FromMilliseconds(250)
        $recognizer.EndSilenceTimeoutAmbiguous = [TimeSpan]::FromMilliseconds(400)

        if ($AudioFile) {
            $resolvedAudio = (Resolve-Path -LiteralPath $AudioFile).Path
            $recognizer.SetInputToWaveFile($resolvedAudio)
            $sourceDescription = "audio file $resolvedAudio"
        } else {
            $recognizer.SetInputToDefaultAudioDevice()
            $sourceDescription = 'default microphone'
        }
        Write-SpeechEvent @{
            event = 'ready'
            message = "$($recognizerInfo.Description); $sourceDescription"
        }

        $null = Register-ObjectEvent -InputObject $recognizer `
            -EventName SpeechRecognized -SourceIdentifier $recognizedEventId
        $null = Register-ObjectEvent -InputObject $recognizer `
            -EventName RecognizeCompleted -SourceIdentifier $completedEventId
        $null = Register-ObjectEvent -InputObject $recognizer `
            -EventName AudioLevelUpdated -SourceIdentifier $audioLevelEventId

        # Async recognition lets this process publish the exact input stream's
        # audio energy while it listens.  Event callbacks are deliberately not
        # used because PowerShell runs callback output in a background job;
        # instead, the main loop drains the event queue and owns stdout.
        $recognizer.RecognizeAsync(
            [System.Speech.Recognition.RecognizeMode]::Multiple
        )
        $recognitionStarted = $true
        $recognitionCompleted = $false
        while (-not $recognitionCompleted) {
            Start-Sleep -Milliseconds 100
            $peakAudioLevel = [int]$recognizer.AudioLevel
            $pendingEvents = @(
                Get-Event -ErrorAction SilentlyContinue |
                    Where-Object {
                        $_.SourceIdentifier -eq $recognizedEventId -or
                        $_.SourceIdentifier -eq $completedEventId -or
                        $_.SourceIdentifier -eq $audioLevelEventId
                    } |
                    Sort-Object EventIdentifier
            )
            foreach ($pendingEvent in $pendingEvents) {
                try {
                    if ($pendingEvent.SourceIdentifier -eq $recognizedEventId) {
                        $result = $pendingEvent.SourceEventArgs.Result
                        Write-SpeechEvent @{
                            event = 'recognized'
                            text = $result.Text
                            confidence = [Math]::Round($result.Confidence, 6)
                        }
                    } elseif ($pendingEvent.SourceIdentifier -eq $audioLevelEventId) {
                        $peakAudioLevel = [Math]::Max(
                            $peakAudioLevel,
                            [int]$pendingEvent.SourceEventArgs.AudioLevel
                        )
                    } else {
                        $completionError = $pendingEvent.SourceEventArgs.Error
                        if ($null -ne $completionError) {
                            throw $completionError
                        }
                        $recognitionCompleted = $true
                    }
                } finally {
                    Remove-Event -EventIdentifier $pendingEvent.EventIdentifier `
                        -ErrorAction SilentlyContinue
                }
            }
            Write-SpeechEvent @{
                event = 'audio-level'
                level = [Math]::Max(0, [Math]::Min(100, $peakAudioLevel))
            }
        }
        Write-SpeechEvent @{ event = 'completed' }
    } finally {
        if ($recognitionStarted) {
            try { $recognizer.RecognizeAsyncCancel() } catch {}
        }
        foreach ($eventId in @($recognizedEventId, $completedEventId, $audioLevelEventId)) {
            Unregister-Event -SourceIdentifier $eventId -ErrorAction SilentlyContinue
            Get-Event -SourceIdentifier $eventId -ErrorAction SilentlyContinue |
                Remove-Event -ErrorAction SilentlyContinue
        }
        try { $recognizer.SetInputToNull() } catch {}
        $recognizer.Dispose()
    }
} catch {
    Write-SpeechEvent @{ event = 'error'; message = $_.Exception.Message }
    exit 1
}
