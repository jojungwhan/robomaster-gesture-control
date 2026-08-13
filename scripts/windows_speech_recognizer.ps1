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

        while ($true) {
            try {
                $result = $recognizer.Recognize(
                    [TimeSpan]::FromSeconds($InitialSilenceSeconds)
                )
            } catch [System.Speech.Recognition.InitialSilenceTimeoutException] {
                $result = $null
            }
            if ($null -ne $result) {
                Write-SpeechEvent @{
                    event = 'recognized'
                    text = $result.Text
                    confidence = [Math]::Round($result.Confidence, 6)
                }
            } elseif ($AudioFile) {
                break
            }
        }
        Write-SpeechEvent @{ event = 'completed' }
    } finally {
        try { $recognizer.SetInputToNull() } catch {}
        $recognizer.Dispose()
    }
} catch {
    Write-SpeechEvent @{ event = 'error'; message = $_.Exception.Message }
    exit 1
}
