<#
Maintains a bounded queue of Chrome downloads for the Stage-2 CT manifest.

Requires Chrome to be signed in, configured to download directly to DownloadDir
without a per-file Save As prompt, and kept focused after the initial delay.
Completed files are detected by their final .nii.gz name; Chrome's temporary
.crdownload files are never treated as complete.
#>
[CmdletBinding()]
param(
    [string]$Manifest = (Join-Path $PSScriptRoot '..\artifacts\ctclip_stage2\train_manifest_1000.csv'),
    [string]$DownloadDir = 'F:\temp_download',
    [int]$StartIndex = 1,
    [int]$Count = 1000,
    [int]$MaxConcurrent = 6,
    [int]$PollSeconds = 5,
    [int]$LaunchDelaySeconds = 2,
    [int]$FocusDelaySeconds = 8
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Windows.Forms

$manifestPath = (Resolve-Path -LiteralPath $Manifest).Path
$downloadPath = (Resolve-Path -LiteralPath $DownloadDir).Path
$rows = @(Import-Csv -LiteralPath $manifestPath)
if ($StartIndex -lt 1 -or $StartIndex -gt $rows.Count) { throw "StartIndex must be between 1 and $($rows.Count)." }
if ($Count -lt 1 -or $MaxConcurrent -lt 1 -or $PollSeconds -lt 1) { throw 'Count, MaxConcurrent, and PollSeconds must be positive.' }

$selected = @($rows | Select-Object -Skip ($StartIndex - 1) -First $Count)
$runId = "{0}_{1}" -f (Get-Date -Format 'yyyyMMdd_HHmmss'), $PID
$logPath = Join-Path (Split-Path -Parent $manifestPath) "chrome_download_concurrent_$runId.log"

function Write-QueueLog([string]$Message) {
    # Antivirus, an editor, or a live log viewer can briefly lock a file on Windows.
    # Logging must never terminate an active browser download queue.
    for ($attempt = 1; $attempt -le 5; $attempt++) {
        try {
            [System.IO.File]::AppendAllText(
                $logPath,
                $Message + [Environment]::NewLine,
                [System.Text.UTF8Encoding]::new($false)
            )
            return
        }
        catch [System.IO.IOException] {
            if ($attempt -lt 5) { Start-Sleep -Milliseconds (200 * $attempt) }
        }
    }
    Write-Warning "Could not write the log just now; the download queue will continue."
}

Write-QueueLog "Started $(Get-Date -Format s): rows=$($selected.Count), max_concurrent=$MaxConcurrent"

function Test-Complete([object]$Row) {
    $path = Join-Path $downloadPath $Row.volume_name
    return (Test-Path -LiteralPath $path -PathType Leaf) -and ((Get-Item -LiteralPath $path).Length -gt 0)
}

function Test-InProgress([object]$Row) {
    $temporary = Join-Path $downloadPath ($Row.volume_name + '.crdownload')
    return Test-Path -LiteralPath $temporary -PathType Leaf
}

function Submit-Download([object]$Row, [int]$Index) {
    [System.Windows.Forms.Clipboard]::SetText($Row.download_url)
    [System.Windows.Forms.SendKeys]::SendWait('^l')
    Start-Sleep -Milliseconds 150
    [System.Windows.Forms.SendKeys]::SendWait('^v')
    Start-Sleep -Milliseconds 150
    [System.Windows.Forms.SendKeys]::SendWait('{ENTER}')
    Write-QueueLog "$(Get-Date -Format s),SUBMITTED,$Index,$($Row.case_id),$($Row.volume_name)"
    Write-Host "submitted [$Index/$($rows.Count)] $($Row.volume_name)"
    Start-Sleep -Seconds $LaunchDelaySeconds
}

Write-Host "Chrome must be signed in and focused within $FocusDelaySeconds seconds."
Write-Host "Keeping at most $MaxConcurrent active downloads; press Ctrl+C to stop submissions."
Start-Sleep -Seconds $FocusDelaySeconds

$cursor = 0
$active = @{}
$completed = 0
$skipped = 0
while ($cursor -lt $selected.Count -or $active.Count -gt 0) {
    foreach ($key in @($active.Keys)) {
        $job = $active[$key]
        if (Test-Complete $job.Row) {
            Write-QueueLog "$(Get-Date -Format s),COMPLETED,$($job.Index),$($job.Row.case_id),$($job.Row.volume_name)"
            Write-Host "completed [$($job.Index)/$($rows.Count)] $($job.Row.volume_name)"
            $active.Remove($key)
            $completed++
        }
    }
    while ($active.Count -lt $MaxConcurrent -and $cursor -lt $selected.Count) {
        $row = $selected[$cursor]
        $index = $StartIndex + $cursor
        $cursor++
        if (Test-Complete $row) {
            Write-QueueLog "$(Get-Date -Format s),SKIPPED_EXISTING,$index,$($row.case_id),$($row.volume_name)"
            Write-Host "skipped existing [$index/$($rows.Count)] $($row.volume_name)"
            $skipped++
            continue
        }
        if (Test-InProgress $row) {
            Write-QueueLog "$(Get-Date -Format s),ADOPTED_IN_PROGRESS,$index,$($row.case_id),$($row.volume_name)"
            Write-Host "waiting for existing download [$index/$($rows.Count)] $($row.volume_name)"
            $active[$index] = [PSCustomObject]@{ Index = $index; Row = $row }
            continue
        }
        Submit-Download $row $index
        $active[$index] = [PSCustomObject]@{ Index = $index; Row = $row }
    }
    if ($active.Count -gt 0) { Start-Sleep -Seconds $PollSeconds }
}
Write-QueueLog "Finished $(Get-Date -Format s): completed=$completed skipped=$skipped"
Write-Host "Finished. Completed=$completed; skipped existing=$skipped; log=$logPath"
