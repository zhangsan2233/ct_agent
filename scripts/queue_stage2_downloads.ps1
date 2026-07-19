<#
Queues gated CT-RATE downloads in the currently focused, already-authorized Chrome.

The script never handles a Hugging Face token. It reads direct URLs from the
Stage-2 manifest, puts each one on the clipboard, opens it in Chrome with
Ctrl+L/Ctrl+V/Enter, then waits. Keep Chrome focused after the start delay.
#>
[CmdletBinding()]
param(
    [string]$Manifest = (Join-Path $PSScriptRoot '..\artifacts\ctclip_stage2\train_manifest_1000.csv'),
    [int]$StartIndex = 1,
    [int]$Count = 1000,
    [int]$IntervalSeconds = 10,
    [int]$FocusDelaySeconds = 8,
    [switch]$ResumeFromLog,
    [string]$DownloadDir,
    [switch]$SkipExisting
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Windows.Forms

$manifestPath = (Resolve-Path -LiteralPath $Manifest).Path
$rows = @(Import-Csv -LiteralPath $manifestPath)
$logPath = Join-Path (Split-Path -Parent $manifestPath) 'chrome_download_queue.log'
if ($ResumeFromLog -and (Test-Path -LiteralPath $logPath)) {
    $lastIndex = 0
    Get-Content -LiteralPath $logPath | ForEach-Object {
        $fields = $_ -split ',', 4
        $parsed = 0
        if ($fields.Count -ge 2 -and [int]::TryParse($fields[1], [ref]$parsed)) {
            $lastIndex = [Math]::Max($lastIndex, $parsed)
        }
    }
    if ($lastIndex -gt 0) { $StartIndex = $lastIndex + 1 }
}
if ($StartIndex -lt 1 -or $StartIndex -gt $rows.Count) {
    throw "StartIndex must be between 1 and $($rows.Count)."
}
if ($Count -lt 1) { throw 'Count must be positive.' }
if ($IntervalSeconds -lt 1) { throw 'IntervalSeconds must be positive.' }

$selected = @($rows | Select-Object -Skip ($StartIndex - 1) -First $Count)
if ($SkipExisting -and -not $DownloadDir) {
    throw 'Use -DownloadDir together with -SkipExisting.'
}
if ($DownloadDir) { $DownloadDir = (Resolve-Path -LiteralPath $DownloadDir).Path }
"Started $(Get-Date -Format s): $($selected.Count) links, index $StartIndex, interval $IntervalSeconds s" |
    Add-Content -LiteralPath $logPath -Encoding utf8

Write-Host "Chrome must already be signed in and focused within $FocusDelaySeconds seconds."
Write-Host "Queueing $($selected.Count) links from row $StartIndex; press Ctrl+C in this window to stop."
Start-Sleep -Seconds $FocusDelaySeconds

for ($offset = 0; $offset -lt $selected.Count; $offset++) {
    $row = $selected[$offset]
    $index = $StartIndex + $offset
    $destination = if ($DownloadDir) { Join-Path $DownloadDir $row.volume_name } else { $null }
    if ($SkipExisting -and $destination -and (Test-Path -LiteralPath $destination -PathType Leaf)) {
        "$(Get-Date -Format s),$index,$($row.case_id),SKIPPED_EXISTING" |
            Add-Content -LiteralPath $logPath -Encoding utf8
        Write-Host "[$index/$($rows.Count)] skipped existing $($row.volume_name)"
        continue
    }
    [System.Windows.Forms.Clipboard]::SetText($row.download_url)
    [System.Windows.Forms.SendKeys]::SendWait('^l')
    Start-Sleep -Milliseconds 150
    [System.Windows.Forms.SendKeys]::SendWait('^v')
    Start-Sleep -Milliseconds 150
    [System.Windows.Forms.SendKeys]::SendWait('{ENTER}')
    "$(Get-Date -Format s),$index,$($row.case_id),$($row.download_url)" |
        Add-Content -LiteralPath $logPath -Encoding utf8
    Write-Host "[$index/$($rows.Count)] $($row.case_id)"
    if ($offset -lt $selected.Count - 1) { Start-Sleep -Seconds $IntervalSeconds }
}

Write-Host "Finished. Log: $logPath"
