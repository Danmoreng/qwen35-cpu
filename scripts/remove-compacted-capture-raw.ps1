[CmdletBinding()]
param([Parameter(Mandatory)][string]$Root,[Parameter(Mandatory)][string]$Manifest)
$ErrorActionPreference='Stop'
$captureRoot=(Resolve-Path -LiteralPath $Root).Path
$plan=Get-Content -Raw -LiteralPath $Manifest | ConvertFrom-Json
if ([IO.Path]::GetFullPath($plan.root) -ne $captureRoot) {throw 'Cleanup root mismatch'}
$verified=@($plan.files | ForEach-Object {
  $candidate=[IO.Path]::GetFullPath($_.path)
  if (!$candidate.StartsWith($captureRoot+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase) -or [IO.Path]::GetExtension($candidate) -ne '.f32') {
    throw 'Cleanup path outside capture root'
  }
  if ((Get-Item -LiteralPath $candidate).Length -ne $_.bytes) {throw 'Capture size changed'}
  if (!(Test-Path -LiteralPath (Join-Path (Split-Path -Parent $candidate) 'diagonal-statistics.json'))) {throw 'Missing compacted statistics'}
  $candidate
})
for($start=0;$start -lt $verified.Count;$start+=256) {
  $end=[Math]::Min($start+255,$verified.Count-1)
  Remove-Item -LiteralPath $verified[$start..$end]
}
Write-Output ('Removed {0:N2} GiB of verified compacted raw captures.' -f ($plan.bytes/1GB))
