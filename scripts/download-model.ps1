[CmdletBinding()]
param(
  [Parameter(Mandatory)][string]$Repo,
  [Parameter(Mandatory)][ValidatePattern('^[0-9a-f]{40}$')][string]$Revision,
  [string]$Destination = 'models/qwen3.5-0.8b'
)
$ErrorActionPreference = 'Stop'
if ($Repo -notmatch '^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$') { throw 'Expected Hugging Face owner/model' }
if (Test-Path -LiteralPath $Destination) { throw 'Choose a new destination directory' }
$files = @('model.q35h','config.json','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt','chat_template.jinja','LICENSE','NOTICE','README.md','quantization.json')
$base = "https://huggingface.co/$Repo/resolve/$Revision"
$folder = (New-Item -ItemType Directory -Path $Destination).FullName
$ProgressPreference = 'SilentlyContinue'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
Invoke-WebRequest -UseBasicParsing -Uri "$base/SHA256SUMS" -OutFile "$folder/SHA256SUMS"
$hashes = @{}
foreach ($line in Get-Content -LiteralPath "$folder/SHA256SUMS") {
  if ($line -notmatch '^([0-9a-f]{64})  ([A-Za-z0-9_.-]+)$') { throw 'Invalid checksum manifest' }
  if ($hashes.ContainsKey($matches[2])) { throw 'Duplicate manifest entry' }
  $hashes[$matches[2]] = $matches[1]
}
if ($hashes.Count -ne $files.Count) { throw 'Unexpected manifest contents' }
foreach ($file in $files) {
  if (!$hashes.ContainsKey($file)) { throw "Missing checksum: $file" }
  Write-Host "Downloading $file"
  $partial = Join-Path $folder "$file.part"
  Invoke-WebRequest -UseBasicParsing -Uri "$base/$file" -OutFile $partial
  if ((Get-FileHash -LiteralPath $partial -Algorithm SHA256).Hash -ne $hashes[$file]) { throw "Checksum mismatch: $file" }
  Move-Item -LiteralPath $partial -Destination (Join-Path $folder $file)
}
Write-Host "Verified model: $folder"
