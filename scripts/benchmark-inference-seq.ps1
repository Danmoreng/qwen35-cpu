# Requires PowerShell 7. Each matrix entry supplies a name, executable, and args.
[CmdletBinding()]
param(
  [Parameter(Mandatory)][string]$Matrix,
  [string]$OutputDir = "benchmarks/run-$(Get-Date -Format yyyyMMdd-HHmmss)",
  [ValidateRange(1,100)][int]$Runs = 3,
  [ValidateRange(0,100)][int]$WarmupRuns = 1,
  [long]$Affinity = 0
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$cases = @(Get-Content -Raw -LiteralPath $Matrix | ConvertFrom-Json)
if (!$cases.Count) { throw 'Empty benchmark matrix' }
if (Test-Path -LiteralPath $OutputDir) { throw 'Use a new output directory for each run' }
$mutex = [Threading.Mutex]::new($false, 'qwen35_cpu_sequential_benchmark')
$held = $false
try {
  try { $held = $mutex.WaitOne(0) } catch [Threading.AbandonedMutexException] { $held = $true }
  if (!$held) { throw 'Another sequential benchmark is running' }
  $out = (New-Item -ItemType Directory -Path $OutputDir).FullName
  $metadata = @{
    utc = [DateTime]::UtcNow.ToString('o'); runs = $Runs; warmup_runs = $WarmupRuns
    affinity = $Affinity; os = [Environment]::OSVersion.ToString()
    logical_cpus = [Environment]::ProcessorCount; git_revision = (& git rev-parse HEAD 2>$null)
    matrix = $cases; binaries = @{}
  }
  foreach ($case in $cases) {
    $exe = (Resolve-Path -LiteralPath $case.executable).Path
    $metadata.binaries[$exe] = (Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash
  }
  $metadata | ConvertTo-Json -Depth 15 | Set-Content -LiteralPath "$out/metadata.json"
  $rows = @()
  for ($run = 0; $run -lt ($WarmupRuns + $Runs); $run++) {
    # Alternate case order to reduce monotonic temperature/order bias.
    $indices = @(0..($cases.Count - 1))
    if ($run % 2) { [array]::Reverse($indices) }
    foreach ($index in $indices) {
      $case = $cases[$index]
      $stem = "case-$index-run-$run"
      $profile = "$out/$stem.json"
      $info = [Diagnostics.ProcessStartInfo]::new()
      $info.FileName = (Resolve-Path -LiteralPath $case.executable).Path
      $info.UseShellExecute = $false
      $info.CreateNoWindow = $true
      $info.RedirectStandardOutput = $true
      $info.RedirectStandardError = $true
      foreach ($arg in $case.args) { $info.ArgumentList.Add([string]$arg) }
      $info.ArgumentList.Add('--profile-json'); $info.ArgumentList.Add($profile)
      $process = [Diagnostics.Process]::new(); $process.StartInfo = $info
      $started = $false
      try {
        [void]$process.Start()
        $started = $true
        if ($Affinity -ne 0) { $process.ProcessorAffinity = [IntPtr]$Affinity }
        $stdout = $process.StandardOutput.ReadToEndAsync()
        $stderr = $process.StandardError.ReadToEndAsync()
        $process.WaitForExit()
        $stdout.GetAwaiter().GetResult() | Set-Content -LiteralPath "$out/$stem.stdout.txt"
        $stderr.GetAwaiter().GetResult() | Set-Content -LiteralPath "$out/$stem.stderr.txt"
        if ($process.ExitCode -ne 0) { throw "Benchmark failed: $($case.name); see $stem.stderr.txt" }
      } finally {
        if ($started -and !$process.HasExited) { $process.Kill($true); $process.WaitForExit() }
        $process.Dispose()
      }
      $data = Get-Content -Raw -LiteralPath $profile | ConvertFrom-Json
      $row = [ordered]@{ name=$case.name; run=$run; warmup=($run -lt $WarmupRuns); profile="$stem.json" }
      foreach ($property in $data.PSObject.Properties) {
        if ($property.Value -is [ValueType]) { $row[$property.Name] = $property.Value }
      }
      $rows += [pscustomobject]$row
      Write-Host "Completed $($case.name), run $run"
    }
  }
  # Normalize columns across different engines before writing the finished CSV.
  $columns = @($rows | ForEach-Object { $_.PSObject.Properties.Name } | Select-Object -Unique)
  $rows | Select-Object $columns | Export-Csv -NoTypeInformation -LiteralPath "$out/results.csv"
  Write-Host "Results: $out/results.csv"
} finally {
  if ($held) { $mutex.ReleaseMutex() }
  $mutex.Dispose()
}
