# Requires PowerShell 7. Each matrix entry supplies a name, executable, and args.
[CmdletBinding()]
param(
  [Parameter(Mandatory)][string]$Matrix,
  [string]$OutputDir = "benchmarks/run-$(Get-Date -Format yyyyMMdd-HHmmss)",
  [ValidateRange(1,100)][int]$Runs = 3,
  [ValidateRange(0,100)][int]$WarmupRuns = 1,
  [long]$Affinity = 0,
  [ValidateRange(0,1)][double]$SpeedTolerance = 0.03
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
    logical_cpus = [Environment]::ProcessorCount
    git_revision = $null; git_diff_sha256 = $null
    matrix = $cases; binaries = @{}; inputs = @{}; commands = @()
    speed_gates = @{ target_ratio=1.0; equivalence_tolerance=$SpeedTolerance; intermediate_ratio=0.95 }
    source_files = @{}
  }
  foreach ($sourceFile in @(Get-ChildItem src,include,scripts,tests,tools -Recurse -File) + @(Get-Item CMakeLists.txt,AGENTS.md)) {
    if ($sourceFile.FullName -notmatch '[\\/]__pycache__[\\/]') {
      $metadata.source_files[$sourceFile.FullName] = (Get-FileHash -LiteralPath $sourceFile.FullName -Algorithm SHA256).Hash
    }
  }
  if (Test-Path -LiteralPath '.git') {
    $metadata.git_revision = (& git rev-parse HEAD)
    $diffText = (& git diff --no-ext-diff HEAD | Out-String)
    $metadata.git_diff_sha256 = [Convert]::ToHexString(
      [Security.Cryptography.SHA256]::HashData([Text.Encoding]::UTF8.GetBytes($diffText)))
  }
  foreach ($case in $cases) {
    $exe = (Resolve-Path -LiteralPath $case.executable).Path
    $metadata.binaries[$exe] = (Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash
    $inputPaths = @()
    for ($argumentIndex=0; $argumentIndex -lt ($case.args.Count-1); $argumentIndex++) {
      $flag = [string]$case.args[$argumentIndex]
      $value = [string]$case.args[$argumentIndex+1]
      if ($flag -in @('--cpu-q4-h128','--cpu-gguf','--weights','--tokens-file','--forced-tokens-file','--prompt-tokens-file')) {
        $inputPaths += $value
      }
      if ($flag -in @('--hf-model-dir','--model-dir')) {
        foreach ($name in @('config.json','tokenizer.json','tokenizer_config.json')) {
          $candidatePath = Join-Path $value $name
          if (Test-Path -LiteralPath $candidatePath -PathType Leaf) { $inputPaths += $candidatePath }
        }
      }
    }
    foreach ($inputCandidate in $inputPaths) {
      $inputPath = (Resolve-Path -LiteralPath $inputCandidate).Path
      if (!$metadata.inputs.ContainsKey($inputPath)) {
        $metadata.inputs[$inputPath] = @{
          sha256 = (Get-FileHash -LiteralPath $inputPath -Algorithm SHA256).Hash
          bytes = (Get-Item -LiteralPath $inputPath).Length
        }
      }
    }
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
      if ((Get-FileHash -LiteralPath $info.FileName -Algorithm SHA256).Hash -ne $metadata.binaries[$info.FileName]) {
        throw 'Executable changed during benchmark'
      }
      $metadata.commands += @{ executable=$info.FileName; args=@($info.ArgumentList); run=$run; case=$case.name }
      $metadata | ConvertTo-Json -Depth 15 | Set-Content -LiteralPath "$out/metadata.json"
      $process = [Diagnostics.Process]::new(); $process.StartInfo = $info
      $started = $false
      $processTimer = [Diagnostics.Stopwatch]::StartNew()
      try {
        [void]$process.Start()
        $started = $true
        if ($Affinity -ne 0) { $process.ProcessorAffinity = [IntPtr]$Affinity }
        $stdout = $process.StandardOutput.ReadToEndAsync()
        $stderr = $process.StandardError.ReadToEndAsync()
        $process.WaitForExit()
        $processTimer.Stop()
        # Some platforms no longer expose process counters after exit. Native
        # profiles also report their own peak RSS before process teardown.
        $peakWorkingSet = $null
        try { $peakWorkingSet = $process.PeakWorkingSet64 } catch { }
        $stdout.GetAwaiter().GetResult() | Set-Content -LiteralPath "$out/$stem.stdout.txt"
        $stderr.GetAwaiter().GetResult() | Set-Content -LiteralPath "$out/$stem.stderr.txt"
        if ($process.ExitCode -ne 0) { throw "Benchmark failed: $($case.name); see $stem.stderr.txt" }
      } finally {
        if ($started -and !$process.HasExited) { $process.Kill($true); $process.WaitForExit() }
        $process.Dispose()
      }
      $data = Get-Content -Raw -LiteralPath $profile | ConvertFrom-Json
      $row = [ordered]@{ name=$case.name; run=$run; warmup=($run -lt $WarmupRuns); profile="$stem.json"; peak_working_set_bytes=$peakWorkingSet; process_wall_time_ms=$processTimer.Elapsed.TotalMilliseconds.ToString('R', [Globalization.CultureInfo]::InvariantCulture) }
      foreach ($property in $data.PSObject.Properties) {
        if ($property.Value -is [ValueType]) {
          # Keep CSV numbers portable even when the host uses decimal commas.
          $row[$property.Name] = if ($property.Value -is [IFormattable]) {
            $property.Value.ToString($null, [Globalization.CultureInfo]::InvariantCulture)
          } else { $property.Value }
        }
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
