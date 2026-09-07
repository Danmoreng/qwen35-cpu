[CmdletBinding()]
param([string]$SourceDir='.cache/llama.cpp',[string]$BuildDir='build-llama-calibration-cuda',[string]$CudaArchitectures='120')
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
if (!(Get-Command cl.exe -ErrorAction SilentlyContinue)) {
  $vswhere=Join-Path ${Env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
  $vsroot=& $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
  if (!$vsroot) {throw 'Install Visual Studio C++ Build Tools.'}
  $vcvars=Join-Path $vsroot 'VC/Auxiliary/Build/vcvars64.bat'
  $environment=cmd /s /c "`"$vcvars`" > nul && set"
  foreach($line in $environment) {
    if($line -match '^(.*?)=(.*)$') {Set-Item -Path "Env:$($matches[1])" -Value $matches[2]}
  }
}
$source=(Resolve-Path $SourceDir).Path
cmake -S "$root/tools/llama-calibration-cuda" -B $BuildDir -G Ninja -DCMAKE_BUILD_TYPE=Release "-DLLAMA_SOURCE_DIR=$source" "-DCMAKE_CUDA_ARCHITECTURES=$CudaArchitectures"
if($LASTEXITCODE -ne 0){throw 'CUDA collector configuration failed'}
cmake --build $BuildDir --target llama-calibration-cuda llama-calibration-resident-cuda --parallel 6
if($LASTEXITCODE -ne 0){throw 'CUDA collector build failed'}
