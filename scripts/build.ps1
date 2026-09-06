[CmdletBinding()]
param([string]$BuildDir="build",[string]$Target="all",[string]$Configuration="Release")
$ErrorActionPreference="Stop"
Set-StrictMode -Version Latest
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
$build=if([IO.Path]::IsPathRooted($BuildDir)){$BuildDir}else{Join-Path $root $BuildDir}
cmake -S $root -B $build -G Ninja "-DCMAKE_BUILD_TYPE=$Configuration"
if($LASTEXITCODE -ne 0){throw 'Configuration failed.'}
cmake --build $build --target $Target
if($LASTEXITCODE -ne 0){throw 'Build failed.'}
