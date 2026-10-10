# Builds the WD2Vaults v0.2 plugin (Release x64) from plugin\ACUFixes-151-src\WD2Vaults_v2 and copies the DLL to
# v2\dist\plugins\WD2Vaults.dll. Exits with 1 on any failure.
$ErrorActionPreference = 'Stop'

function Fail([string]$msg) { Write-Host "BUILD FAILED: $msg"; exit 1 }

$root    = Split-Path -Parent $PSScriptRoot
$sln     = Join-Path $root 'plugin\ACUFixes-151-src'
$proj    = Join-Path $sln 'WD2Vaults_v2\WD2Vaults.vcxproj'
$msbuild = 'C:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\MSBuild.exe'
$built   = Join-Path $sln 'build\x64\Release\plugins_v2\WD2Vaults.dll'
$distDir = Join-Path $PSScriptRoot 'dist\plugins'

if (-not (Test-Path -LiteralPath $msbuild)) { Fail "MSBuild not found: $msbuild" }
if (-not (Test-Path -LiteralPath $proj))    { Fail "project not found: $proj" }

# SolutionDir must end with a backslash; written as \\ so the argument parser keeps it before the closing quote.
$solutionDir = '/p:SolutionDir=' + $sln + '\\'

# The CommonLibACU projects hard-code PlatformToolset v142, which is not installed here (only v145 is).
# A global property overrides those hard-coded values for every project in the build, without editing them.
& $msbuild $proj /t:Build /p:Configuration=Release /p:Platform=x64 $solutionDir /p:PlatformToolset=v145 /m /v:minimal
if ($LASTEXITCODE -ne 0) { Fail "MSBuild exit code $LASTEXITCODE" }

if (-not (Test-Path -LiteralPath $built)) { Fail "build reported success but the DLL is missing: $built" }

New-Item -ItemType Directory -Force -Path $distDir | Out-Null
$dest = Join-Path $distDir 'WD2Vaults.dll'
Copy-Item -LiteralPath $built -Destination $dest -Force

$f = Get-Item -LiteralPath $dest
Write-Host ("BUILD OK: {0}" -f $f.FullName)
Write-Host ("  size: {0:N0} bytes" -f $f.Length)
Write-Host ("  timestamp: {0:yyyy-MM-dd HH:mm:ss}" -f $f.LastWriteTime)
exit 0
