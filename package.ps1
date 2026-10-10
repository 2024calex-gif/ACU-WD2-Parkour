# Assembles v2\dist\ from the built plugin DLL and a patch folder:
#   dist\plugins\WD2Vaults.dll          (written by build.ps1)
#   dist\plugins\WD2Vaults\             (cleared, then filled with mapping.json, every *.json from -PatchDir and
#                                        clipnames_parkour.tsv from v2\data, used by the flight recorder)
#   dist\plugins\WD2Vaults\variants\    (with -VariantsDir: one subfolder per variant, each with its *.baked.json and
#                                        mapping.json; report.md and images are skipped)
#   dist\plugins\WD2Vaults\variant.txt  (with -DefaultVariant: that variant name, no trailing newline)
#   dist\install.ps1, dist\uninstall.ps1 (written here)
# Only dist\plugins\WD2Vaults\ is cleared; nothing else in dist is deleted.
param(
    [string]$PatchDir = (Join-Path $PSScriptRoot 'out\patches'),
    [string]$VariantsDir,
    [string]$DefaultVariant
)
$ErrorActionPreference = 'Stop'

function Fail([string]$msg) { Write-Host "PACKAGE FAILED: $msg"; exit 1 }

$dist    = Join-Path $PSScriptRoot 'dist'
$plugins = Join-Path $dist 'plugins'
$dll     = Join-Path $plugins 'WD2Vaults.dll'
$target  = Join-Path $plugins 'WD2Vaults'

$mappingSrc = Join-Path $PatchDir 'mapping.json'
if (-not (Test-Path -LiteralPath $mappingSrc -PathType Leaf)) {
    Fail "mapping.json not found at $mappingSrc (pass -PatchDir <folder containing mapping.json>)"
}
if (-not (Test-Path -LiteralPath $dll -PathType Leaf)) { Fail "plugin DLL missing: $dll (run build.ps1 first)" }
$clipNamesSrc = Join-Path $PSScriptRoot 'data\clipnames_parkour.tsv'
if (-not (Test-Path -LiteralPath $clipNamesSrc -PathType Leaf)) {
    Fail "clip names missing: $clipNamesSrc (run python v2\tools\make_clipnames.py first)"
}

# variant sources: one subfolder of -VariantsDir per variant
$variantSrcs = @()
if ($VariantsDir) {
    if (-not (Test-Path -LiteralPath $VariantsDir -PathType Container)) { Fail "variants folder not found: $VariantsDir" }
    $variantSrcs = @(Get-ChildItem -LiteralPath $VariantsDir -Directory | Sort-Object Name)
    if ($variantSrcs.Count -eq 0) { Write-Host "WARNING: no variant subfolders in $VariantsDir" }
    foreach ($v in $variantSrcs) {
        if (-not (Test-Path -LiteralPath (Join-Path $v.FullName 'mapping.json') -PathType Leaf)) {
            Fail "variant '$($v.Name)' has no mapping.json in $($v.FullName)"
        }
    }
}
if ($DefaultVariant) {
    if (-not $VariantsDir) { Fail "-DefaultVariant needs -VariantsDir" }
    if (@($variantSrcs | ForEach-Object { $_.Name }) -notcontains $DefaultVariant) {
        Fail "default variant '$DefaultVariant' is not a subfolder of $VariantsDir"
    }
}

try { $map = Get-Content -LiteralPath $mappingSrc -Raw | ConvertFrom-Json }
catch { Fail "mapping.json is not valid JSON: $($_.Exception.Message)" }

$jsonFiles = @(Get-ChildItem -LiteralPath $PatchDir -Filter '*.json' -File)
$names = @($jsonFiles | ForEach-Object { $_.Name })
foreach ($c in @($map.clips)) {
    if ($c.source -and ($names -notcontains $c.source)) {
        Write-Host "WARNING: clip '$($c.name)' names source '$($c.source)', which is not in $PatchDir"
    }
}

# clear only our own folder, then fill it
if (Test-Path -LiteralPath $target) { Remove-Item -LiteralPath $target -Recurse -Force }
New-Item -ItemType Directory -Force -Path $target | Out-Null
foreach ($j in $jsonFiles) { Copy-Item -LiteralPath $j.FullName -Destination $target -Force }
if (-not (Test-Path -LiteralPath (Join-Path $target 'mapping.json'))) { Fail "mapping.json was not copied to $target" }
Copy-Item -LiteralPath $clipNamesSrc -Destination $target -Force
if (-not (Test-Path -LiteralPath (Join-Path $target 'clipnames_parkour.tsv'))) { Fail "clipnames_parkour.tsv was not copied to $target" }

# variants: clear the variants folder and variant.txt, then refill from -VariantsDir / -DefaultVariant
$variantsTarget = Join-Path $target 'variants'
$variantFile    = Join-Path $target 'variant.txt'
if (Test-Path -LiteralPath $variantsTarget) { Remove-Item -LiteralPath $variantsTarget -Recurse -Force }
if (Test-Path -LiteralPath $variantFile) { Remove-Item -LiteralPath $variantFile -Force }
$variantCount = 0
if ($VariantsDir) {
    New-Item -ItemType Directory -Force -Path $variantsTarget | Out-Null
    foreach ($v in $variantSrcs) {
        $vDst = Join-Path $variantsTarget $v.Name
        New-Item -ItemType Directory -Force -Path $vDst | Out-Null
        $baked = 0
        foreach ($f in @(Get-ChildItem -LiteralPath $v.FullName -File)) {
            if ($f.Name -eq 'mapping.json' -or $f.Name -like '*.baked.json') {
                Copy-Item -LiteralPath $f.FullName -Destination $vDst -Force
                if ($f.Name -like '*.baked.json') { $baked++ }
            }
        }
        if (-not (Test-Path -LiteralPath (Join-Path $vDst 'mapping.json') -PathType Leaf)) { Fail "mapping.json was not copied to $vDst" }
        if ($baked -eq 0) { Write-Host "WARNING: variant '$($v.Name)' has no *.baked.json" }
        $variantCount++
    }
}
if ($DefaultVariant) {
    [System.IO.File]::WriteAllText($variantFile, $DefaultVariant, (New-Object -TypeName System.Text.UTF8Encoding -ArgumentList $false))
}

$installScript = @'
# Installs the WD2Vaults plugin next to ACUFixes. Run it from the dist folder.
# Adds or replaces WD2Vaults.dll and the files of the package's WD2Vaults folder, including its variants\ folder
# and variant.txt. Before copying, the game's old variants\ folder and variant.txt are removed, so stale variants do not linger.
# Never deletes anything else, and leaves an existing WD2Vaults\dump and WD2Vaults\captures folder untouched.
$ErrorActionPreference = 'Stop'
$game = "C:\Program Files (x86)\Steam\steamapps\common\Assassin's Creed Unity\ACUFixes\plugins"
$src  = Join-Path $PSScriptRoot 'plugins'

if (Get-Process ACU -ErrorAction SilentlyContinue) { Write-Host "ACU.exe is running. Close the game first, then run this again."; exit 1 }
if (-not (Test-Path -LiteralPath (Join-Path $game 'ACUFixes.dll'))) { Write-Host "ACUFixes plugin folder not found: $game"; exit 1 }
$pkgDll    = Join-Path $src 'WD2Vaults.dll'
$pkgFolder = Join-Path $src 'WD2Vaults'
if (-not (Test-Path -LiteralPath $pkgDll -PathType Leaf)) { Write-Host "package incomplete: $pkgDll missing"; exit 1 }
if (-not (Test-Path -LiteralPath $pkgFolder -PathType Container)) { Write-Host "package incomplete: $pkgFolder missing"; exit 1 }

Copy-Item -LiteralPath $pkgDll -Destination (Join-Path $game 'WD2Vaults.dll') -Force
$dstFolder      = Join-Path $game 'WD2Vaults'
$dstVariants    = Join-Path $dstFolder 'variants'
$dstVariantFile = Join-Path $dstFolder 'variant.txt'
New-Item -ItemType Directory -Force -Path $dstFolder | Out-Null
# old variants and variant.txt go first, so only what this package ships is left
if (Test-Path -LiteralPath $dstVariants) { Remove-Item -LiteralPath $dstVariants -Recurse -Force }
Remove-Item -LiteralPath $dstVariantFile -Force -ErrorAction SilentlyContinue
# file by file, so an existing dump folder (and anything else in it) is left alone
# (variant.txt is copied here too, when the package has one)
$n = 0
foreach ($f in @(Get-ChildItem -LiteralPath $pkgFolder -File)) { Copy-Item -LiteralPath $f.FullName -Destination $dstFolder -Force; $n++ }
# the variants folder, recursively
$nv = 0
$pkgVariants = Join-Path $pkgFolder 'variants'
if (Test-Path -LiteralPath $pkgVariants -PathType Container) {
    Copy-Item -LiteralPath $pkgVariants -Destination $dstFolder -Recurse -Force
    if (-not (Test-Path -LiteralPath $dstVariants -PathType Container)) { Write-Host "variants copy failed: $dstVariants missing"; exit 1 }
    $nv = @(Get-ChildItem -LiteralPath $dstVariants -Recurse -File).Count
}
Write-Host "installed: $game\WD2Vaults.dll and $n file(s) in $dstFolder"
Write-Host "variants: $nv file(s) in $dstVariants"
if (Test-Path -LiteralPath $dstVariantFile -PathType Leaf) { Write-Host "variant.txt installed: $dstVariantFile" }
if (Test-Path -LiteralPath (Join-Path $dstFolder 'dump')) { Write-Host "existing dump folder kept: $dstFolder\dump" }
'@

$uninstallScript = @'
# Removes what install.ps1 added: WD2Vaults.dll, the WD2Vaults folder (with its variants\ folder and variant.txt) and WD2Vaults-log.log.
# Before anything is deleted, WD2Vaults\dump, WD2Vaults\captures (all files), variant.txt and the log are copied to
# v2\data\game_logs\uninstall_<yyyyMMdd_HHmm>\. The variants\ folder is not copied (it comes from the package).
# If that copy fails, nothing is deleted.
$ErrorActionPreference = 'Stop'
$game = "C:\Program Files (x86)\Steam\steamapps\common\Assassin's Creed Unity\ACUFixes\plugins"

if (Get-Process ACU -ErrorAction SilentlyContinue) { Write-Host "ACU.exe is running. Close the game first, then run this again."; exit 1 }

$dumpDir       = Join-Path $game 'WD2Vaults\dump'
$capDir        = Join-Path $game 'WD2Vaults\captures'
$variantDir    = Join-Path $game 'WD2Vaults\variants'
$variantFile   = Join-Path $game 'WD2Vaults\variant.txt'
$logFile       = Join-Path $game 'WD2Vaults-log.log'
$hasDump       = Test-Path -LiteralPath $dumpDir -PathType Container
$hasCaps       = Test-Path -LiteralPath $capDir -PathType Container
$hasVariantTxt = Test-Path -LiteralPath $variantFile -PathType Leaf
$hasLog        = Test-Path -LiteralPath $logFile -PathType Leaf

if ($hasDump -or $hasCaps -or $hasVariantTxt -or $hasLog) {
    $v2     = Split-Path -Parent $PSScriptRoot
    $backup = Join-Path $v2 ('data\game_logs\uninstall_' + (Get-Date -Format 'yyyyMMdd_HHmm'))
    try {
        New-Item -ItemType Directory -Force -Path $backup | Out-Null
        if ($hasDump) {
            Copy-Item -LiteralPath $dumpDir -Destination (Join-Path $backup 'dump') -Recurse -Force -ErrorAction Stop
            if (-not (Test-Path -LiteralPath (Join-Path $backup 'dump'))) { throw "dump copy not found in $backup" }
        }
        if ($hasCaps) {
            Copy-Item -LiteralPath $capDir -Destination (Join-Path $backup 'captures') -Recurse -Force -ErrorAction Stop
            if (-not (Test-Path -LiteralPath (Join-Path $backup 'captures'))) { throw "captures copy not found in $backup" }
        }
        if ($hasVariantTxt) {
            Copy-Item -LiteralPath $variantFile -Destination $backup -Force -ErrorAction Stop
            if (-not (Test-Path -LiteralPath (Join-Path $backup 'variant.txt'))) { throw "variant.txt copy not found in $backup" }
        }
        if ($hasLog) {
            Copy-Item -LiteralPath $logFile -Destination $backup -Force -ErrorAction Stop
            if (-not (Test-Path -LiteralPath (Join-Path $backup 'WD2Vaults-log.log'))) { throw "log copy not found in $backup" }
        }
    }
    catch {
        Write-Host "backup failed, nothing was deleted: $($_.Exception.Message)"
        exit 1
    }
    Write-Host "saved dump/captures/variant.txt/log to $backup"
}
else {
    Write-Host "no dump, captures, variant.txt or log to save"
}

Remove-Item -LiteralPath (Join-Path $game 'WD2Vaults.dll') -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath $variantDir -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath $variantFile -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath (Join-Path $game 'WD2Vaults') -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath $logFile -Force -ErrorAction SilentlyContinue
$left = @(@('WD2Vaults.dll', 'WD2Vaults', 'WD2Vaults-log.log') | Where-Object { Test-Path -LiteralPath (Join-Path $game $_) })
if ($left.Count -gt 0) { Write-Host "NOT removed (locked or in use?): $($left -join ', ')"; exit 1 }
Write-Host "removed WD2Vaults.dll, the WD2Vaults folder (variants and variant.txt with it) and WD2Vaults-log.log"
'@

New-Item -ItemType Directory -Force -Path $dist | Out-Null
Set-Content -LiteralPath (Join-Path $dist 'install.ps1') -Value $installScript -Encoding ASCII
Set-Content -LiteralPath (Join-Path $dist 'uninstall.ps1') -Value $uninstallScript -Encoding ASCII

Write-Host "PACKAGE OK"
Write-Host "  patches from: $PatchDir"
Write-Host ("  copied {0} json file(s) to {1}" -f $jsonFiles.Count, $target)
if ($VariantsDir) { Write-Host ("  variants from: {0} ({1} folder(s) to {2})" -f $VariantsDir, $variantCount, $variantsTarget) }
if ($DefaultVariant) { Write-Host "  default variant: $DefaultVariant (variant.txt)" }
Write-Host "  dll: $dll"
Write-Host "  wrote: $(Join-Path $dist 'install.ps1'), $(Join-Path $dist 'uninstall.ps1')"
exit 0
