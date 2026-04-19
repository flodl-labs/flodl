# bench-publish.ps1 — Run publication benchmarks with locked GPU clocks.
#
# WSL2 cannot lock GPU clocks because nvidia-smi inside WSL is a shim
# that doesn't expose the driver's clock control plane. Since WSL2 shares
# the host GPU driver, clocks locked from Windows apply to WSL2 workloads.
#
# Usage (from an admin PowerShell):
#
#   .\benchmarks\bench-publish.ps1                              — auto-detects base clock
#   .\benchmarks\bench-publish.ps1 -Rounds 20                   — more rounds
#   .\benchmarks\bench-publish.ps1 -Clock 1800                  — override clock frequency
#   .\benchmarks\bench-publish.ps1 -Output benchmarks/report.txt — save report to file
#
# Requires:
#   - Admin PowerShell (for nvidia-smi clock control)
#   - WSL2 with the flodl dev container set up

param(
    [int]$Rounds    = 10,
    [int]$Clock     = 0,
    [string]$Output = ''
)

$ErrorActionPreference = 'Stop'

# --- Locate nvidia-smi ---
# nvidia-smi may not be in PATH depending on driver install method.
# Search common locations if the bare command isn't found.
$nvSmi = 'nvidia-smi'
if (-not (Get-Command $nvSmi -ErrorAction SilentlyContinue)) {
    $candidates = @(
        "$env:SystemRoot\System32\nvidia-smi.exe",
        "${env:ProgramFiles}\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
        "${env:ProgramW6432}\NVIDIA Corporation\NVSMI\nvidia-smi.exe"
    )
    # Studio/newer drivers put nvidia-smi in the DriverStore — pick the newest.
    $dsHit = Get-ChildItem -Path "$env:SystemRoot\System32\DriverStore\FileRepository" `
        -Recurse -Filter 'nvidia-smi.exe' -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($dsHit) { $candidates += $dsHit.FullName }

    foreach ($c in $candidates) {
        if (Test-Path $c) { $nvSmi = $c; break }
    }
    if (-not (Test-Path $nvSmi)) {
        Write-Host 'ERROR: nvidia-smi not found.' -ForegroundColor Red
        Write-Host 'Searched: PATH, System32, NVIDIA Corporation\NVSMI, DriverStore'
        Write-Host 'Install the NVIDIA driver or add nvidia-smi to PATH.'
        exit 1
    }
    Write-Host "=== Found nvidia-smi at: $nvSmi ===" -ForegroundColor Yellow
}

# --- Detect GPU name ---
$gpuName = (& $nvSmi --query-gpu=name --format=csv,noheader).Trim()

# --- Auto-detect base clock if not specified ---
if ($Clock -eq 0) {
    # Try default applications clock first (works on most desktop GPUs).
    $line = & $nvSmi --query-gpu=clocks.default_applications.graphics --format=csv,noheader 2>$null
    $match = [regex]::Match("$line", '(\d+)')
    if ($match.Success) {
        $Clock = [int]$match.Groups[1].Value
        Write-Host ('=== Detected {0} base clock: {1} MHz ===' -f $gpuName, $Clock) -ForegroundColor Green
    }

    # Fallback: parse "Default Applications Clocks" section from verbose output.
    if ($Clock -eq 0) {
        $section = $false
        foreach ($l in (& $nvSmi -q -d CLOCK 2>$null)) {
            if ($l -match 'Default Applications Clocks') { $section = $true; continue }
            if ($section -and $l -match 'Graphics\s*:\s*(\d+)') {
                $Clock = [int]$Matches[1]
                Write-Host ('=== Detected {0} base clock: {1} MHz (from verbose query) ===' -f $gpuName, $Clock) -ForegroundColor Green
                break
            }
            if ($section -and $l -match '^\s*$') { $section = $false }
        }
    }

    # Last resort: use current clock as a reasonable default.
    if ($Clock -eq 0) {
        $line = & $nvSmi --query-gpu=clocks.current.graphics --format=csv,noheader 2>$null
        $match = [regex]::Match("$line", '(\d+)')
        if ($match.Success) {
            $Clock = [int]$match.Groups[1].Value
            Write-Host ('=== Could not detect base clock for {0} ===' -f $gpuName) -ForegroundColor Yellow
            Write-Host ('=== Using current clock: {0} MHz (specify -Clock for exact control) ===' -f $Clock) -ForegroundColor Yellow
        }
    }

    if ($Clock -eq 0) {
        Write-Host 'ERROR: Could not detect any GPU clock frequency.' -ForegroundColor Red
        Write-Host 'Specify manually:  .\bench-publish.ps1 -Clock 1800'
        Write-Host ''
        Write-Host 'Debug output:'
        & $nvSmi -q -d CLOCK
        exit 1
    }
} else {
    Write-Host ('=== {0} — using specified clock: {1} MHz ===' -f $gpuName, $Clock) -ForegroundColor Green
}

# --- Lock GPU clocks ---
Write-Host ('=== Locking GPU clocks to {0} MHz ===' -f $Clock) -ForegroundColor Cyan
& $nvSmi -lgc ('{0},{0}' -f $Clock)
Write-Host ''

try {
    # --- Resolve repo root inside WSL ---
    # The script lives in benchmarks/; repo root is its parent. wslpath
    # handles any checkout location (native WSL \\wsl.localhost\... or
    # Windows-mounted drives /mnt/c/...) without hard-coding.
    $scriptParent = (Resolve-Path "$PSScriptRoot\..").Path
    $wslRepo = (& wsl wslpath -a $scriptParent).Trim()
    if ([string]::IsNullOrEmpty($wslRepo)) {
        Write-Host 'ERROR: wslpath did not resolve the repo root.' -ForegroundColor Red
        Write-Host ('  Windows path was: {0}' -f $scriptParent)
        exit 1
    }

    # --- Run benchmarks in WSL2 ---
    # Clock is passed through so run.sh tags the report metadata with
    # "locked at X MHz (host)" — the actual lock already happened above.
    Write-Host ('=== Starting bench-publish in WSL2, {0} rounds ({1}) ===' -f $Rounds, $wslRepo) -ForegroundColor Cyan
    $wslCmd = 'cd {0}; fdl bench publish --rounds {1} --lock-clocks {2}' -f $wslRepo, $Rounds, $Clock
    if ($Output -ne '') {
        $wslCmd += ' --output {0}' -f $Output
    }
    wsl -e bash -c $wslCmd
}
finally {
    # --- Always unlock clocks ---
    Write-Host ''
    Write-Host '=== Unlocking GPU clocks ===' -ForegroundColor Cyan
    & $nvSmi -rgc
}
