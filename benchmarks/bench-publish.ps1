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

# --- Detect GPU name ---
$gpuName = (nvidia-smi --query-gpu=name --format=csv,noheader).Trim()

# --- Auto-detect base clock if not specified ---
if ($Clock -eq 0) {
    # Try default applications clock first (works on most desktop GPUs).
    $line = nvidia-smi --query-gpu=clocks.default_applications.graphics --format=csv,noheader 2>$null
    $match = [regex]::Match("$line", '(\d+)')
    if ($match.Success) {
        $Clock = [int]$match.Groups[1].Value
        Write-Host ('=== Detected {0} base clock: {1} MHz ===' -f $gpuName, $Clock) -ForegroundColor Green
    }

    # Fallback: parse "Default Applications Clocks" section from verbose output.
    if ($Clock -eq 0) {
        $section = $false
        foreach ($l in (nvidia-smi -q -d CLOCK 2>$null)) {
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
        $line = nvidia-smi --query-gpu=clocks.current.graphics --format=csv,noheader 2>$null
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
        nvidia-smi -q -d CLOCK
        exit 1
    }
} else {
    Write-Host ('=== {0} — using specified clock: {1} MHz ===' -f $gpuName, $Clock) -ForegroundColor Green
}

# --- Lock GPU clocks ---
Write-Host ('=== Locking GPU clocks to {0} MHz ===' -f $Clock) -ForegroundColor Cyan
nvidia-smi -lgc ('{0},{0}' -f $Clock)
Write-Host ''

try {
    # --- Run benchmarks in WSL2 ---
    # CLOCK= (empty) skips the lock attempt inside WSL since we handle it here.
    Write-Host ('=== Starting bench-publish in WSL2, {0} rounds ===' -f $Rounds) -ForegroundColor Cyan
    $outputArg = ''
    if ($Output -ne '') {
        $outputArg = ' OUTPUT={0}' -f $Output
    }
    $wslCmd = 'cd /home/peta/src/fab2s/ai/rdl; make bench-publish ROUNDS={0} CLOCK={1}' -f $Rounds, $outputArg
    wsl -e bash -c $wslCmd
}
finally {
    # --- Always unlock clocks ---
    Write-Host ''
    Write-Host '=== Unlocking GPU clocks ===' -ForegroundColor Cyan
    nvidia-smi -rgc
}
