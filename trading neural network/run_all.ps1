param(
    [string]$BindHost = "127.0.0.1",
    [int]$Port = 8020,
    [switch]$SkipInstall,
    [switch]$Reload,
    [switch]$SkipResearch,
    [string]$ConfigPath = "config.yaml",
    [string]$CsvPath = "",
    [double]$InitialCapital = 100000,
    [double]$PositionFraction = 0.10,
    [double]$BrokerageFeeBps = 10,
    [double]$SlippageBps = 5
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$backendPath = Join-Path $projectRoot "backend"

if (-not (Test-Path $backendPath)) {
    throw "Could not find backend folder at: $backendPath"
}

Set-Location $backendPath

$venvPath = Join-Path $backendPath ".venv"
$venvPython = Join-Path $backendPath ".venv\Scripts\python.exe"

function Test-PythonExe([string]$ExePath) {
    if (-not (Test-Path $ExePath)) {
        return $false
    }
    try {
        & $ExePath -c "import sys; print(sys.version)" 2>$null | Out-Null
        return ($LASTEXITCODE -eq 0)
    } catch {
        return $false
    }
}

function Test-PythonModule([string]$ExePath, [string]$ModuleName) {
    try {
        & $ExePath -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('$ModuleName') else 1)" 2>$null
        return ($LASTEXITCODE -eq 0)
    } catch {
        return $false
    }
}

function Get-SystemPythonLauncher {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        try {
            & py -3 -c "import sys; print(sys.executable)" 2>$null | Out-Null
            if ($LASTEXITCODE -eq 0) {
                return "py"
            }
        } catch {}
    }
    if (Get-Command python -ErrorAction SilentlyContinue) {
        try {
            & python -c "import sys; print(sys.executable)" 2>$null | Out-Null
            if ($LASTEXITCODE -eq 0) {
                return "python"
            }
        } catch {}
    }
    return $null
}

function Find-SystemPythonExe {
    $candidates = @()
    try {
        $wherePy = & where.exe python 2>$null
        if ($wherePy) {
            $candidates += ($wherePy -split "`r?`n" | Where-Object { $_ -match "python\.exe$" })
        }
    } catch {}
    $candidates += @(
        "$env:LocalAppData\\Programs\\Python\\Python312\\python.exe",
        "$env:LocalAppData\\Programs\\Python\\Python311\\python.exe",
        "$env:LocalAppData\\Programs\\Python\\Python310\\python.exe",
        "$env:LocalAppData\\Programs\\Python\\Python39\\python.exe",
        "$env:ProgramFiles\\Python312\\python.exe",
        "$env:ProgramFiles\\Python311\\python.exe",
        "$env:ProgramFiles\\Python310\\python.exe",
        "$env:ProgramFiles\\Python39\\python.exe"
    )
    foreach ($candidate in $candidates | Select-Object -Unique) {
        if (Test-PythonExe $candidate) {
            return $candidate
        }
    }
    return $null
}

if (-not (Test-PythonExe $venvPython)) {
    $launcher = Get-SystemPythonLauncher
    $pythonExe = Find-SystemPythonExe
    if ((-not $launcher) -and (-not $pythonExe)) {
        throw "Python 3.9+ is not installed or not available in PATH. Install Python and run again."
    }

    if (Test-Path $venvPath) {
        Write-Host "Existing .venv is invalid. Recreating..."
        Remove-Item $venvPath -Recurse -Force
    }

    Write-Host "Creating virtual environment..."
    if ($launcher -eq "py") {
        & py -3 -m venv .venv
    } elseif ($launcher -eq "python") {
        & python -m venv .venv
    } else {
        & $pythonExe -m venv .venv
    }
}

if (-not (Test-PythonExe $venvPython)) {
    throw "Virtual environment creation failed or Python inside .venv is not runnable: $venvPython"
}

if (-not (Test-PythonModule $venvPython "pip")) {
    & $venvPython -m ensurepip --upgrade 2>$null | Out-Null
}

$missingCoreModules = @()
if (-not (Test-PythonModule $venvPython "uvicorn")) { $missingCoreModules += "uvicorn" }
if (-not (Test-PythonModule $venvPython "fastapi")) { $missingCoreModules += "fastapi" }

if ($SkipInstall -and $missingCoreModules.Count -gt 0) {
    Write-Host "SkipInstall was requested, but required packages are missing: $($missingCoreModules -join ', '). Installing dependencies..."
}

if ((-not $SkipInstall) -or $missingCoreModules.Count -gt 0) {
    Write-Host "Installing dependencies..."
    & $venvPython -m pip install --upgrade pip
    & $venvPython -m pip install -r requirements.txt
}

if (-not (Test-PythonModule $venvPython "uvicorn")) {
    throw "uvicorn is still missing in backend/.venv. Run: backend\\.venv\\Scripts\\python.exe -m pip install -r backend\\requirements.txt"
}

if ((-not (Test-Path ".env")) -and (Test-Path ".env.example")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created backend/.env from .env.example (edit if using live broker keys)."
}

if (-not $SkipResearch) {
    Write-Host ""
    Write-Host "Running reproducible research pipeline (walk-forward + backtest + metrics)..."
    $trainArgs = @(
        "train.py",
        "--config", $ConfigPath,
        "--initial-capital", "$InitialCapital",
        "--position-fraction", "$PositionFraction",
        "--brokerage-fee-bps", "$BrokerageFeeBps",
        "--slippage-bps", "$SlippageBps"
    )
    if (-not [string]::IsNullOrWhiteSpace($CsvPath)) {
        $trainArgs += @("--csv-path", $CsvPath)
    }
    & $venvPython @trainArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Research pipeline failed. Fix the error above or use -SkipResearch to start API only."
    }
    Write-Host "Research pipeline completed."
}

Write-Host ""
Write-Host "Starting GeoQuant Neural Trader at http://$BindHost`:$Port"
Write-Host "Press Ctrl+C to stop."
Write-Host ""

$uvicornArgs = @("-m", "uvicorn", "app.main:app", "--host", $BindHost, "--port", "$Port")
if ($Reload) {
    $uvicornArgs += "--reload"
}
& $venvPython @uvicornArgs
