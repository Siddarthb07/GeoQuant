# Convenience wrapper when launched from backend/
& (Join-Path (Split-Path -Parent $PSScriptRoot) "run_all.ps1") @args
