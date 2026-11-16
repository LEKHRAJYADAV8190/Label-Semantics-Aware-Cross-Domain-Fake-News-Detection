# Prints the saved result tables. It does not train a model.
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
python (Join-Path $PSScriptRoot 'scripts\run_demo.py') @args
