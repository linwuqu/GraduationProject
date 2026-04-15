param(
    [switch]$NoActivate
)

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$venvPath = Join-Path $projectRoot '.venv'
$activateScript = Join-Path $venvPath 'Scripts\Activate.ps1'
$cacheDir = Join-Path $projectRoot '.uv-cache'

foreach ($name in @('VIRTUAL_ENV', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'PIP_NO_INDEX')) {
    Remove-Item ("Env:{0}" -f $name) -ErrorAction Ignore
}

$proxyUrl = 'http://127.0.0.1:7897'
$env:NO_PROXY = 'localhost,127.0.0.1,::1'
$env:HTTP_PROXY = $proxyUrl
$env:HTTPS_PROXY = $proxyUrl
$env:ALL_PROXY = $proxyUrl
$env:UV_CACHE_DIR = $cacheDir
$env:UV_HTTP_TIMEOUT = '300'

if (-not (Test-Path $cacheDir)) {
    New-Item -ItemType Directory -Path $cacheDir | Out-Null
}

Set-Location $projectRoot

if (-not (Test-Path $activateScript)) {
    Write-Error ".venv not found. Expected activation script at $activateScript"
    exit 1
}

if (-not $NoActivate) {
    . $activateScript
}

Write-Host "Project root: $projectRoot"
Write-Host ".venv: $venvPath"
Write-Host "UV_CACHE_DIR: $env:UV_CACHE_DIR"
Write-Host "UV_HTTP_TIMEOUT: $env:UV_HTTP_TIMEOUT"
Write-Host "VIRTUAL_ENV: $env:VIRTUAL_ENV"
Write-Host "Proxy: $env:HTTP_PROXY"
