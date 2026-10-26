$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
Set-Location $PSScriptRoot

Write-Host "== 1. Poetry ==" -ForegroundColor Cyan
if (-not (Get-Command poetry -ErrorAction SilentlyContinue)) {
    (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
    $env:Path = "$env:APPDATA\Python\Scripts;$env:Path"
}
poetry --version
poetry config virtualenvs.in-project true

Write-Host "== 2. Dependances (flask, openai, python-dotenv) ==" -ForegroundColor Cyan
poetry add flask openai python-dotenv

Write-Host "== 3. Tailwind CSS (binaire standalone Windows) ==" -ForegroundColor Cyan
$tw = ".\tailwindcss.exe"
$expected = "E0E260CE048014E9268F6237FF18F8CCF02CEF521CBD0AE04E82C2CDF7AA3955"
if (Test-Path .\tw_parts) {
    Write-Host "Assemblage de tailwindcss.exe depuis tw_parts..." -ForegroundColor Yellow
    if (Test-Path $tw) { Remove-Item $tw -Force }
    $out = [IO.File]::Create((Join-Path $PSScriptRoot "tailwindcss.exe"))
    Get-ChildItem .\tw_parts\tailwind.part* | Sort-Object Name | ForEach-Object {
        $b = [IO.File]::ReadAllBytes($_.FullName); $out.Write($b, 0, $b.Length)
    }
    $out.Close()
    if ((Get-FileHash $tw -Algorithm SHA256).Hash -ne $expected) { throw "tailwindcss.exe corrompu" }
    Remove-Item .\tw_parts -Recurse -Force
    Write-Host "tailwindcss.exe OK" -ForegroundColor Green
}
if ((Test-Path $tw) -and ((Get-Item $tw).Length -lt 90MB)) {
    Write-Host "tailwindcss.exe incomplet, nouveau telechargement..." -ForegroundColor Yellow
    Remove-Item $tw -Force
}
if (-not (Test-Path $tw)) {
    curl.exe -L --fail --retry 3 -o tailwindcss.exe https://github.com/tailwindlabs/tailwindcss/releases/latest/download/tailwindcss-windows-x64.exe
    if ($LASTEXITCODE -ne 0) { throw "Echec du telechargement de Tailwind" }
}
Unblock-File $tw
.\tailwindcss.exe -i .\static\css\input.css -o .\static\dist\css\style.css

Write-Host ""
Write-Host "OK ! Lance run.bat (serveur Flask) et tailwind.bat (watch CSS)." -ForegroundColor Green
