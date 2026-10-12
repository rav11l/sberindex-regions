# Скачивает выгрузки СберИндекса, нужные для обоих исследований, и сверяет контрольные суммы.
# Запуск из корня репозитория: powershell -ExecutionPolicy Bypass -File scripts\download.ps1
# Сайт отдаёт zip по адресу /api/dataset/v1/download/<набор>/csv с заголовком RqUID.
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$dir = "data\raw"
New-Item -ItemType Directory -Force $dir | Out-Null
$ids = @(
  "kolichestvo-vnutrennikh-turistov",
  "izmenenie-kolichestva-aktivnikh-torgovo-servisnikh-tochek-po-kategoriyam",
  "real_estate_deals",
  "potrebitelskaya-aktivnost-po-kategoriyam-tovarov-v-razreze-vozrastov",
  "dinamika-tsen-obyavlenii",
  "residential_real_estate_prices",
  "predlozheniya-nedvizhimosti",
  "nedelnaa-inflazia-v-razreze-analiticeskih-komponentov",
  "real-key-interest-rate"
)
foreach ($id in $ids) {
  $out = Join-Path $dir "si_$id.csv.zip"
  $headers = @{ RqUID = [guid]::NewGuid().ToString("N") }
  Invoke-WebRequest -Uri "https://sberindex.ru/api/dataset/v1/download/$id/csv" -Headers $headers -OutFile $out
  Write-Host ("{0,-74} {1,8:N0} байт" -f $id, (Get-Item $out).Length)
}
Write-Host "`nСверка хешей:"
python -c "import sys; sys.path.insert(0,'src'); from siregions import data; [print(f'{v:9s} {k}') for k,v in data.verify().items()]"
