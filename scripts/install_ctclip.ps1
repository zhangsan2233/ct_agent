param(
  [string]$Python = ""
)

$ErrorActionPreference = "Stop"
if (-not $Python -or -not (Test-Path -LiteralPath $Python)) {
  $Python = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
}

& $Python -c "import torch; assert torch.cuda.is_available()"
if ($LASTEXITCODE -ne 0) {
  & $Python -m pip install `
    torch==2.7.1+cu128 torchvision==0.22.1+cu128 `
    --index-url https://download.pytorch.org/whl/cu128
}

& $Python -m pip install `
  "transformers>=4.44.0" `
  "einops>=0.6.0" `
  "beartype>=0.18.0" `
  "ftfy>=6.2.0" `
  "regex>=2024.5.15" `
  "vector-quantize-pytorch==1.1.2" `
  "nibabel>=5.2.0"

$ProjectPython = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
& $ProjectPython (Join-Path $PSScriptRoot "download_ctclip_assets.py")
