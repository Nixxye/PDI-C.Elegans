# Treina a Mask R-CNN do colega (submodule cnn/) na GPU NVIDIA (CUDA) com o dataset do projeto.
#
#   powershell -ExecutionPolicy Bypass -File tools\train_maskrcnn.ps1            # treino completo (10 epocas)
#   powershell -ExecutionPolicy Bypass -File tools\train_maskrcnn.ps1 -Smoke     # teste de 1 minuto
#   powershell -ExecutionPolicy Bypass -File tools\train_maskrcnn.ps1 -Epochs 20
#
# Se for interrompido, rode de novo: ele retoma de runs/cnn_train/checkpoint.pth.
# Saidas: runs/cnn_train/{checkpoint.pth, model_<epoca>.pth} e o log em runs/cnn_train/log.txt (fora do submodule).
param(
    [int]$Epochs = 10,
    [int]$BatchSize = 2,
    [double]$Lr = 0.0025,        # 0,00125 e o valor do colega para batch 1; escalado linearmente para batch 2
    [int]$Workers = 2,
    [switch]$Smoke
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
$py = Join-Path $repo ".venv\Scripts\python.exe"
$data = Join-Path $repo "data\csb-1_extracted\csb-1_dataset"
$out = Join-Path $repo $(if ($Smoke) { "runs\cnn_smoke_launcher" } else { "runs\cnn_train" })
New-Item -ItemType Directory -Force $out | Out-Null

if (-not (Test-Path (Join-Path $data "coco_annotations\train_annotations.json"))) { throw "Dataset nao encontrado em $data" }
& $py -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"
if ($LASTEXITCODE -ne 0) { throw "CUDA indisponivel neste ambiente (.venv)" }

$argsList = @(
    "cnn\detection\train.py", "--device", "cuda", "--amp",
    "--data-path", $data,
    "--weights", "MaskRCNN_ResNet50_FPN_Weights.DEFAULT",
    "--epochs", $Epochs, "--lr", $Lr, "--lr-steps", "6", "8",
    "--batch-size", $BatchSize, "--workers", $Workers,
    "--output-dir", $out
)
if ($Smoke) { $argsList += @("--epochs", "1", "--max-train-images", "8", "--max-validation-images", "4") }
$ckpt = Join-Path $out "checkpoint.pth"
if ((Test-Path $ckpt) -and -not $Smoke) { $argsList += @("--resume", $ckpt); Write-Host "Retomando de $ckpt" }

$env:PYTHONUTF8 = "1"
$ErrorActionPreference = "Continue"  # avisos do Python vao para stderr e nao devem abortar o script
Write-Host "Treinando na GPU; log em $out\log.txt"
& $py @argsList 2>&1 | Tee-Object -FilePath (Join-Path $out "log.txt") -Append
if ($LASTEXITCODE -ne 0) { throw "treino terminou com erro (codigo $LASTEXITCODE)" }
Write-Host "Pronto. Pesos em $ckpt"
