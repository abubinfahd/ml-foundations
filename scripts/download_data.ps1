<#
.SYNOPSIS
    Download the datasets used in ml-foundations, with SHA-256 verification.

.DESCRIPTION
    Files land in data\ (gitignored). The script is idempotent: a file that
    already exists with a valid checksum is skipped.

.PARAMETER Target
    A single dataset name, or "all" for every dataset.
    Datasets: penguins wine titanic iris adult mnist fashion

.EXAMPLE
    .\scripts\download_data.ps1 fashion
    .\scripts\download_data.ps1 all
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Target
)

$ErrorActionPreference = "Stop"

# Repo root is the parent of the folder holding this script.
$Root = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $Root "data"
New-Item -ItemType Directory -Force -Path $DataDir | Out-Null

# name|file|url|sha256 — one entry per file
$Manifest = @(
    "penguins|penguins.csv|https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins.csv|f204db2c753b0937caac3cb35258562c14f073e4bbc76be24b4c51ce22767a93"
    "wine|winequality-red.csv|https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv|4a402cf041b025d4566d954c3b9ba8635a3a8a01e039005d97d6a710278cf05e"
    "titanic|titanic.csv|https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv|4a437fde05fe5264e1701a7387ac6fb75393772ba38bb2c9c566405af5af4bd7"
    "iris|iris.data|https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data|6f608b71a7317216319b4d27b4d9bc84e6abd734eda7872b71a458569e2656c0"
    "adult|adult.data|https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data|5b00264637dbfec36bdeaab5676b0b309ff9eb788d63554ca0a249491c86603d"
    "mnist|mnist-train-images-idx3-ubyte.gz|https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz|440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609"
    "mnist|mnist-train-labels-idx1-ubyte.gz|https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz|3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c"
    "mnist|mnist-t10k-images-idx3-ubyte.gz|https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz|8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6"
    "mnist|mnist-t10k-labels-idx1-ubyte.gz|https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz|f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6"
    "fashion|fashion-train-images-idx3-ubyte.gz|https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/train-images-idx3-ubyte.gz|3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84"
    "fashion|fashion-train-labels-idx1-ubyte.gz|https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/train-labels-idx1-ubyte.gz|a04f17134ac03560a47e3764e11b92fc97de4d1bfaf8ba1a3aa29af54cc90845"
    "fashion|fashion-t10k-images-idx3-ubyte.gz|https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/t10k-images-idx3-ubyte.gz|346e55b948d973a97e58d2351dde16a484bd415d4595297633bb08f03db6a073"
    "fashion|fashion-t10k-labels-idx1-ubyte.gz|https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/t10k-labels-idx1-ubyte.gz|67da17c76eaffca5446c3361aaab5c3cd6d1c2608764d35dfb1850b086bf8dd5"
)

$Datasets = "penguins wine titanic iris adult mnist fashion"

function Show-Usage {
    Write-Host "Usage: .\scripts\download_data.ps1 <dataset|all>"
    Write-Host "Datasets: $Datasets"
}

function Test-Checksum {
    param([string]$File, [string]$Expected)
    if (-not (Test-Path -LiteralPath $File)) { return $false }
    $actual = (Get-FileHash -LiteralPath $File -Algorithm SHA256).Hash.ToLower()
    return $actual -eq $Expected.ToLower()
}

# TLS 1.2 for older Windows PowerShell (5.1); harmless on newer.
try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 } catch {}

$matched = $false
foreach ($entry in $Manifest) {
    $name, $file, $url, $sha = $entry -split '\|'
    if ($Target -ne "all" -and $Target -ne $name) { continue }
    $matched = $true

    $dest = Join-Path $DataDir $file
    if (Test-Checksum -File $dest -Expected $sha) {
        Write-Host "[skip] $file (already downloaded, checksum OK)"
        continue
    }

    Write-Host "[get ] $file"
    Invoke-WebRequest -Uri $url -OutFile $dest -UseBasicParsing

    if (Test-Checksum -File $dest -Expected $sha) {
        Write-Host "[ ok ] $file (checksum verified)"
    }
    else {
        Write-Error "[FAIL] $file - checksum mismatch. Delete data\$file and retry, or open an issue if the problem persists."
        exit 1
    }
}

if (-not $matched) {
    Write-Host "Unknown dataset: $Target"
    Show-Usage
    exit 1
}

Write-Host "Done."
