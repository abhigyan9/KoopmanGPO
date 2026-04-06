param()

$ErrorActionPreference = 'Stop'

function Get-Setting {
    param(
        [string]$Name,
        [string]$Default
    )
    $value = [System.Environment]::GetEnvironmentVariable($Name)
    if ([string]::IsNullOrWhiteSpace($value)) {
        return $Default
    }
    return $value
}

$PYTHON_BIN = Get-Setting 'PYTHON_BIN' 'python'
$SYSTEM = Get-Setting 'SYSTEM' 'chaotic_lorenz'
$timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$OUTDIR = Get-Setting 'OUTDIR' (Join-Path 'Figures' "numtraj_sweep_$timestamp")
$NUM_TEST = Get-Setting 'NUM_TEST' '40'
$LIFTING_ORDER = Get-Setting 'LIFTING_ORDER' '19'
$MAX_ITER = Get-Setting 'MAX_ITER' '2500'
$LEARN_RATE = Get-Setting 'LEARN_RATE' '0.001'
$DEVICE = Get-Setting 'DEVICE' 'cuda:0'
$TRAIN_METHOD = Get-Setting 'TRAIN_METHOD' 'Horizon'
$ROUTINE = Get-Setting 'ROUTINE' 'Z_only'
$HP1_SCALE = Get-Setting 'HP1_SCALE' '1.0'
$OPT_W1 = Get-Setting 'OPT_W1' '1.0'
$OPT_W2 = Get-Setting 'OPT_W2' '1.0'
$OPT_W3 = Get-Setting 'OPT_W3' '0.0'
$SEED_Z = Get-Setting 'SEED_Z' '1234'
$SEED_HP = Get-Setting 'SEED_HP' '1234'
$SLEEP_BETWEEN_RUNS = Get-Setting 'SLEEP_BETWEEN_RUNS' '10'
$TRAJ_LIST_RAW = Get-Setting 'TRAJ_LIST' '20 50 100 150 200 250'
$FAIL_FAST = Get-Setting 'FAIL_FAST' '0'

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$runnerPy = Join-Path $scriptDir 'run_igpk_numtraj.py'
$logDir = Join-Path $OUTDIR 'logs'
$indexTsv = Join-Path $OUTDIR 'completed_runs.tsv'

New-Item -ItemType Directory -Force -Path $OUTDIR | Out-Null
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
"num_train`tstatus`ttext_summary`tjson_summary`tlog_file" | Set-Content -Encoding utf8 $indexTsv

$trajList = @($TRAJ_LIST_RAW -split '[,\s]+' | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })

Write-Host "Output directory: $OUTDIR"
Write-Host "Training trajectory counts: $($trajList -join ' ')"

foreach ($numTrain in $trajList) {
    $tag = 'numtrain_{0:D4}' -f [int]$numTrain
    $logFile = Join-Path $logDir "$tag.log"
    $txtFile = Join-Path $OUTDIR "$tag.txt"
    $jsonFile = Join-Path $OUTDIR "$tag.json"

    Write-Host '------------------------------------------------------------'
    Write-Host "Starting run for num_train=$numTrain"

    $argList = @(
        $runnerPy,
        '--system', $SYSTEM,
        '--num-train', "$numTrain",
        '--num-test', "$NUM_TEST",
        '--lifting-order', "$LIFTING_ORDER",
        '--max-iter', "$MAX_ITER",
        '--learn-rate', "$LEARN_RATE",
        '--device', $DEVICE,
        '--train-method', $TRAIN_METHOD,
        '--routine', $ROUTINE,
        '--hp1-scale', "$HP1_SCALE",
        '--opt-weights', "$OPT_W1", "$OPT_W2", "$OPT_W3",
        '--seed-z', "$SEED_Z",
        '--seed-hp', "$SEED_HP",
        '--outdir', $OUTDIR,
        '--tag', $tag
    )

    $status = 'FAILED'
    try {
        & $PYTHON_BIN @argList *> $logFile
        if ($LASTEXITCODE -eq 0) {
            $status = 'OK'
            Write-Host "Finished run for num_train=$numTrain"
        }
        else {
            Write-Host "Run failed for num_train=$numTrain. See $logFile"
        }
    }
    catch {
        $_ | Out-String | Add-Content -Encoding utf8 $logFile
        Write-Host "Run failed for num_train=$numTrain. See $logFile"
    }

    "${numTrain}`t${status}`t${txtFile}`t${jsonFile}`t${logFile}" | Add-Content -Encoding utf8 $indexTsv

    if (($status -ne 'OK') -and ($FAIL_FAST -eq '1')) {
        throw "Stopping because FAIL_FAST=1 and num_train=$numTrain failed."
    }

    Start-Sleep -Seconds ([int]$SLEEP_BETWEEN_RUNS)
}

Write-Host '------------------------------------------------------------'
Write-Host "Sweep complete. Index file: $indexTsv"
