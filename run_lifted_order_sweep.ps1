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
$SYSTEM = Get-Setting 'SYSTEM' 'unforced_poc'
$timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$OUTDIR = Get-Setting 'OUTDIR' (Join-Path 'Figures' "lifted_order_sweep_$timestamp")
$NUM_TRAIN = Get-Setting 'NUM_TRAIN' '100'
$NUM_TEST = Get-Setting 'NUM_TEST' '40'
$MAX_ITER = Get-Setting 'MAX_ITER' '100000'
$LEARN_RATE = Get-Setting 'LEARN_RATE' '0.002'
$DEVICE = Get-Setting 'DEVICE' 'cuda:0'
$TRAIN_METHOD = Get-Setting 'TRAIN_METHOD' 'Zero-Mean'
$ROUTINE = Get-Setting 'ROUTINE' 'standard'
$HP1_SCALE = Get-Setting 'HP1_SCALE' '1.0'
$OPT_W1 = Get-Setting 'OPT_W1' '1.0'
$OPT_W2 = Get-Setting 'OPT_W2' '1.0'
$OPT_W3 = Get-Setting 'OPT_W3' '0.0'
$SEED_Z = Get-Setting 'SEED_Z' '1234'
$SEED_HP = Get-Setting 'SEED_HP' '1234'
$SLEEP_BETWEEN_RUNS = Get-Setting 'SLEEP_BETWEEN_RUNS' '10'
$LIFTED_ORDER_LIST_RAW = Get-Setting 'LIFTED_ORDER_LIST' '5 10 15 20 25 30 35 40 45 50'
$FAIL_FAST = Get-Setting 'FAIL_FAST' '0'

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$runnerPy = Join-Path $scriptDir 'run_igpk_lifted_order.py'
$logDir = Join-Path $OUTDIR 'logs'
$indexTsv = Join-Path $OUTDIR 'completed_runs.tsv'

New-Item -ItemType Directory -Force -Path $OUTDIR | Out-Null
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
"lifted_order`tnum_train`tstatus`ttext_summary`tjson_summary`tlog_file" | Set-Content -Encoding utf8 $indexTsv

$liftedOrderList = @($LIFTED_ORDER_LIST_RAW -split '[,\s]+' | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })

Write-Host "Output directory: $OUTDIR"
Write-Host "Number of training trajectories: $NUM_TRAIN"
Write-Host "Lifted orders: $($liftedOrderList -join ' ')"

foreach ($liftedOrder in $liftedOrderList) {
    $tag = 'lifted_order_{0:D4}' -f [int]$liftedOrder
    $logFile = Join-Path $logDir "$tag.log"
    $txtFile = Join-Path $OUTDIR "$tag.txt"
    $jsonFile = Join-Path $OUTDIR "$tag.json"

    Write-Host '------------------------------------------------------------'
    Write-Host "Starting run for lifted_order=$liftedOrder, num_train=$NUM_TRAIN"

    $argList = @(
        $runnerPy,
        '--system', $SYSTEM,
        '--num-train', "$NUM_TRAIN",
        '--num-test', "$NUM_TEST",
        '--lifting-order', "$liftedOrder",
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
            Write-Host "Finished run for lifted_order=$liftedOrder"
        }
        else {
            Write-Host "Run failed for lifted_order=$liftedOrder. See $logFile"
        }
    }
    catch {
        $_ | Out-String | Add-Content -Encoding utf8 $logFile
        Write-Host "Run failed for lifted_order=$liftedOrder. See $logFile"
    }

    "${liftedOrder}`t${NUM_TRAIN}`t${status}`t${txtFile}`t${jsonFile}`t${logFile}" | Add-Content -Encoding utf8 $indexTsv

    if (($status -ne 'OK') -and ($FAIL_FAST -eq '1')) {
        throw "Stopping because FAIL_FAST=1 and lifted_order=$liftedOrder failed."
    }

    Start-Sleep -Seconds ([int]$SLEEP_BETWEEN_RUNS)
}

Write-Host '------------------------------------------------------------'
Write-Host "Sweep complete. Index file: $indexTsv"
