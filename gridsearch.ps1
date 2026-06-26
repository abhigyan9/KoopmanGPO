param(
    [ValidateSet(1, 2)]
    [int]$MaxJobs = 2,

    [string]$PythonExe = "python",
    [string]$SystemName = "Lorenz96_8D",
    [string]$DateStamp = (Get-Date -Format "yyyyMMdd"),
    [int]$LiftingOrder = 100,
    [int]$MaxIter = 200000,
    [string]$Device = "cuda:0",
    [int]$SeedZ = 1234,
    [int]$SeedHP = 1234,
    [int]$FullCostEvalEvery = 50
)

$ErrorActionPreference = "Stop"

# Edit these lists to define the grid.
$LearnRates = @(0.1, 0.02, 0.01, 0.002, 0.001)
$Momentums = @(0.7, 0.75, 0.8)
$StopTols = @(1e-4 , 1e-3)
$TrajBatchSizes = @(16, 20)

$Root = $PSScriptRoot
$OutDir = Join-Path $Root "Figures\GridSearch\${SystemName}_${DateStamp}"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

function Receive-CompletedJobs {
    param(
        [array]$Jobs
    )

    $remaining = @()
    foreach ($job in $Jobs) {
        if ($job.State -eq "Running" -or $job.State -eq "NotStarted") {
            $remaining += $job
            continue
        }

        Write-Host ""
        Write-Host "===== Job $($job.Id): $($job.Name) [$($job.State)] ====="
        $jobOutput = Receive-Job -Job $job 2>&1
        foreach ($line in $jobOutput) {
            Write-Host $line
        }
        Remove-Job -Job $job
    }

    return $remaining
}

$combos = @()
foreach ($lr in $LearnRates) {
    foreach ($mom in $Momentums) {
        foreach ($batch in $TrajBatchSizes) {
            foreach ($tol in $StopTols) {
                $combos += [pscustomobject]@{
                    LearnRate = $lr
                    Momentum = $mom
                    TrajBatchSize = $batch
                    StopTol = $tol
                }
            }
        }
    }
}

Write-Host "Grid search output folder: $OutDir"
Write-Host "Total runs: $($combos.Count)"
Write-Host "Max parallel jobs: $MaxJobs"

$jobs = @()
$runIndex = 0

foreach ($combo in $combos) {
    $jobs = @(Receive-CompletedJobs -Jobs $jobs)

    while (@($jobs | Where-Object { $_.State -eq "Running" -or $_.State -eq "NotStarted" }).Count -ge $MaxJobs) {
        Start-Sleep -Seconds 5
        $jobs = @(Receive-CompletedJobs -Jobs $jobs)
    }

    $runIndex += 1
    $jobName = "grid_${runIndex}_lr-$($combo.LearnRate)_mom-$($combo.Momentum)_tol-$($combo.StopTol)_batch-$($combo.TrajBatchSize)"

    $pyArgs = @(
        "gridsearch_runner.py",
        "--learn-rate", "$($combo.LearnRate)",
        "--momentum", "$($combo.Momentum)",
        "--stop-tol", "$($combo.StopTol)",
        "--traj-batch-size", "$($combo.TrajBatchSize)",
        "--system-name", "$SystemName",
        "--lifting-order", "$LiftingOrder",
        "--max-iter", "$MaxIter",
        "--device", "$Device",
        "--seed-z", "$SeedZ",
        "--seed-hp", "$SeedHP",
        "--full-cost-eval-every", "$FullCostEvalEvery",
        "--datestamp", "$DateStamp",
        "--outdir", "$OutDir"
    )

    Write-Host ""
    Write-Host "Starting run $runIndex/$($combos.Count): lr=$($combo.LearnRate), momentum=$($combo.Momentum), stop_tol=$($combo.StopTol), batch=$($combo.TrajBatchSize)"

    $job = Start-Job -Name $jobName -ScriptBlock {
        param($RootPath, $PythonCommand, $PythonArgs)
        Set-Location $RootPath
        & $PythonCommand @PythonArgs
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Python worker exited with code $LASTEXITCODE"
        }
    } -ArgumentList $Root, $PythonExe, $pyArgs

    $jobs += $job
    Start-Sleep -Seconds 2
}

while ($jobs.Count -gt 0) {
    Start-Sleep -Seconds 5
    $jobs = @(Receive-CompletedJobs -Jobs $jobs)
}

Write-Host ""
Write-Host "All grid jobs finished. Running aggregator..."
& $PythonExe "gridsearch_results.py" --outdir "$OutDir"
if ($LASTEXITCODE -ne 0) {
    throw "Aggregator exited with code $LASTEXITCODE"
}

Write-Host ""
Write-Host "Done. Results saved in: $OutDir"
