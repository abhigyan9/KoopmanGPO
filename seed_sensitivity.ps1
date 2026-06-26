param(
    [ValidateSet(1, 2)]
    [int]$MaxJobs = 2,

    [string]$PythonExe = "python",
    [string]$SystemName = "Cart_data",
    [string]$DateStamp = (Get-Date -Format "yyyyMMdd"),
    [double]$TrainFrac = 0.60,
    [int]$LiftingOrder = 35,
    [int]$MaxIter = 100000,
    [string]$Device = "cuda:0",
    [double]$LearnRate = 0.002,
    [double]$Momentum = 0.75,
    [double]$StopTol = 1e-3,
    [string]$Routine = "standard",
    [string]$TrajBatchSize = "15",
    [int]$FullCostEvalEvery = 50
)

$ErrorActionPreference = "Stop"

# Edit these lists to define the seed grid.
$ZSeeds = @(1, 10, 46, 91, 137, 182, 227, 273, 318, 363, 409, 454, 500, 1234)
$HPSeeds = @(1, 10, 46, 91, 137, 182, 227, 273, 318, 363, 409, 454, 500, 1234)

$Root = $PSScriptRoot
$OutDir = Join-Path $Root "Figures\SeedSensitivity\${SystemName}_${DateStamp}"
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
foreach ($zSeed in $ZSeeds) {
    foreach ($hpSeed in $HPSeeds) {
        $combos += [pscustomobject]@{
            ZSeed = $zSeed
            HPSeed = $hpSeed
        }
    }
}

Write-Host "Seed sensitivity output folder: $OutDir"
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
    $jobName = "seed_${runIndex}_zs-$($combo.ZSeed)_hps-$($combo.HPSeed)"

    $pyArgs = @(
        "seed_sensitivity_runner.py",
        "--seed-z", "$($combo.ZSeed)",
        "--seed-hp", "$($combo.HPSeed)",
        "--system-name", "$SystemName",
        "--train-frac", "$TrainFrac",
        "--lifting-order", "$LiftingOrder",
        "--max-iter", "$MaxIter",
        "--device", "$Device",
        "--learn-rate", "$LearnRate",
        "--momentum", "$Momentum",
        "--stop-tol", "$StopTol",
        "--routine", "$Routine",
        "--traj-batch-size", "$TrajBatchSize",
        "--full-cost-eval-every", "$FullCostEvalEvery",
        "--datestamp", "$DateStamp",
        "--outdir", "$OutDir"
    )

    Write-Host ""
    Write-Host "Starting run $runIndex/$($combos.Count): z_seed=$($combo.ZSeed), hp_seed=$($combo.HPSeed)"

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
Write-Host "All seed sensitivity jobs finished. Running aggregator..."
& $PythonExe "seed_sensitivity_results.py" --outdir "$OutDir"
if ($LASTEXITCODE -ne 0) {
    throw "Aggregator exited with code $LASTEXITCODE"
}

Write-Host ""
Write-Host "Done. Results saved in: $OutDir"
