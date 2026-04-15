# basic.ps1 — Windows equivalent of benchmarks/basic.sh
# Usage: cd examples; .\benchmarks\basic.ps1

$SCENE_DIR = "data\360_v2"
$RESULT_DIR = "results\benchmark"
$SCENE_LIST = @("garden", "bicycle", "stump", "bonsai", "counter", "kitchen", "room")
$RENDER_TRAJ_PATH = "ellipse"

foreach ($SCENE in $SCENE_LIST) {
    if ($SCENE -in @("bonsai", "counter", "kitchen", "room")) {
        $DATA_FACTOR = 2
    } else {
        $DATA_FACTOR = 4
    }

    Write-Host "Running $SCENE"

    # train without eval
    python simple_trainer.py default --eval_steps -1 --disable_viewer --data_factor $DATA_FACTOR `
        --render_traj_path $RENDER_TRAJ_PATH `
        --data_dir "$SCENE_DIR\$SCENE" `
        --result_dir "$RESULT_DIR\$SCENE"

    # run eval and render
    $ckpts = Get-ChildItem "$RESULT_DIR\$SCENE\ckpts\*"
    foreach ($CKPT in $ckpts) {
        python simple_trainer.py default --disable_viewer --data_factor $DATA_FACTOR `
            --render_traj_path $RENDER_TRAJ_PATH `
            --data_dir "$SCENE_DIR\$SCENE" `
            --result_dir "$RESULT_DIR\$SCENE" `
            --ckpt $CKPT.FullName
    }
}

# Print results
foreach ($SCENE in $SCENE_LIST) {
    Write-Host "=== Eval Stats ==="
    Get-ChildItem "$RESULT_DIR\$SCENE\stats\val*.json" | ForEach-Object {
        Write-Host $_.FullName
        Get-Content $_.FullName
        Write-Host ""
    }

    Write-Host "=== Train Stats ==="
    Get-ChildItem "$RESULT_DIR\$SCENE\stats\train*_rank0.json" | ForEach-Object {
        Write-Host $_.FullName
        Get-Content $_.FullName
        Write-Host ""
    }
}
