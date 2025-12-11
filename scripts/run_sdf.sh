GPU_IDX=0
SUBDARADIR=SDF
SDF_CHECKPOINT=ckpt_020000.pth

DATA_DIR=./example/data/
EXP_DIR=./example/exp/

run_with_timing() {
    local desc="$1"; shift
    local start=$(date +%s)
    /usr/bin/env time -f "[time] %e s elapsed | CPU %P | MaxRSS %M KB" "$@"
    local status=$?
    local end=$(date +%s)
    echo "[wall] ${desc} took $((end-start)) s (exit $status)"
    return $status
}

for SCAN in "47984" "44234" "354371"; do
    CONF=./confs/sdf.conf
    run_with_timing "sdf-train $SCAN" python run_sdf.py --conf $CONF --mode train --subdatadir $SUBDARADIR --datadir $DATA_DIR --expdir $EXP_DIR --dataname $SCAN --gpu $GPU_IDX
    run_with_timing "sdf-validate $SCAN" python run_sdf.py --conf $CONF --mode validate_mesh --subdatadir $SUBDARADIR --datadir $DATA_DIR --expdir $EXP_DIR --dataname $SCAN --gpu $GPU_IDX --checkpoint_name $SDF_CHECKPOINT
done