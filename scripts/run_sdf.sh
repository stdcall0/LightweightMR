GPU_IDX=0
SUBDARADIR=SDF
SDF_CHECKPOINT=ckpt_020000.pth
OPT_PERF=${OPT_PERF:-on}

DATA_DIR=./example/data/
EXP_DIR=./example/exp/

run_with_timing() {
    local desc="$1"; shift
    local start=$(date +%s)
    local tmp_log=$(mktemp)
    /usr/bin/env time -f "%e %P %M" "$@" 2> "$tmp_log"
    exit_code=$?
    local end=$(date +%s)
    read elapsed cpu rss_kb < "$tmp_log"
    rm -f "$tmp_log"
    # Convert KB -> MiB for readability.
    rss_mib=$(python - <<PY
import math
try:
    v = float("${rss_kb:-0}")
    print(f"{v/1024:.2f}")
except Exception:
    print("0.00")
PY
)
    echo "[time] ${elapsed} s elapsed | CPU ${cpu} | MaxRSS ${rss_mib} MiB (${rss_kb} KB)"
    echo "[wall] ${desc} took $((end-start)) s (exit $exit_code)"
    return $exit_code
}

for SCAN in "47984" "44234" "354371"; do
    CONF=./confs/sdf.conf
    run_with_timing "sdf-train $SCAN" python run_sdf.py --conf $CONF --mode train --subdatadir $SUBDARADIR --datadir $DATA_DIR --expdir $EXP_DIR --dataname $SCAN --gpu $GPU_IDX --opt_perf $OPT_PERF
    run_with_timing "sdf-validate $SCAN" python run_sdf.py --conf $CONF --mode validate_mesh --subdatadir $SUBDARADIR --datadir $DATA_DIR --expdir $EXP_DIR --dataname $SCAN --gpu $GPU_IDX --checkpoint_name $SDF_CHECKPOINT --opt_perf $OPT_PERF
done