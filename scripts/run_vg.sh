GPU_IDX=1
SDF_SUBDATADIR=SDF
SDF_CHECKPOINT=ckpt_020000.pth
SUBDATADIR=VG
VG_CHECKPOINT=ckpt_008000.pth

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

for SCAN in "47984" "44234" "354371"
do
    CONF="./confs/vg.conf"
    run_with_timing "vg-train $SCAN" python run_vg.py --conf $CONF --mode train --sdf_subdatadir $SDF_SUBDATADIR --sdf_checkpoint_name $SDF_CHECKPOINT \
    --datadir $DATA_DIR --expdir $EXP_DIR --dataname $SCAN --subdatadir $SUBDATADIR --gpu $GPU_IDX
    run_with_timing "vg-validate $SCAN" python run_vg.py --conf $CONF --mode validate_mesh_delaunay --sdf_subdatadir $SDF_SUBDATADIR --sdf_checkpoint_name $SDF_CHECKPOINT \
    --datadir $DATA_DIR --expdir $EXP_DIR --dataname $SCAN --subdatadir $SUBDATADIR --gpu $GPU_IDX --checkpoint_name $VG_CHECKPOINT
done
