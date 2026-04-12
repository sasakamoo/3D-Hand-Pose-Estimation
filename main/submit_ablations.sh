#!/bin/bash
# =============================================================================
# submit_ablations.sh  —  Launch all 5 ablation jobs in parallel
# =============================================================================
# Usage (from HOISDF repo root or anywhere):
#   bash main/submit_ablations.sh
#
# Optionally pass a checkpoint path:
#   bash main/submit_ablations.sh /path/to/snapshot_69_1331.pth.tar
#
# All 5 jobs run simultaneously, each using one A100 GPU for ~12h.
# Results land in:
#   $REPO/outputs/model_dump/abl_<name>/
#   /scratch/kghasemz/logs/abl_<name>_<jobid>.log
# =============================================================================

REPO=/home/kghasemz/projects/def-vislearn/kghasemz/HOISDF
CKPT_DIR=$REPO/outputs/model_dump/freihand_run1

# Use provided checkpoint or auto-find latest
if [ -n "$1" ]; then
    RESUME="$1"
else
    RESUME=$(ls -v "$CKPT_DIR"/snapshot_*.pth.tar 2>/dev/null | tail -1)
    if [ -z "$RESUME" ]; then
        echo "ERROR: No checkpoint found in $CKPT_DIR"
        echo "Usage: bash main/submit_ablations.sh /path/to/checkpoint.pth.tar"
        exit 1
    fi
fi

echo "=================================================="
echo "  Submitting 5 ablation jobs"
echo "  Base checkpoint: $RESUME"
echo "=================================================="

# Submit each ablation as a separate job
# --job-name sets %x in the output filename

submit_one() {
    local ABLATION=$1
    local RUN_NAME="abl_${ABLATION}"
    sbatch \
        --job-name="$ABLATION" \
        "$REPO/main/run_ablation.sh" \
        "$ABLATION" "$RUN_NAME" "$RESUME"
}

submit_one baseline
submit_one bone_loss
submit_one reproj_loss
submit_one lr_sched
submit_one sdf_pts

echo ""
echo "All 5 jobs submitted. Monitor with:"
echo "  squeue -u kghasemz"
echo ""
echo "Watch a specific job:"
echo "  tail -f /scratch/kghasemz/logs/abl_<name>_<jobid>.log"
echo ""
echo "Results will be in:"
echo "  $REPO/outputs/model_dump/abl_<name>/"