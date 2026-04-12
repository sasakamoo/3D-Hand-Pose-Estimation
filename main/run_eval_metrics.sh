#!/bin/bash
# =============================================================================
# run_eval_metrics.sh  —  Evaluate HOISDF checkpoints with full metrics
# =============================================================================
# Computes MPJPE, PA-MPJPE, PCK-3D@50mm, AUC-3D on the FreiHAND eval set.
#
# Usage:
#   # Evaluate a single checkpoint
#   sbatch main/run_eval_metrics.sh /path/to/best.pth.tar
#
#   # Evaluate all ablation best checkpoints at once
#   sbatch main/run_eval_metrics.sh --all
#
#   # Evaluate the original trained model
#   sbatch main/run_eval_metrics.sh --original
# =============================================================================

#SBATCH --account=rrg-vislearn
#SBATCH --job-name=hoisdf_eval_metrics
#SBATCH --output=/scratch/kghasemz/logs/eval_metrics_%j.log
#SBATCH --error=/scratch/kghasemz/logs/eval_metrics_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=0-02:00
#SBATCH --exclude=ng30706,ng11105,ng11106,ng30708

module load StdEnv/2023 gcc/12.3 cuda/12.2 python/3.10 opencv/4.8.1
source /scratch/kghasemz/envs/hoisdf/bin/activate

REPO=/home/kghasemz/projects/def-vislearn/kghasemz/HOISDF
cd "$REPO" || { echo "ERROR: REPO not found"; exit 1; }

export PYTHONPATH="$REPO:$PYTHONPATH"
export MPLBACKEND=Agg
export DISPLAY=
export MPLCONFIGDIR=/tmp
export TORCH_HOME=/scratch/kghasemz/torch_hub

FREIHAND=/home/kghasemz/projects/def-vislearn/kghasemz/dataset
SAVE_DIR=/scratch/kghasemz/hoisdf_eval_metrics
ORIG_CKPT=$REPO/outputs/model_dump/freihand_run1/snapshot_69_1331.pth.tar
ABL_ROOT=$REPO/outputs/model_dump

mkdir -p "$SAVE_DIR"
mkdir -p /scratch/kghasemz/logs

echo "========================================================"
echo "  HOISDF Evaluation — Full Metrics"
echo "  Job ID  : $SLURM_JOB_ID"
echo "  Node    : $SLURMD_NODENAME"
echo "  Save to : $SAVE_DIR"
echo "========================================================"

python -c "
import torch, sys
if not torch.cuda.is_available(): sys.exit(1)
torch.zeros(1).cuda()
print(f'CUDA OK — {torch.cuda.get_device_name(0)}')
" || { echo "CUDA failed"; exit 1; }

# ── Determine which checkpoints to evaluate ───────────────────────────────────

if [ "$1" == "--all" ]; then
    # Evaluate all ablation best checkpoints
    echo "Mode: evaluate all ablations"

    MODEL_PATHS=""
    NAMES=""

    for abl in baseline bone_loss reproj_loss lr_sched sdf_pts; do
        # Try abl4_ first (latest round), then abl3_, abl2_, abl_
        for prefix in abl4 abl3 abl2 abl; do
            ckpt="$ABL_ROOT/${prefix}_${abl}/best.pth.tar"
            if [ -f "$ckpt" ]; then
                MODEL_PATHS="$MODEL_PATHS $ckpt"
                NAMES="$NAMES ${prefix}_${abl}"
                echo "  Found: $ckpt"
                break
            fi
        done
    done

    if [ -z "$MODEL_PATHS" ]; then
        echo "ERROR: No ablation checkpoints found in $ABL_ROOT"
        exit 1
    fi

    python main/evaluate.py \
        --model_path $MODEL_PATHS \
        --names      $NAMES \
        --freihand_dir "$FREIHAND" \
        --save_dir     "$SAVE_DIR" \
        --batch_size   22

elif [ "$1" == "--original" ]; then
    # Evaluate only the original trained model
    echo "Mode: evaluate original model"
    echo "  Checkpoint: $ORIG_CKPT"

    python main/evaluate.py \
        --model_path "$ORIG_CKPT" \
        --names      original_ep69 \
        --freihand_dir "$FREIHAND" \
        --save_dir     "$SAVE_DIR" \
        --batch_size   22

elif [ "$1" == "--all_with_original" ]; then
    # Evaluate original + all ablations together for comparison
    echo "Mode: evaluate original + all ablations"

    MODEL_PATHS="$ORIG_CKPT"
    NAMES="original"

    for abl in baseline bone_loss reproj_loss lr_sched sdf_pts; do
        for prefix in abl4 abl3 abl2 abl; do
            ckpt="$ABL_ROOT/${prefix}_${abl}/best.pth.tar"
            if [ -f "$ckpt" ]; then
                MODEL_PATHS="$MODEL_PATHS $ckpt"
                NAMES="$NAMES $abl"
                echo "  Found: $ckpt"
                break
            fi
        done
    done

    python main/evaluate.py \
        --model_path $MODEL_PATHS \
        --names      $NAMES \
        --freihand_dir "$FREIHAND" \
        --save_dir     "$SAVE_DIR" \
        --batch_size   22

elif [ -n "$1" ]; then
    # Evaluate a single provided checkpoint
    echo "Mode: evaluate single checkpoint"
    echo "  Checkpoint: $1"

    # Use provided name as second arg, or derive from path
    NAME="${2:-$(basename $(dirname $1))}"

    python main/evaluate.py \
        --model_path "$1" \
        --names      "$NAME" \
        --freihand_dir "$FREIHAND" \
        --save_dir     "$SAVE_DIR" \
        --batch_size   22

else
    # Default: evaluate original model
    echo "No argument given — evaluating original model"
    echo "  Usage:"
    echo "    sbatch run_eval_metrics.sh                     # original model"
    echo "    sbatch run_eval_metrics.sh --original          # original model"
    echo "    sbatch run_eval_metrics.sh --all               # all ablations"
    echo "    sbatch run_eval_metrics.sh --all_with_original # original + all ablations"
    echo "    sbatch run_eval_metrics.sh /path/to/ckpt.pth.tar [name]"

    python main/evaluate.py \
        --model_path "$ORIG_CKPT" \
        --names      original_ep69 \
        --freihand_dir "$FREIHAND" \
        --save_dir     "$SAVE_DIR" \
        --batch_size   22
fi

EXIT_CODE=$?
echo "========================================================"
echo "  Evaluation done (exit: $EXIT_CODE)"
echo "  Results in: $SAVE_DIR"
echo "========================================================"
exit $EXIT_CODE