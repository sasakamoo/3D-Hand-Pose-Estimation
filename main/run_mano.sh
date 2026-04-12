#!/bin/bash
# =============================================================================
# run_mano.sh  —  MANO mesh visualisation + optional fine-tuning with MANO loss
# =============================================================================
# Usage:
#   # Visualise MANO predictions from existing checkpoint
#   sbatch main/run_mano.sh --vis /path/to/checkpoint.pth.tar
#
#   # Fine-tune existing checkpoint WITH MANO supervision enabled
#   sbatch main/run_mano.sh --train /path/to/checkpoint.pth.tar
#
#   # Both: fine-tune then visualise
#   sbatch main/run_mano.sh --train_vis /path/to/checkpoint.pth.tar
# =============================================================================

#SBATCH --account=rrg-vislearn
#SBATCH --job-name=hoisdf_mano
#SBATCH --output=/scratch/kghasemz/logs/mano_%j.log
#SBATCH --error=/scratch/kghasemz/logs/mano_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=0-12:00
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
ORIG_CKPT=$REPO/outputs/model_dump/freihand_run1/snapshot_69_1331.pth.tar
VIS_DIR=/scratch/kghasemz/hoisdf_mano_vis
MANO_CKPT_DIR=$REPO/outputs/model_dump/mano_finetune

mkdir -p "$VIS_DIR"
mkdir -p "$MANO_CKPT_DIR"
mkdir -p /scratch/kghasemz/logs

MODE="$1"
CKPT="${2:-$ORIG_CKPT}"

echo "========================================================"
echo "  HOISDF MANO run"
echo "  Mode       : $MODE"
echo "  Checkpoint : $CKPT"
echo "  Job ID     : $SLURM_JOB_ID"
echo "  Node       : $SLURMD_NODENAME"
echo "========================================================"

python -c "
import torch, sys
if not torch.cuda.is_available(): sys.exit(1)
torch.zeros(1).cuda()
print(f'CUDA OK — {torch.cuda.get_device_name(0)}')
" || { echo "CUDA failed"; exit 1; }

# ── check training_mano.json exists ───────────────────────────────────────────
if [ ! -f "$FREIHAND/training_mano.json" ]; then
    echo "WARNING: $FREIHAND/training_mano.json not found!"
    echo "  MANO supervision will be disabled (zero params used)."
    echo "  Download from: https://lmb.informatik.uni-freiburg.de/resources/datasets/FreihandDataset.en.html"
fi

# ── Mode dispatch ─────────────────────────────────────────────────────────────

if [ "$MODE" == "--vis" ]; then
    echo "Mode: Visualise MANO predictions"
    # python main/visualize_mano.py \
    #     --model_path   "$CKPT"    \
    #     --freihand_dir "$FREIHAND" \
    #     --n_samples    32          \
    #     --save_dir     "$VIS_DIR"  \
    #     --split        evaluation  \
    #     --seed         42          
        # --batch_size   8
    python main/slide_visualize.py \
        --model_path  "$CKPT" \
        --freihand_dir "$FREIHAND" \
        --save_dir    /scratch/kghasemz/slides \
        --stride      1 \
        --gpu         0
elif [ "$MODE" == "--train" ]; then
    echo "Mode: Fine-tune with MANO supervision"
    python main/ablation_train.py \
        --ablation     baseline   \
        --resume       "$CKPT"    \
        --freihand_dir "$FREIHAND" \
        --run_dir_name mano_finetune \
        --end_epoch    100

elif [ "$MODE" == "--train_vis" ]; then
    echo "Mode: Fine-tune then visualise"

    python main/ablation_train.py \
        --ablation     baseline   \
        --resume       "$CKPT"    \
        --freihand_dir "$FREIHAND" \
        --run_dir_name mano_finetune \
        --end_epoch    100

    BEST_CKPT=$REPO/outputs/model_dump/mano_finetune/best.pth.tar
    if [ -f "$BEST_CKPT" ]; then
        python main/visualize_mano.py \
            --model_path   "$BEST_CKPT" \
            --freihand_dir "$FREIHAND"  \
            --n_samples    32            \
            --save_dir     "$VIS_DIR"   \
            --split        evaluation   \
            --seed         42           \
            --batch_size   8
    else
        echo "ERROR: best.pth.tar not found after training"
        exit 1
    fi

else
    echo "Unknown mode: $MODE"
    echo "Usage:"
    echo "  sbatch run_mano.sh --vis     [checkpoint]   # visualise only"
    echo "  sbatch run_mano.sh --train   [checkpoint]   # fine-tune with MANO"
    echo "  sbatch run_mano.sh --train_vis [checkpoint] # fine-tune + visualise"
    exit 1
fi

echo "========================================================"
echo "  Done. Results in: $VIS_DIR"
echo "========================================================"