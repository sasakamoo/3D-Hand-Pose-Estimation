#!/bin/bash
# =============================================================================
# run_train.sh  —  Hand-only HOISDF full training on FreiHAND
# Submit:  sbatch main/run_train.sh
# Resume:  sbatch main/run_train.sh --continue
# =============================================================================

#SBATCH --account=rrg-vislearn
#SBATCH --job-name=hoisdf_freihand
#SBATCH --output=/scratch/kghasemz/logs/hoisdf_train_%j.log
#SBATCH --error=/scratch/kghasemz/logs/hoisdf_train_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=1-00:00
#SBATCH --exclude=ng30706        # known broken GPU node

# ── environment ───────────────────────────────────────────────────────────────
module load StdEnv/2023 gcc/12.3 cuda/12.2 python/3.10 opencv/4.8.1
source /scratch/kghasemz/envs/hoisdf/bin/activate

REPO=/lustre06/project/6085198/kghasemz/HOISDF
cd "$REPO" || { echo "ERROR: REPO not found: $REPO"; exit 1; }

export PYTHONPATH="$REPO:$PYTHONPATH"
export MPLBACKEND=Agg
export DISPLAY=
export MPLCONFIGDIR=/tmp
export TORCH_HOME=/scratch/kghasemz/torch_hub

# ── paths ─────────────────────────────────────────────────────────────────────
FREIHAND=/home/kghasemz/projects/def-vislearn/kghasemz/dataset
OUT_DIR=/scratch/kghasemz/hoisdf_freihand

mkdir -p "$OUT_DIR"
mkdir -p /scratch/kghasemz/logs

# config.py writes outputs relative to CWD; symlink outputs → scratch
# so checkpoints land on scratch instead of the project filesystem
if [ ! -L "$REPO/outputs" ]; then
    mkdir -p "$OUT_DIR/outputs"
    ln -s "$OUT_DIR/outputs" "$REPO/outputs"
fi

# Pass --continue through if provided when submitting
EXTRA_ARGS="$@"

echo "========================================================"
echo "  HOISDF FreiHAND training"
echo "  Job ID  : $SLURM_JOB_ID"
echo "  Node    : $SLURMD_NODENAME"
echo "  Repo    : $REPO"
echo "  FreiHAND: $FREIHAND"
echo "  Output  : $OUT_DIR"
echo "  Resume  : ${EXTRA_ARGS:-no}"
echo "  Python  : $(which python)"
echo "========================================================"

# CUDA health check
python -c "
import torch, sys
if not torch.cuda.is_available():
    print('ERROR: CUDA not available'); sys.exit(1)
x = torch.zeros(1).cuda()
print(f'CUDA OK — {torch.cuda.get_device_name(0)}')
" || { echo "CUDA health check failed"; exit 1; }

python main/train.py \
    --freihand_dir         "$FREIHAND"   \
    --run_dir_name         freihand_run1 \
    --end_epoch            70            \
    --point_sampling_epoch 40            \
    --lr_drop              9             \
    $EXTRA_ARGS

EXIT_CODE=$?
echo "========================================================"
echo "  Training done (exit: $EXIT_CODE)"
echo "  Checkpoints: $OUT_DIR/outputs/model_dump/freihand_run1/"
echo "========================================================"
exit $EXIT_CODE