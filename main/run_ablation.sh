#!/bin/bash
# =============================================================================
# run_ablation.sh  —  Single ablation fine-tuning job
# =============================================================================
# Called by submit_ablations.sh — do not run directly.
# Args: $1=ablation_name  $2=run_dir_name  $3=checkpoint_path
# =============================================================================

#SBATCH --account=rrg-vislearn
#SBATCH --job-name=abl_%x
#SBATCH --output=/scratch/kghasemz/logs/abl_%x_%j.log
#SBATCH --error=/scratch/kghasemz/logs/abl_%x_%j.log
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

ABLATION="$1"
RUN_NAME="$2"
RESUME="$3"
FREIHAND=/home/kghasemz/projects/def-vislearn/kghasemz/dataset

mkdir -p /scratch/kghasemz/logs

echo "========================================================"
echo "  Ablation : $ABLATION"
echo "  Run name : $RUN_NAME"
echo "  Resume   : $RESUME"
echo "  Job ID   : $SLURM_JOB_ID"
echo "  Node     : $SLURMD_NODENAME"
echo "========================================================"

python -c "
import torch, sys
if not torch.cuda.is_available(): sys.exit(1)
torch.zeros(1).cuda()
print(f'CUDA OK — {torch.cuda.get_device_name(0)}')
" || { echo "CUDA failed"; exit 1; }

python main/ablation_train.py \
    --ablation     "$ABLATION"    \
    --resume       "$RESUME"      \
    --freihand_dir "$FREIHAND"    \
    --run_dir_name "$RUN_NAME"    \
    --end_epoch    80             \
    --lr           1e-5 \
    --reproj_weight 0.01

echo "========================================================"
echo "  $ABLATION done"
echo "========================================================"