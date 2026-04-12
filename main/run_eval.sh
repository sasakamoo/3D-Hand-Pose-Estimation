#!/bin/bash
#SBATCH --account=rrg-vislearn
#SBATCH --job-name=hoisdf_eval
#SBATCH --output=/scratch/kghasemz/logs/hoisdf_eval_%j.log
#SBATCH --error=/scratch/kghasemz/logs/hoisdf_eval_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=0-01:00
#SBATCH --exclude=ng30706

module load StdEnv/2023 gcc/12.3 cuda/12.2 python/3.10 opencv/4.8.1
source /scratch/kghasemz/envs/hoisdf/bin/activate

REPO=/home/kghasemz/projects/def-vislearn/kghasemz/HOISDF
cd "$REPO" || { echo "ERROR: REPO not found: $REPO"; exit 1; }

export PYTHONPATH="$REPO:$PYTHONPATH"
export MPLBACKEND=Agg
export DISPLAY=
export MPLCONFIGDIR=/tmp
export TORCH_HOME=/scratch/kghasemz/torch_hub

FREIHAND=/home/kghasemz/projects/def-vislearn/kghasemz/dataset
CKPT_DIR=/home/kghasemz/projects/def-vislearn/kghasemz/HOISDF/outputs/model_dump/freihand_run1
SAVE_DIR=/scratch/kghasemz/hoisdf_eval

mkdir -p "$SAVE_DIR"
mkdir -p /scratch/kghasemz/logs

if [ -n "$1" ]; then
    MODEL_PATH="$1"
else
    MODEL_PATH=$(ls -v "$CKPT_DIR"/snapshot_*.pth.tar 2>/dev/null | tail -1)
    if [ -z "$MODEL_PATH" ]; then
        echo "ERROR: No checkpoint found in $CKPT_DIR"
        exit 1
    fi
fi

echo "========================================================"
echo "  HOISDF FreiHAND evaluation"
echo "  Job ID     : $SLURM_JOB_ID"
echo "  Node       : $SLURMD_NODENAME"
echo "  Checkpoint : $MODEL_PATH"
echo "  FreiHAND   : $FREIHAND"
echo "  Output     : $SAVE_DIR"
echo "  Python     : $(which python)"
echo "========================================================"

python -c "import torch,sys; x=torch.zeros(1).cuda(); print(f'CUDA OK — {torch.cuda.get_device_name(0)}')" || exit 1

python main/visualize_predictions.py \
    --model_path   "$MODEL_PATH"  \
    --freihand_dir "$FREIHAND"    \
    --n_samples    32             \
    --save_dir     "$SAVE_DIR"    \
    --split        evaluation     \
    --seed         42             \
    --batch_size   8

echo "Done. Results in: $SAVE_DIR"
