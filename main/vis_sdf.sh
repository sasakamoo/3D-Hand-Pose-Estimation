#!/bin/bash
# =============================================================================
# run_overfit.sh  —  Hand-only HOISDF overfit sanity test on FreiHAND
# =============================================================================
# Submit from HOISDF-main/:
#   sbatch main/run_overfit.sh
#
# Outputs land in /scratch/kghasemz/hoisdf_overfit/
# Log:              /scratch/kghasemz/logs/hoisdf_overfit_<JOBID>.log
# =============================================================================

#SBATCH --account=rrg-vislearn
#SBATCH --job-name=hoisdf_overfit
#SBATCH --output=/scratch/kghasemz/logs/hoisdf_overfit_%j.log
#SBATCH --error=/scratch/kghasemz/logs/hoisdf_overfit_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=0-00:30        # 30 minutes is plenty for 400 iters

# ── environment ───────────────────────────────────────────────────────────────
module load StdEnv/2023 gcc/12.3 cuda/12.2 python/3.10 opencv/4.8.1
source /scratch/kghasemz/envs/hoisdf/bin/activate
 
cd "$REPO" || { echo "ERROR: REPO not found: $REPO"; exit 1; }
 
export PYTHONPATH="$REPO:$PYTHONPATH"
export MPLBACKEND=Agg
export DISPLAY=
export MPLCONFIGDIR=/tmp
export TORCH_HOME=/scratch/kghasemz/torch_hub
 
# ── paths ─────────────────────────────────────────────────────────────────────
FREIHAND=/home/kghasemz/projects/def-vislearn/kghasemz/dataset
SAVE_DIR=/scratch/kghasemz/hoisdf_overfit
 
mkdir -p "$SAVE_DIR"
mkdir -p /scratch/kghasemz/logs
 
echo "========================================================"
echo "  HOISDF overfit test"
echo "  Job ID  : $SLURM_JOB_ID"
echo "  Node    : $SLURMD_NODENAME"
echo "  Repo    : $REPO"
echo "  FreiHAND: $FREIHAND"
echo "  Output  : $SAVE_DIR"
echo "  Python  : $(which python)"
echo "  CUDA    : $(python -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null || echo 'checking...')"
echo "========================================================"
 
# Verify CUDA is actually working before spending time on the job
python -c "
import torch, sys
if not torch.cuda.is_available():
    print('ERROR: CUDA not available'); sys.exit(1)
x = torch.zeros(1).cuda()
print(f'CUDA OK — {torch.cuda.get_device_name(0)}')
" || { echo "CUDA health check failed — try resubmitting on a different node"; exit 1; }
 
# Note: --gpu is NOT passed here. SLURM sets CUDA_VISIBLE_DEVICES automatically
# via --gres=gpu:1, so the only available device is always index 0 inside the job.
# python main/overfit_test.py \
#     --freihand_dir "$FREIHAND" \
#     --n_samples    4           \
#     --iters        400         \
#     --lr           1e-4        \
#     --device       cuda        \
#     --save_dir     "$SAVE_DIR"
python main/visualize_sdf_points.py \
    --freihand_dir /home/kghasemz/projects/def-vislearn/kghasemz/dataset \
    --model_path outputs/model_dump/freihand_run1/snapshot_69_1331.pth.tar \
    --use_model_sdf --n_samples 8

EXIT_CODE=$?
echo "========================================================"
echo "  Overfit test finished (exit code: $EXIT_CODE)"
echo "  Results in: $SAVE_DIR"
echo "========================================================"
exit $EXIT_CODE
 