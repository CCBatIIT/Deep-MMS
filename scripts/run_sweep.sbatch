#!/bin/sh
# SLURM array job: train one cell of the profiling sweep per array task.
#
# 1) Generate configs on the login node:
#        python scripts/generate_configs.py
# 2) Build the file list the array indexes into:
#        find json_inputs/profiling_2026 -name '*.json' | sort > sweep_configs.txt
# 3) Submit one task per config (adjust the upper bound to `wc -l sweep_configs.txt`):
#        sbatch --array=1-1950%50 scripts/run_sweep.sbatch sweep_configs.txt
#
# Each task runs scripts/train_dispatch.py, which reads the "architecture" key
# from the JSON and trains the matching model.  Outputs land under runs/
# (git-ignored).
#
#SBATCH --job-name="DMMS_sweep"
#SBATCH --output="logs/sweep.%A_%a.out"
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH -n 5
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jdepaoloboisvert@hawk.iit.edu
#SBATCH --account=cis250004p
#SBATCH -t 24:00:00
#SBATCH --exclude=v020

set -eu

CONFIG_LIST="${1:-sweep_configs.txt}"

source ~/.bashrc
module load anaconda3
module load cuda
conda activate jax_12

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"

# Pick the config for this array task (1-indexed).
CONFIG=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$CONFIG_LIST")
if [ -z "$CONFIG" ]; then
    echo "No config at line ${SLURM_ARRAY_TASK_ID} of ${CONFIG_LIST}" >&2
    exit 1
fi

echo "[$(date)] Task ${SLURM_ARRAY_TASK_ID}: training $CONFIG"
python scripts/train_dispatch.py "$CONFIG"
echo "[$(date)] Task ${SLURM_ARRAY_TASK_ID}: done"

# --- Local (non-SLURM) fallback ---------------------------------------------
# Run the whole sweep serially without a scheduler:
#     for cfg in $(find json_inputs/profiling_2026 -name '*.json' | sort); do
#         python scripts/train_dispatch.py "$cfg"
#     done
