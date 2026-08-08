#!/usr/bin/env bash
#SBATCH --job-name=faar-one-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --time=08:00:00
#SBATCH --signal=TERM@120

# Edit the resource directives for the confirmed lab allocation.
# B0--B2 checkpoint each example. Allocate enough walltime for each
# uncheckpointed B3/B4 visual stage.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

export VLM_BACKEND="${VLM_BACKEND:-openai}"
export FAAR_PYTHON="${FAAR_PYTHON:-$PWD/.venv-aaai/bin/python}"
export FAAR_DATASET="${FAAR_DATASET:-ohrbench}"
export FAAR_SPLIT="${FAAR_SPLIT:-test}"

"${FAAR_PYTHON}" cluster/run_baselines.py \
  --project-root "${SLURM_SUBMIT_DIR:-$PWD}" \
  --dataset "${FAAR_DATASET}" \
  --split "${FAAR_SPLIT}" \
  --resume
