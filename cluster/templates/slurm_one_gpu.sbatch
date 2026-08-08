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
# The launcher preserves every completed per-example checkpoint after SIGTERM.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

export VLM_BACKEND="${VLM_BACKEND:-openai}"
export FAAR_PYTHON="${FAAR_PYTHON:-$PWD/.venv/bin/python}"

"${FAAR_PYTHON}" cluster/launcher.py \
  --project-root "${SLURM_SUBMIT_DIR:-$PWD}" \
  --gate off \
  --recovery off \
  --split test \
  --out "results/b0_slurm_${SLURM_JOB_ID}.json"
