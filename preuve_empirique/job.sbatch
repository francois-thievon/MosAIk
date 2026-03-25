#!/bin/bash

#SBATCH --job-name=extreme_ratio_sweep_v1
#SBATCH --nodes=1
#SBATCH --partition=gpu_prod_long
#SBATCH --time=12:00:00
#SBATCH --output=/Users/theobeaumont/Desktop/MosAIk/preuve_empirique/logslurms/slurm-%A_%a.out
#SBATCH --error=/Users/theobeaumont/Desktop/MosAIk/preuve_empirique/logslurms/slurm-%A_%a.err
#SBATCH --array=1-1

set -euo pipefail

current_dir="/Users/theobeaumont/Desktop/MosAIk/preuve_empirique"
source_dir="/Users/theobeaumont/Desktop/MosAIk"
work_subdir="preuve_empirique"
base_seed=1234

echo "Session ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Running on $(hostname)"

echo "Copying source to local scratch"
date
mkdir -p "$TMPDIR/code"
rsync -a --exclude __pycache__ --exclude .pytest_cache --exclude .mypy_cache "$source_dir"/ "$TMPDIR/code"/

echo "Checkout commit 601f3fc51563a838e3a7cb61316b688a25f55048"
cd "$TMPDIR/code"
git checkout 601f3fc51563a838e3a7cb61316b688a25f55048

echo "Setup virtual environment"
/opt/dce/dce_venv.sh /mounts/datasets/venvs/torch-2.7.1 "$TMPDIR/venv"
source "$TMPDIR/venv/bin/activate"

cd "$TMPDIR/code/$work_subdir"

task_seed=$((base_seed + SLURM_ARRAY_TASK_ID - 1))
run_output_dir="$current_dir/extreme_ratio_sweep_v1_outputs/run_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$run_output_dir"

echo "Launching extreme ratio sweep"
python -u extreme_ratio_sweep_v1.py --output-dir "$run_output_dir" --seed "$task_seed" --no-show 

echo "Sync outputs back"
mkdir -p "$current_dir/extreme_ratio_sweep_v1_outputs"
rsync -av "$run_output_dir"/ "$current_dir/extreme_ratio_sweep_v1_outputs/run_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"/
