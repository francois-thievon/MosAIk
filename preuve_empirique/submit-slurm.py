#!/usr/bin/env python3

import argparse
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


def run_cmd(command: str) -> str:
    return subprocess.check_output(command, shell=True, text=True).strip()


def has_pending_git_changes() -> bool:
    unstaged = run_cmd("git diff --name-only | wc -l")
    staged = run_cmd("git diff --name-only --cached | wc -l")
    return int(unstaged) + int(staged) > 0


def sanitize_sweep_args(raw_args: list[str]) -> list[str]:
    sanitized: list[str] = []
    skip_next = False
    for index, arg in enumerate(raw_args):
        if skip_next:
            skip_next = False
            continue

        if arg in {"--output-dir", "--seed"}:
            skip_next = True
            continue
        if arg.startswith("--output-dir=") or arg.startswith("--seed="):
            continue
        if arg == "--no-show":
            continue

        sanitized.append(arg)

        # Keep short options untouched if present in the future.
        _ = index

    return sanitized


def make_job(
    commit_id: str,
    repo_root: Path,
    script_dir: Path,
    relative_workdir: Path,
    nruns: int,
    partition: str,
    walltime: str,
    base_seed: int,
    sweep_args: list[str],
) -> str:
    sweep_args_shell = shlex.join(sweep_args)
    logs_dir = (script_dir / "logslurms").as_posix()
    output_root = script_dir.as_posix()
    return f"""#!/bin/bash

#SBATCH --job-name=extreme_ratio_sweep_v1
#SBATCH --nodes=1
#SBATCH --partition={partition}
#SBATCH --time={walltime}
#SBATCH --output={logs_dir}/slurm-%A_%a.out
#SBATCH --error={logs_dir}/slurm-%A_%a.err
#SBATCH --array=1-{nruns}

set -euo pipefail

current_dir="{output_root}"
source_dir="{repo_root}"
work_subdir="{relative_workdir.as_posix()}"
base_seed={base_seed}

echo "Session ${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}"
echo "Running on $(hostname)"

echo "Copying source to local scratch"
date
mkdir -p "$TMPDIR/code"
rsync -a --exclude __pycache__ --exclude .pytest_cache --exclude .mypy_cache "$source_dir"/ "$TMPDIR/code"/

echo "Checkout commit {commit_id}"
cd "$TMPDIR/code"
git checkout {commit_id}

echo "Setup virtual environment"
/opt/dce/dce_venv.sh /mounts/datasets/venvs/torch-2.7.1 "$TMPDIR/venv"
source "$TMPDIR/venv/bin/activate"

cd "$TMPDIR/code/$work_subdir"

task_seed=$((base_seed + SLURM_ARRAY_TASK_ID - 1))
run_output_dir="$current_dir/extreme_ratio_sweep_v1_outputs/run_${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}"
mkdir -p "$run_output_dir"

echo "Launching extreme ratio sweep"
python -u extreme_ratio_sweep_v1.py --output-dir "$run_output_dir" --seed "$task_seed" --no-show {sweep_args_shell}

echo "Sync outputs back"
mkdir -p "$current_dir/extreme_ratio_sweep_v1_outputs"
rsync -av "$run_output_dir"/ "$current_dir/extreme_ratio_sweep_v1_outputs/run_${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}"/
"""


def submit_job(job_script: str, sbatch_path: Path) -> None:
    sbatch_path.write_text(job_script, encoding="utf-8")
    if shutil.which("sbatch") is None:
        raise RuntimeError(
            "La commande 'sbatch' est introuvable sur cette machine. "
            f"Le fichier de soumission a ete genere dans {sbatch_path}. "
            "Lance cette commande sur un noeud/login de cluster SLURM, "
            "ou utilise --dry-run pour uniquement generer le script."
        )
    subprocess.run(["sbatch", str(sbatch_path)], check=True)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Submit SLURM jobs for extreme_ratio_sweep_v1.py. "
            "Use '--' then sweep args, for example: "
            "python submit-slurm.py --nruns 4 -- --repeats 20 --epochs 40"
        )
    )
    parser.add_argument("--nruns", type=int, default=1)
    parser.add_argument("--partition", default="gpu_prod_long")
    parser.add_argument("--time", default="12:00:00")
    parser.add_argument("--base-seed", type=int, default=1234)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_known_args()


def main() -> None:
    args, sweep_args_raw = parse_args()
    if args.nruns < 1:
        raise ValueError("--nruns must be >= 1")

    if has_pending_git_changes() and not args.allow_dirty:
        raise RuntimeError(
            "Git worktree has uncommitted changes. Commit/stash first, "
            "or pass --allow-dirty."
        )

    commit_id = run_cmd("git log --pretty=format:'%H' -n 1")
    repo_root = Path(run_cmd("git rev-parse --show-toplevel")).resolve()
    script_dir = Path(__file__).resolve().parent
    relative_workdir = script_dir.relative_to(repo_root)

    sweep_args = sanitize_sweep_args(sweep_args_raw)
    if sweep_args != sweep_args_raw:
        print("Warning: --seed / --output-dir / --no-show are managed by submit-slurm.py and were ignored.")

    print(f"Using commit: {commit_id}")
    print(f"Repository root: {repo_root}")
    print(f"Working subdir in SLURM job: {relative_workdir}")
    print(f"Array jobs: {args.nruns}")

    log_dir = script_dir / "logslurms"
    log_dir.mkdir(parents=True, exist_ok=True)

    sbatch_path = script_dir / "job.sbatch"
    job_script = make_job(
        commit_id=commit_id,
        repo_root=repo_root,
        script_dir=script_dir,
        relative_workdir=relative_workdir,
        nruns=args.nruns,
        partition=args.partition,
        walltime=args.time,
        base_seed=args.base_seed,
        sweep_args=sweep_args,
    )

    if args.dry_run:
        sbatch_path.write_text(job_script, encoding="utf-8")
        print(f"Dry run: wrote {sbatch_path}")
        return

    try:
        submit_job(job_script, sbatch_path)
    except RuntimeError as exc:
        print(str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()