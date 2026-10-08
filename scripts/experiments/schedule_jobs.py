from __future__ import annotations

import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pydantic.v1 as pydantic
import pydantic_argparse

from docdjinn.logging import get_logger

logger = get_logger(__name__)


def read_jobs_from_bash(config_file):
    """Parse CONFIGS array from a bash file and return as a Python list."""
    cmd = f'bash -c \'source {shlex.quote(config_file)} && printf "%s\\n" "${{CONFIGS[@]}}"\''
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    print(lines)
    return lines


def get_free_gpu(gpu_slots):
    """Return the first available GPU id, or None if all busy."""
    for gpu_id, slot in gpu_slots.items():
        if slot is None:
            return gpu_id
    return None


def get_running_jobs_info(gpu_slots):
    """Return formatted string with current running jobs."""
    running = []
    for gpu_id, slot in gpu_slots.items():
        if slot is not None:
            job_name, process, start_time, log_file = slot
            elapsed = datetime.now() - start_time
            elapsed_str = str(elapsed).split(".")[0]
            running.append(f"GPU{gpu_id}: {log_file.name} ({elapsed_str})\n")
    return running


def main(config: ScheduleJobs):
    gpu_ids = [int(gid) for gid in config.gpu_ids.split(",")]
    config_path = Path(config.config_file)

    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    # Read job definitions
    jobs = read_jobs_from_bash(str(config_path))
    total_jobs = len(jobs)
    if total_jobs == 0:
        print(f"No jobs found in {config_path}")
        sys.exit(1)

    # Prepare logs directory
    log_dir = Path("data/runs/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting job scheduler with {total_jobs} jobs across GPUs {gpu_ids}")

    gpu_slots = {i: None for i in gpu_ids}
    completed_jobs = 0

    while jobs or any(gpu_slots.values()):
        # Check for finished jobs
        for gpu_id, slot in list(gpu_slots.items()):
            if slot is not None:
                job_name, process, start_time, log_file = slot
                if process.poll() is not None:
                    completed_jobs += 1
                    elapsed = datetime.now() - start_time
                    elapsed_str = str(elapsed).split(".")[0]
                    return_code = process.returncode
                    status = (
                        "SUCCESS"
                        if return_code == 0
                        else f"FAILED (code: {return_code})"
                    )
                    print(
                        f"[GPU {gpu_id}] {status}: {log_file.name} (duration: {elapsed_str})"
                    )
                    log_file.close()
                    gpu_slots[gpu_id] = None

        # Launch new jobs if GPUs are free
        while jobs and (free_gpu := get_free_gpu(gpu_slots)) is not None:
            job = jobs.pop(0)
            parts = job.split()
            if len(parts) < 4:
                print(f"⚠️  Skipping malformed job line: {job}")
                continue

            name, script, dataset, model, *extra_args = parts
            log_path = log_dir / f"{name}.log"
            log_file = open(log_path, "w")

            extra = " ".join(extra_args)
            cmd = f"CUDA_VISIBLE_DEVICES={free_gpu} bash {script} {dataset} {model} {extra}".strip()
            start_time = datetime.now()

            print(
                f"[GPU {free_gpu}] 🏃 Starting job: {name} "
                f"({completed_jobs + len([s for s in gpu_slots.values() if s])}/{total_jobs}) "
                f"→ log: {log_path}"
            )
            print(cmd)

            process = subprocess.Popen(
                cmd, shell=True, stdout=log_file, stderr=subprocess.STDOUT
            )
            gpu_slots[free_gpu] = (name, process, start_time, log_file)

        # Periodic status updates
        running_jobs = get_running_jobs_info(gpu_slots)
        if running_jobs:
            print(
                f"Status: {completed_jobs}/{total_jobs} completed, "
                f"{len(jobs)} queued | Running:\n{', '.join(running_jobs)}"
            )

        time.sleep(5)

    print("All jobs completed!")


class ScheduleJobs(pydantic.BaseModel):
    gpu_ids: str
    config_file: str  # Path to bash config file with job definitions


if __name__ == "__main__":
    parser = pydantic_argparse.ArgumentParser(
        model=ScheduleJobs,
    )
    main(parser.parse_typed_args())
