import subprocess
import os
import math
from datetime import datetime


def get_slurm_remaining_time_minutes(job_id=None):
    if job_id is None:
        job_id = os.environ.get("SLURM_JOB_ID")

    if not job_id:
        return None

    try:
        result = subprocess.run(
            ["scontrol", "show", "job", job_id], capture_output=True, text=True, check=True
        )

        time_limit = None
        start_time = None

        for line in result.stdout.split("\n"):
            if "TimeLimit=" in line:
                time_str = line.split("TimeLimit=")[1].split()[0]
                if time_str == "UNLIMITED":
                    print("Job has unlimited time limit.")
                    return None

                time_limit = parse_time_to_minutes(time_str)

            if "StartTime=" in line:
                start_time_str = line.split("StartTime=")[1].split()[0]
                start_time = datetime.strptime(start_time_str, "%Y-%m-%dT%H:%M:%S")

        if time_limit is None or start_time is None:
            print(f"Could not find necessary information for job {job_id}")
            return None

        elapsed_time = (datetime.now() - start_time).total_seconds() / 60
        remaining_time = max(0, time_limit - math.floor(elapsed_time))

        return int(remaining_time)

    except subprocess.CalledProcessError as e:
        print(f"Error running scontrol: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error getting SLURM remaining time: {e}")
        return None


def parse_time_to_minutes(time_str):
    if "-" in time_str:
        days, time_part = time_str.split("-")
    else:
        days, time_part = "0", time_str

    if ":" in time_part:
        if time_part.count(":") == 2:
            hours, mins, secs = time_part.split(":")
        else:
            hours, mins = time_part.split(":")
            secs = "0"
    else:
        hours, mins, secs = "0", time_part, "0"

    return int(days) * 1440 + int(hours) * 60 + int(mins) + math.floor(int(secs) / 60)
