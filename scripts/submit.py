import argparse
import hashlib
import json
import os
from typing import List
import pprint
import time

import yaml
import submitit


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--save_dir", type=str, default=None, required=True)
    parser.add_argument("--prefix", type=str, default=None)
    parser.add_argument("--dry", type=int, default=1)

    # the easiest way is to just specify a config
    parser.add_argument("--compute_config", type=str, default=None)
    parser.add_argument("--program_config", type=str, default=None)

    # sbatch stuff
    parser.add_argument("--account", type=str)
    parser.add_argument("--partition", type=str)
    parser.add_argument("--cpus", type=int, default=None, help="# cpu for the task")
    parser.add_argument("--gpus", type=int, default=None, help="# gpu for the task")
    parser.add_argument("--nodes", type=int, default=None, help="# nodes for the task")
    parser.add_argument("--gpu_type", type=str, default=None, help="a40/titanrtx/etc")
    parser.add_argument("--nodelist", type=str, default=None, help="iliad5/iliad6/etc")
    parser.add_argument("--mem", type=str, default=None, help="32gb")
    parser.add_argument("--exclude", type=str, default="", help="locus-0-37")
    parser.add_argument("--time", type=int, default=None, help="time, in minute")
    parser.add_argument("--task_per_node", type=int, default=1, help="run multi-seed in one job")
    parser.add_argument(
        "--max_pjob", type=int, default=None, help="max # jobs launching in parallel"
    )

    # program related
    parser.add_argument("--main", type=str, default=None)
    # args to override or sweep, e.g. --args x=1;2;3 y=1,2;3,4 z=1
    parser.add_argument("--args", type=str, nargs="+", default=None)

    args = parser.parse_args()
    return args


def process_main_args(main_args: List[str], program_config):
    from_config = {}
    if program_config is not None:
        from_config = yaml.safe_load(open(program_config, "r"))
        print("loaded base config:")
        pprint.pprint(from_config)

    full_args = [from_config]
    if main_args is None:
        return full_args, []

    override_keys = []
    for arg in main_args:
        new_full_args = []
        key, vals = arg.split("=")
        override_keys.append(key)
        vals = vals.split(":")
        for val in vals:
            for args in full_args:
                new_args = args.copy()
                new_args[key] = val
                new_full_args.append(new_args)
        full_args = new_full_args

    return full_args, override_keys


def generate_dict_hash(params_dict, hash_len=7):
    hash_obj = hashlib.sha1(json.dumps(params_dict, sort_keys=True).encode())
    return hash_obj.hexdigest()[:hash_len]


def get_all_commands(args):
    all_main_args, overrides = process_main_args(args.args, args.program_config)
    name2commands = {}
    for main_args in all_main_args:
        cmd = []
        name_entries = []
        if args.program_config is not None:
            name_entries.append(args.program_config.split("/")[-1].rsplit(".", 1)[0])

        for key, val in main_args.items():
            cmd.append(f"--{key}")
            cmd.append(str(val))
            if key in overrides and key not in [
                "config",
                "config_path",
                "eval_against",
                "ppo.resume_from",
            ]:
                if "." in key:
                    keys = key.split(".")
                    if "hidden_dim" in keys:
                        key = "_".join(keys[-2:])
                    else:
                        key = keys[-1]
                name_entries.append(f"{key}{val}")

        job_name = "_".join(name_entries)
        # Cap length of job_name to 128 characters
        job_name = (
            job_name[:128]
            .replace("{", "")
            .replace("}", "")
            .replace("'", "")
            .replace('"', "")
            .replace(":", "")
            .replace("/", "")
        )
        job_name += "_" + generate_dict_hash(main_args)
        if args.prefix is not None:
            job_name = f"{args.prefix}_{job_name}"
        save_dir = os.path.join(args.save_dir, job_name)
        cmd.append("--save_dir")
        cmd.append(save_dir)

        name2commands[job_name] = (cmd, save_dir)
    return name2commands


def submit(args, job_name, command, save_dir, dry):
    from_config = {}
    if args.compute_config is not None:
        from_config = yaml.safe_load(open(args.compute_config, "r"))
        # remove the null values
        from_config = {k: v for k, v in from_config.items() if v is not None}

    for key in [
        "account",
        "partition",
        "cpus",
        "gpus",
        "gpu_type",
        "mem",
        "nodes",
        "time",
        "nodelist",
        "exclude",
    ]:
        if vars(args)[key] is not None:
            from_config[key] = vars(args)[key]

    print(">>> compute config")
    pprint.pprint(from_config)

    executor = submitit.AutoExecutor(folder=os.path.join(os.path.join(save_dir, "submitit")))
    if from_config.get("gpu_type", None) is not None:
        gres = f"gpu:{from_config['gpu_type']}:{from_config['gpus']}"
    elif from_config.get("gpus", None) is not None:
        gres = f"gpu:{from_config['gpus']}"
    else:
        gres = None

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    additional_parameters = {}
    if from_config.get("nodelist", None) is not None:
        additional_parameters["nodelist"] = from_config["nodelist"]

    executor.update_parameters(
        slurm_account=from_config["account"],
        slurm_partition=args.partition,
        cpus_per_task=from_config["cpus"] // args.task_per_node,
        tasks_per_node=args.task_per_node,
        nodes=from_config["nodes"],
        slurm_gres=gres,
        slurm_exclude=from_config["exclude"],
        slurm_mem=from_config["mem"],
        slurm_time=from_config["time"],
        slurm_job_name=job_name,
        slurm_additional_parameters=additional_parameters,
    )

    def launch_training(args, from_config, cmd):
        """This runs *inside* the SLURM allocation."""
        import os
        import subprocess
        import submitit

        if from_config["nodes"] == 1:
            base_cmd = [
                "torchrun",
                "--standalone",
                "--nproc_per_node",
                str(from_config["gpus"]),
                args.main,
            ]
        else:
            env = submitit.JobEnvironment()  # works inside the job
            master_addr = env.hostnames[0]  # first node in the allocation
            os.environ["MASTER_ADDR"] = master_addr
            os.environ["MASTER_PORT"] = "1234"

            base_cmd = [
                "torchrun",
                "--nnodes",
                str(from_config["nodes"]),
                "--nproc_per_node",
                str(from_config["gpus"]),
                "--rdzv-backend",
                "c10d",
                "--rdzv-id",
                env.job_id,  # or any unique string
                "--rdzv-endpoint",
                f"{master_addr}:1234",
                args.main,
            ]
        command = base_cmd + cmd

        print(f">>> job: {job_name}")
        print(f">>> command: {' '.join(command)}")
        subprocess.run(command, check=True)

    # function = submitit.helpers.CommandFunction(command)
    if not args.dry:
        job = executor.submit(launch_training, args, from_config, command)
        return job
    else:
        return None


def wait_if_full(jobs, max_pjob):
    def remove_done_jobs(jobs):
        for i in range(len(jobs) - 1, -1, -1):
            if jobs[i].done():
                jobs.pop(i)
        return len(jobs)

    if max_pjob is None:
        return

    while remove_done_jobs(jobs) >= max_pjob and len(jobs) > 0:
        print(f"reached max job {len(jobs)}, waiting")
        time.sleep(120)

    print(f"{len(jobs)} remaining, time to launch new job!")
    return


if __name__ == "__main__":
    args = parse_args()
    name2commands = get_all_commands(args)

    print(">>> will submit these commands")
    jobs = []
    for name, (cmd, save_dir) in name2commands.items():
        job = submit(args, name, cmd, save_dir, args.dry)
        jobs.append(job)
        wait_if_full(jobs, args.max_pjob)
    print("all job launched!")
