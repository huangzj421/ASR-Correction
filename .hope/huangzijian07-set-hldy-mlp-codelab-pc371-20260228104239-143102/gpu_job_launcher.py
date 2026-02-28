#!/usr/bin/env python3

import os
import sys
import json
import subprocess
from argparse import ArgumentParser, REMAINDER


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--work_dir", default=".", help="work directory")
    parser.add_argument("--launch_mode", default="ddp", help="launch mode.")
    parser.add_argument("--log_dir", default=None, help="base directory to use for log files.")

    # positional arguments
    parser.add_argument(
        "training_script_args",
        nargs=REMAINDER,
        help="The full path to the single GPU training "
        "program/script to be launched in parallel, "
        "followed by all the arguments for the "
        "training script")

    return parser.parse_args()


def run():
    try:
        # resource
        resource_config = os.environ["AFO_RESOURCE_CONFIG"]
        print(f"resource config is {resource_config}")

        resource_dict = json.loads(resource_config)
        nnodes = resource_dict["worker"]["num"]
        nproc_per_node = resource_dict["worker"]["gpu"]

        # topo
        cluster_spec = os.environ["AFO_ENV_CLUSTER_SPEC"]
        print(f"cluster spec is {cluster_spec}")

        spec_dict = json.loads(cluster_spec)

        worker_list = spec_dict["worker"]
        assert len(worker_list) == nnodes
        master_str = worker_list[0]
        master_addr, master_ports  = master_str.split(":")
        master_port = master_ports.split(",")[0]
        print(f"master address is {master_addr}, master port is {master_port}.")

        node_rank = spec_dict["index"]
        print(f"node_rank is {node_rank}.")

        # write hostfile to local dir and set env var
        try:
            hosts = "\n".join([f"{addr} slots={nproc_per_node}" for addr in worker_list])
            print(hosts)

            hostfile = "/workdir/hostfile"
            with open(hostfile, "w") as fptr:
                fptr.write(hosts)
            print(f"write hostfile to {hostfile} which can be visited via environment variable HOSTFILE")
            os.environ["HOSTFILE"] = hostfile
        except Exception as e:
            print(e)
    except:
        # quickrun
        print(f"run in quickrun mode.")
        nnodes = 1
        nproc_per_node = 1
        node_rank = 0
        master_addr = "127.0.0.1"
        master_port = "64200"

    args = parse_args()

    training_script = " ".join(args.training_script_args)

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    os.environ["WORLD_SIZE"] = str(nnodes * nproc_per_node)
    os.environ["LOCAL_WORLD_SIZE"] = str(nproc_per_node)

    # /usr/bin is put on top of PATH, so we may invoke python of wrong version.
    os.environ["PATH"] = f"/usr/local/bin:/usr/local/conda/bin:{os.environ['PATH']}"

    # /usr/local/nvidia/cpu_lib is added to LD_LIBRARY_PATH and causes weird runtime error.
    tgt_str = "/usr/local/nvidia/cpu_lib"
    ld_paths = os.environ["LD_LIBRARY_PATH"].split(":")
    # filter out cpu_lib and update LD_LIBRARY_PATH
    ld_paths = [p for p in ld_paths if tgt_str not in p]
    os.environ["LD_LIBRARY_PATH"] = ":".join(ld_paths)

    # work as distorch.distributed.multiproc
    if args.launch_mode == "multiproc":
        for local_rank in range(0, nproc_per_node):
            rank = node_rank * nproc_per_node + local_rank

            current_env = os.environ.copy()
            current_env["RANK"] = str(rank)
            current_env["LOCAL_RANK"] = str(local_rank)
            current_env["CUDA_VISIBLE_DEVICES"] = str(local_rank)

            redirect = ""
            if args.log_dir:
                log_dir = f"{args.log_dir}/{rank}"
                stdout_file = f"{log_dir}/stdout"
                stderr_file = f"{log_dir}/stderr"
                redirect = f"1>{stdout_file} 2>{stderr_file}"

            cmd = ""
            cmd += f"cd {args.work_dir} || exit 1; "
            if args.log_dir:
                cmd += f"umask 000; mkdir -p -m777 {log_dir} || exit 1; "
            cmd += "[ -f .hope/path.sh ] && echo 'source .hope/path.sh' && . .hope/path.sh; "
            cmd += f"python3 {training_script} {redirect} || exit 1;"

            print("run cmd: \n" + cmd + "\n************", flush=True)

            # spawn the processes
            processes = []
            process = subprocess.Popen(cmd, env=current_env, shell=True)
            processes.append(process)

        for process in processes:
            process.wait()

        exit(0)

    if args.launch_mode == "native":
        launch_cmd = ""
    elif args.launch_mode == "python":
        launch_cmd = f"python3"
    elif args.launch_mode == "ddp":
        launch_cmd = f"python3 -m torch.distributed.run --nnodes={nnodes} --nproc_per_node={nproc_per_node}"
        if nnodes == 1:
            launch_cmd += " --standalone"
        else:
            launch_cmd += f" --node_rank={node_rank} --master_addr={master_addr} --master_port={master_port}"
    else:
        print(f"Unspupported launch mode {args.launch_mode}", file=sys.stderr)
        exit(1)

    redirect = ""
    if args.log_dir:
        log_dir = f"{args.log_dir}/{node_rank}"
        stdout_file = f"{log_dir}/stdout"
        stderr_file = f"{log_dir}/stderr"
        redirect = f"1>{stdout_file} 2>{stderr_file}"

    cmd = ""
    cmd += f"cd {args.work_dir} || exit 1; "
    if args.log_dir:
        cmd += f"umask 000; mkdir -p -m777 {log_dir} || exit 1; "
    cmd += "[ -f .hope/path.sh ] && echo 'source .hope/path.sh' && . .hope/path.sh; "
    cmd += f"{launch_cmd} {training_script} {redirect}"

    print("run cmd: \n" + cmd + "\n************", flush=True)

    ret = subprocess.run(cmd, shell=True)
    exit(ret.returncode)


if __name__ == '__main__':
    run()
