"""Linux/AutoDL matrix runner. Owns only servers started by this invocation."""
import argparse
import itertools
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time


def check_port_available(port):
    # Match the HTTP servers' reuse policy: TIME_WAIT is not a live listener.
    # listen() also rejects a competing bound socket that uses SO_REUSEADDR.
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", port))
            probe.listen(1)
        except OSError as exc:
            raise OSError(f"Cannot reserve 127.0.0.1:{port}; check live listeners with "
                          f"ss -ltnp 'sport = :{port}'. No unrelated process was stopped.") from exc


def stop_owned(process):
    # Each server has a private process group; never kill by executable name.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def main():
    import requests
    from benchmarks.benchmark_utils import environment_info

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--b1-dataset", required=True)
    p.add_argument("--b16-dataset", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--frameworks", nargs="+", choices=("mini", "vllm"), default=["mini", "vllm"])
    p.add_argument("--batches", nargs="+", type=int, choices=(1, 16), default=[1, 16])
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--profile", action="store_true")
    args = p.parse_args()
    if args.rounds < 1:
        p.error("rounds must be positive")
    root = Path(__file__).resolve().parents[1]
    datasets = {1: str(Path(args.b1_dataset).resolve()), 16: str(Path(args.b16_dataset).resolve())}
    if not all(Path(path).is_file() for path in datasets.values()):
        p.error("Dataset file missing")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "environment.json").write_text(json.dumps(environment_info(), indent=2))
    cells = list(itertools.product(args.frameworks, args.batches, ("target", "dspark")))
    for round_id in range(1, args.rounds + 1):
        for framework, batch, mode in (cells if round_id % 2 else list(reversed(cells))):
            check_port_available(args.port)
            name = f"{framework}-{mode}-b{batch}-r{round_id}"
            env = dict(os.environ, BENCH_MODE=mode, BATCH_SIZE=str(batch), PORT=str(args.port),
                       MAX_NUM_BATCHED_TOKENS="8192", PROFILE=str(int(args.profile)),
                       VLLM_WORKER_MULTIPROC_METHOD="spawn")
            server = ["bash", "scripts/autodl/" + ("start_vllm.sh" if framework == "vllm" else "start_mini_http.sh")]
            if args.profile:
                server = ["nsys", "profile", "--trace=cuda,nvtx,osrt", "--trace-fork-before-exec=true",
                          "--cuda-graph-trace=node", "--capture-range=cudaProfilerApi",
                          "--capture-range-end=repeat", "-o", str(output / name)] + server
            count = (batch * 2 if args.profile else (100 if batch == 1 else 512))
            client = [sys.executable, "-m", "benchmarks.benchmark_vllm_serving",
                      "--mode", mode, "--framework", framework, "--concurrency", str(batch),
                      "--count", str(count), "--dataset", datasets[batch],
                      "--tokenizer", env.get("TARGET_MODEL", env.get("MODEL_ROOT", "/root/autodl-tmp/models") + "/Qwen3.8-27B-FP8"),
                      "--url", f"http://127.0.0.1:{args.port}", "--output", str(output / (name + ".json"))]
            if args.profile:
                client.append("--profile")
            (output / (name + "-commands.json")).write_text(json.dumps({"server": server, "client": client,
                "settings": {k: env.get(k) for k in ("BENCH_MODE", "BATCH_SIZE", "MAX_NUM_BATCHED_TOKENS",
                "TARGET_MODEL", "DRAFT_MODEL", "CONDA_ENV", "UPSTREAM_ENV", "PROFILE")}}, indent=2))
            print("Starting " + name, flush=True)
            with (output / (name + "-server.log")).open("w") as log:
                process = subprocess.Popen(server, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                try:
                    deadline = time.monotonic() + 1200
                    while True:
                        if process.poll() is not None:
                            raise RuntimeError(f"{name} server exited; inspect its log")
                        try:
                            if requests.get(f"http://127.0.0.1:{args.port}/health", timeout=2).status_code == 200:
                                break
                        except requests.RequestException:
                            pass
                        if time.monotonic() >= deadline:
                            raise TimeoutError(f"{name} startup timed out")
                        time.sleep(2)
                    subprocess.run(client, cwd=root, env=env, check=True)
                finally:
                    stop_owned(process)
            time.sleep(5)
    subprocess.run([sys.executable, "-m", "benchmarks.summarize_framework_matrix", str(output)],
                   cwd=root, check=True)


if __name__ == "__main__":
    main()
