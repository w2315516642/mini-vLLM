"""Expose a dedicated unified/P/D engine to the benchmark's single driver."""

import argparse
import json
from pathlib import Path

from benchmarks.benchmark_utils import environment_info


def main():
    from minivllm.engine.arg_utils import EngineArgs
    from minivllm.engine.llm_engine import LLMEngine
    from minivllm.engine.pd_rpc import PDControlServer

    parser = argparse.ArgumentParser(description=__doc__)
    EngineArgs.add_cli_args(parser)
    parser.add_argument("--control-address", required=True)
    parser.add_argument("--control-authkey", required=True)
    parser.add_argument("--info-file", required=True)
    parser.set_defaults(disable_log_stats=True)
    args = parser.parse_args()
    engine = LLMEngine.from_engine_args(EngineArgs.from_cli_args(args))
    server = PDControlServer(engine, args.control_address, args.control_authkey.encode())
    info = {"endpoint": args.control_address, "config": engine.get_runtime_stats()["config"],
            "environment": environment_info()}
    path = Path(args.info_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(f"Benchmark server ready: {path}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.close()
        engine._run_workers("close_transfer_engine")


if __name__ == "__main__":
    main()
