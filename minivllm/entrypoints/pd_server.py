"""Launch one prefill-only or decode-only mini-vLLM process."""

import argparse

from minivllm.configs import PDRole
from minivllm.engine.arg_utils import EngineArgs
from minivllm.engine.llm_engine import LLMEngine
from minivllm.engine.pd_rpc import PDControlServer


def main() -> None:
    parser = argparse.ArgumentParser()
    EngineArgs.add_cli_args(parser)
    parser.add_argument(
        "--control-address",
        required=True,
        help="trusted control-plane host:port",
    )
    parser.add_argument(
        "--control-authkey",
        required=True,
        help="shared control-plane secret",
    )
    args = parser.parse_args()
    engine_args = EngineArgs.from_cli_args(args)
    if engine_args.pd_role == PDRole.UNIFIED.value:
        parser.error("pd_server requires --pd-role prefill or decode")
    engine = LLMEngine.from_engine_args(engine_args)
    server = PDControlServer(
        engine,
        args.control_address,
        args.control_authkey.encode("utf-8"),
    )
    try:
        server.serve_forever()
    finally:
        server.close()
        engine._run_workers("close_transfer_engine")


if __name__ == "__main__":
    main()
