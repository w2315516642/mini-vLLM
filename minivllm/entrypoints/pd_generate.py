"""Send one text request through separately launched P and D engines."""

import argparse
from contextlib import closing

from minivllm.engine.pd_rpc import PDClient
from minivllm.sampling_params import SamplingParams


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefill-control", required=True)
    parser.add_argument("--decode-control", required=True)
    parser.add_argument("--control-authkey", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()
    client = PDClient(
        args.prefill_control,
        args.decode_control,
        args.control_authkey.encode("utf-8"),
    )
    try:
        sampling_params = SamplingParams(
            temperature=0.0, max_tokens=args.max_tokens
        )
        if args.stream:
            with closing(client.generate_stream(
                args.prompt, sampling_params
            )) as stream:
                for output in stream:
                    for completion in output.outputs:
                        print(completion.text, end="", flush=True)
            print()
        else:
            output = client.generate(args.prompt, sampling_params)
            print(output.outputs[0].text)
    finally:
        client.close()


if __name__ == "__main__":
    main()
