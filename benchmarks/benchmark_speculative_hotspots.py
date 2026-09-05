"""Compare Markov argmax and verification LM Head on identical inputs."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from benchmarks.benchmark_utils import environment_info


def measure(fn, repeats):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeats


def peak(fn):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    result = fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - before) / 2**20


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--operator', choices=['all', 'markov', 'lmhead'], default='all')
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 16])
    parser.add_argument('--vocab-size', type=int, default=248320)
    parser.add_argument('--rank', type=int, default=128)
    parser.add_argument('--hidden-size', type=int, default=5120)
    parser.add_argument('--width', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--output', default='build/benchmarks/speculative-hotspots.json')
    args = parser.parse_args()
    if min(*args.batch_sizes, args.vocab_size, args.rank, args.hidden_size,
           args.width, args.repeats) <= 0:
        parser.error('Dimensions and repeats must be positive')
    torch.manual_seed(42)
    records = []
    with torch.inference_mode():
        for b in args.batch_sizes:
            for operator in ('markov', 'lmhead'):
                if args.operator not in ('all', operator):
                    continue
                if operator == 'markov':
                    from minivllm import dspark_ops
                    from minivllm.spec_decode.dspark_cuda import markov_argmax
                    base = torch.randn(b, args.vocab_size, device='cuda')
                    x = torch.randn(b, args.rank, device='cuda', dtype=torch.bfloat16)
                    weight = torch.randn(args.vocab_size, args.rank, device='cuda', dtype=x.dtype)
                    methods = {
                        'scalar_cuda': lambda: dspark_ops.markov_argmax(base, x, weight),
                        'gemm_argmax': lambda: (base + torch.mm(x, weight.T,
                                                out_dtype=torch.float32)).argmax(-1),
                        'tiled': lambda: markov_argmax(base, x, weight),
                    }
                    expected = methods['scalar_cuda']()
                    for fn in methods.values():
                        torch.testing.assert_close(fn(), expected)
                else:
                    x = torch.randn(b, args.width + 1, args.hidden_size,
                                    device='cuda', dtype=torch.bfloat16).mul_(.02)
                    weight = torch.randn(args.vocab_size, args.hidden_size,
                                         device='cuda', dtype=x.dtype).mul_(.02)
                    methods = {
                        'per_request': lambda: [F.linear(row, weight).float() for row in x],
                        'batched': lambda: F.linear(x.flatten(0, 1), weight).float(),
                    }
                    expected = torch.cat(methods['per_request']())
                    actual = methods['batched']()
                    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=1e-2)
                    del actual
                del expected
                # Alternate order; warmup/JIT and correctness are not timed.
                timings = {name: [] for name in methods}
                for names in (list(methods), list(reversed(methods))):
                    for name in names:
                        timings[name].append(measure(methods[name], args.repeats))
                record = {'operator': operator, 'batch_size': b,
                          'ms': {n: sum(t)/len(t) for n,t in timings.items()},
                          'extra_peak_mib': {n: peak(fn) for n,fn in methods.items()}}
                records.append(record)
                print(json.dumps(record), flush=True)
                del methods, x, weight
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'config': vars(args), 'environment': environment_info(),
                               'records': records}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
