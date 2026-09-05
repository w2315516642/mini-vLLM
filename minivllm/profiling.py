"""Opt-in single-worker Nsight capture and metadata-only NVTX ranges."""

from contextlib import contextmanager
from functools import wraps


_active = False


@contextmanager
def nvtx_range(name):
    if not _active:
        yield
        return
    import torch
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def nvtx_function(name):
    """Read only tensor shapes, never device values; layer ranges nest ops."""
    def decorate(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            if not _active:
                return fn(*args, **kwargs)
            shapes = [tuple(x.shape) for x in (*args, *kwargs.values())
                      if hasattr(x, "shape")][:3]
            layer = getattr(args[0], "layer_idx", None) if args else None
            label = f"{name}:{fn.__name__} shapes={shapes}"
            if name == "linear":
                label += f" weight={tuple(args[0].weight.shape)}"
            if layer is not None:
                label += f" layer={layer}"
            with nvtx_range(label):
                return fn(*args, **kwargs)
        return wrapped
    return decorate


class DecodeCapture:
    def __init__(self, skip, steps):
        if skip < 0 or steps <= 0:
            raise ValueError("Capture skip must be nonnegative and steps positive")
        self.skip, self.steps = skip, steps
        self.seen = 0
        self.records = []
        self.started = self.stopped = False
        self.current = None

    def begin_step(self, metadata):
        global _active
        speculative = set(metadata.speculative_seq_ids)
        # Speculative verification is represented as packed prompts internally.
        # Exclude real prefill, not all prompt-shaped calls.
        if any(s not in speculative for s in metadata.prompt_seq_ids):
            if self.started:
                self.close()
            return
        if self.stopped:
            return
        self.seen += 1
        if self.seen <= self.skip:
            return
        import torch
        if not self.started:
            torch.cuda.synchronize()
            torch.cuda.profiler.start()
            self.started = True
        self.current = {
            "B": len(metadata.prompt_seq_ids) + len(metadata.generation_seq_ids),
            "M": metadata.num_valid_tokens,
            "T": max([1, *metadata.prompt_lens]),
            "decode_step": self.seen,
        }
        torch.cuda.nvtx.range_push(f"decode_step {self.current}")
        _active = True

    def end_step(self, output=None):
        global _active
        if self.current is None:
            return
        import torch
        self.current["produced_tokens"] = sum(
            len(o.output_token_ids) for o in (output or {}).values())
        self.current["failed"] = output is None
        self.records.append(self.current)
        self.current = None
        _active = False
        torch.cuda.nvtx.range_pop()
        if output is None or len(self.records) >= self.steps:
            self.close()

    def close(self):
        global _active
        if self.current is not None:
            self.end_step()
        if self.started and not self.stopped:
            import torch
            # Drain only at the capture boundary, never between operators.
            torch.cuda.synchronize()
            torch.cuda.profiler.stop()
        self.stopped = True
        _active = False
        return {"requested_steps": self.steps, "skip": self.skip,
                "complete": len(self.records) == self.steps,
                "records": self.records}


def capture_worker_step(fn):
    @wraps(fn)
    def wrapped(self, *args, **kwargs):
        capture = getattr(self, "_decode_capture", None)
        if capture is None:
            return fn(self, *args, **kwargs)
        output = None
        try:
            output = fn(self, *args, **kwargs)
            return output
        finally:
            capture.end_step(output)
    return wrapped
