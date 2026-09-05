"""Opt-in worker timeline; no per-stage synchronization or tensor inspection."""

import time


class StageProfile:
    def __init__(self, max_steps):
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self.max_steps = max_steps
        self.steps = []
        self.current = None

    def begin(self):
        if len(self.steps) >= self.max_steps:
            return
        self.current = {"stages": [], "counts": {}}
        self.steps.append(self.current)
        self._boundary()

    def _boundary(self):
        import torch
        self.event = torch.cuda.Event(enable_timing=True)
        self.event.record()
        self.started = time.perf_counter()

    def mark(self, name):
        if self.current is None:
            return
        previous, started = self.event, self.started
        self._boundary()
        self.current["stages"].append(
            (name, previous, self.event, (self.started - started) * 1000))

    def counts(self, **values):
        if self.current is not None:
            self.current["counts"].update(values)

    def end(self):
        self.current = None

    def finish(self):
        import torch
        # Only drain once, outside benchmark request timing. Events measure
        # stream elapsed time (including idle gaps), NOT summed kernel time.
        torch.cuda.synchronize()
        records = []
        for step in self.steps:
            records.append({"counts": step["counts"], "stages": {
                name: {"stream_ms": start.elapsed_time(end), "host_ms": host}
                for name, start, end, host in step["stages"]}})
        return {"max_steps": self.max_steps, "steps": records,
                "limit_reached": len(records) >= self.max_steps}
