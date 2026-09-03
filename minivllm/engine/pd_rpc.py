"""Trusted-cluster RPC control plane for independently launched P/D engines."""

from __future__ import annotations

import threading
import time
import uuid
from contextlib import closing
from dataclasses import dataclass
from multiprocessing.connection import Client, Connection, Listener
from typing import Any, Iterator, Optional, Tuple

from minivllm.engine.output_processor import OutputProcessor
from minivllm.engine.pd_runtime import PDEngineBridge
from minivllm.multimodal import MultiModalInputs
from minivllm.outputs import RequestOutput, RequestOutputKind
from minivllm.sampling_params import SamplingParams


_RPC_METHODS = {
    "abort_request",
    "abort_decode_handoff",
    "activate_decode_handoff",
    "add_request",
    "execute_cache_transfers",
    "get_num_unfinished_requests",
    "get_transfer_layouts",
    "has_unfinished_requests",
    "pop_prefill_handoffs",
    "prepare_decode_handoff",
    "release_prefill_handoff",
    "step",
}


def parse_control_address(value: str) -> Tuple[str, int]:
    try:
        host, port_text = value.rsplit(":", 1)
        port = int(port_text)
    except (AttributeError, ValueError) as exc:
        raise ValueError(
            f"control address must use host:port form, got {value!r}"
        ) from exc
    if not host or not 0 < port < 65536:
        raise ValueError(f"invalid control address: {value!r}")
    return host, port


class RemoteEngineError(RuntimeError):
    pass


class PDControlServer:
    """Expose only the engine operations required by the PD coordinator."""

    def __init__(
        self,
        engine,
        address: str,
        authkey: bytes,
    ) -> None:
        if not authkey:
            raise ValueError("control-plane authkey must not be empty")
        self.engine = engine
        self.listener = Listener(
            parse_control_address(address), authkey=authkey
        )
        self._closed = False
        self._engine_lock = threading.RLock()

    def serve_forever(self) -> None:
        while not self._closed:
            try:
                connection = self.listener.accept()
            except (OSError, EOFError):
                return
            thread = threading.Thread(
                target=self._serve_connection,
                args=(connection,),
                daemon=True,
            )
            thread.start()

    def _serve_connection(self, connection: Connection) -> None:
        with connection:
            while not self._closed:
                try:
                    request = connection.recv()
                except EOFError:
                    return
                try:
                    method_name = request["method"]
                    if method_name not in _RPC_METHODS:
                        raise ValueError(
                            f"RPC method {method_name!r} is not allowed"
                        )
                    method = getattr(self.engine, method_name)
                    with self._engine_lock:
                        result = method(
                            *request.get("args", ()),
                            **request.get("kwargs", {}),
                        )
                    connection.send({"ok": True, "result": result})
                except Exception as exc:
                    connection.send(
                        {
                            "ok": False,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )

    def close(self) -> None:
        self._closed = True
        self.listener.close()


class RemoteEngineClient:
    """Duck-typed engine proxy consumed directly by ``PDEngineBridge``."""

    def __init__(self, address: str, authkey: bytes) -> None:
        if not authkey:
            raise ValueError("control-plane authkey must not be empty")
        self.connection = Client(
            parse_control_address(address), authkey=authkey
        )
        self._lock = threading.RLock()

    def call(self, method: str, *args, **kwargs):
        if method not in _RPC_METHODS:
            raise ValueError(f"RPC method {method!r} is not allowed")
        with self._lock:
            self.connection.send(
                {"method": method, "args": args, "kwargs": kwargs}
            )
            response = self.connection.recv()
        if not response["ok"]:
            raise RemoteEngineError(
                f"{response['error_type']}: {response['error']}"
            )
        return response["result"]

    def __getattr__(self, name: str):
        if name not in _RPC_METHODS:
            raise AttributeError(name)
        return lambda *args, **kwargs: self.call(name, *args, **kwargs)

    def close(self) -> None:
        self.connection.close()


class PDClient:
    """Single-owner synchronous client for a dedicated pair of P/D engines.

    Do not attach concurrent PDClients to the same engines: step() returns all
    scheduled requests, so concurrent serving needs a central output dispatcher.
    """

    def __init__(
        self,
        prefill_control: str,
        decode_control: str,
        authkey: bytes,
    ) -> None:
        self.prefill = RemoteEngineClient(prefill_control, authkey)
        self.decode = RemoteEngineClient(decode_control, authkey)
        self.bridge = PDEngineBridge(self.prefill, self.decode)
        self._generating = False

    def generate(
        self,
        prompt: Optional[str],
        sampling_params: SamplingParams,
        prompt_token_ids: Optional[list[int]] = None,
        multi_modal_inputs: Optional[MultiModalInputs] = None,
        request_id: Optional[str] = None,
    ) -> RequestOutput:
        output, _ = self.generate_with_metrics(
            prompt,
            sampling_params,
            prompt_token_ids,
            multi_modal_inputs,
            request_id,
        )
        return output

    def generate_with_metrics(
        self,
        prompt: Optional[str],
        sampling_params: SamplingParams,
        prompt_token_ids: Optional[list[int]] = None,
        multi_modal_inputs: Optional[MultiModalInputs] = None,
        request_id: Optional[str] = None,
    ) -> Tuple[RequestOutput, "PDGenerationMetrics"]:
        with closing(self._generate(
            prompt, sampling_params, prompt_token_ids, multi_modal_inputs,
            request_id,
        )) as stream:
            for output, metrics in stream:
                if metrics is not None:
                    return output, metrics
        raise RuntimeError("PD generation ended without a final output")

    def generate_stream(
        self,
        prompt: Optional[str],
        sampling_params: SamplingParams,
        prompt_token_ids: Optional[list[int]] = None,
        multi_modal_inputs: Optional[MultiModalInputs] = None,
        request_id: Optional[str] = None,
        *,
        output_kind: RequestOutputKind = RequestOutputKind.DELTA,
    ) -> Iterator[RequestOutput]:
        """Yield P's first token and then D's updates, with shared offsets.

        Close this generator when abandoning a request. The RPC connection
        remains reusable; only this request's cache/state is released.
        """
        processor = OutputProcessor(sampling_params, output_kind)
        with closing(self._generate(
            prompt, sampling_params, prompt_token_ids, multi_modal_inputs,
            request_id,
        )) as stream:
            for output, _ in stream:
                update = processor.process(output)
                if update is not None:
                    yield update

    def _generate(
        self, prompt, sampling_params, prompt_token_ids, multi_modal_inputs,
        request_id,
    ) -> Iterator[Tuple[RequestOutput, Optional["PDGenerationMetrics"]]]:
        """One PD execution loop shared by final, measured and streamed APIs."""
        if self._generating:
            raise RuntimeError("Finish or close the active PD generation first")
        request_id = request_id or uuid.uuid4().hex
        started_at = time.perf_counter()
        self._generating = True
        transferred = False
        finished = False
        first_token_at = None
        try:
            self.prefill.add_request(
                request_id, prompt, sampling_params, prompt_token_ids,
                multi_modal_inputs=multi_modal_inputs,
            )
            while True:
                for output in self.prefill.step():
                    if output.request_id != request_id:
                        continue
                    if output.outputs and output.outputs[0].token_ids:
                        first_token_at = first_token_at or time.perf_counter()
                    if output.is_finished():
                        finished = True
                        finished_at = time.perf_counter()
                        yield output, PDGenerationMetrics(
                            prefill_s=finished_at - started_at,
                            transfer_s=0.0,
                            decode_s=0.0,
                            ttft_s=(first_token_at or finished_at) - started_at,
                            tpot_s=0.0,
                        )
                        return
                    # P already sampled the first token. Emit it before the
                    # transfer; D's cumulative snapshot will include it again.
                    yield output, None
                selected = [
                    handoff for handoff in self.prefill.pop_prefill_handoffs()
                    if handoff.request_id == request_id
                ]
                if selected:
                    if len(selected) != 1:
                        raise RuntimeError("P produced duplicate request handoffs")
                    transfer_started_at = time.perf_counter()
                    self.bridge.transfer(selected[0])
                    transfer_finished_at = time.perf_counter()
                    transferred = True
                    break

            decode_started_at = time.perf_counter()
            while self.decode.has_unfinished_requests():
                for output in self.decode.step():
                    if output.request_id != request_id:
                        continue
                    if not output.is_finished():
                        yield output, None
                        continue
                    self.bridge.coordinator.finish(request_id)
                    finished = True
                    finished_at = time.perf_counter()
                    num_decode_intervals = max(
                        len(output.outputs[0].token_ids) - 1, 1
                    )
                    yield output, PDGenerationMetrics(
                        prefill_s=transfer_started_at - started_at,
                        transfer_s=transfer_finished_at - transfer_started_at,
                        decode_s=finished_at - decode_started_at,
                        ttft_s=(
                            (first_token_at or transfer_started_at) - started_at
                        ),
                        tpot_s=(
                            (finished_at - decode_started_at) / num_decode_intervals
                        ),
                    )
                    return
            raise RuntimeError("D released the request without a final output")
        finally:
            self._generating = False
            if not finished:
                if transferred:
                    self.bridge.coordinator.cancel(request_id)
                    self.decode.abort_request(request_id)
                else:
                    self.prefill.abort_request(request_id)

    def close(self) -> None:
        self.prefill.close()
        self.decode.close()


@dataclass(frozen=True)
class PDGenerationMetrics:
    prefill_s: float
    transfer_s: float
    decode_s: float
    ttft_s: float
    tpot_s: float
