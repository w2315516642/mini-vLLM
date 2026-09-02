"""Project-owned TCP data plane inspired by Mooncake's segment transfers."""

from __future__ import annotations

import json
import socket
import struct
from threading import RLock, Thread
from typing import Any, Dict, Mapping, Tuple

import torch

from minivllm.distributed.kv_transfer.backend import TransferBackend
from minivllm.distributed.kv_transfer.types import (
    RegisteredBuffer,
    TransferEndpoint,
    TransferHandle,
    TransferPlan,
    TransferStatus,
)


_LENGTH_STRUCT = struct.Struct("!Q")
_MAX_HEADER_BYTES = 8 << 20


def _split_host_port(hostname: str) -> Tuple[str, int]:
    try:
        host, port_text = hostname.rsplit(":", 1)
        port = int(port_text)
    except (ValueError, AttributeError) as exc:
        raise ValueError(
            f"hostname must use host:port form, got {hostname!r}"
        ) from exc
    if not host or not 0 < port < 65536:
        raise ValueError(f"invalid transfer endpoint: {hostname!r}")
    return host, port


def _recv_exact(connection: socket.socket, length: int) -> bytes:
    chunks = []
    remaining = length
    while remaining:
        chunk = connection.recv(min(remaining, 1 << 20))
        if not chunk:
            raise ConnectionError(
                f"peer closed with {remaining} transfer bytes remaining"
            )
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _send_message(connection: socket.socket, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, separators=(",", ":")).encode("utf-8")
    connection.sendall(_LENGTH_STRUCT.pack(len(payload)))
    connection.sendall(payload)


def _recv_message(connection: socket.socket) -> Dict[str, Any]:
    length = _LENGTH_STRUCT.unpack(_recv_exact(connection, _LENGTH_STRUCT.size))[0]
    if length <= 0 or length > _MAX_HEADER_BYTES:
        raise ValueError(f"invalid transfer header length: {length}")
    value = json.loads(_recv_exact(connection, length).decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("transfer header must be a JSON object")
    return value


class P2PTransferBackend(TransferBackend):
    """Push registered tensor slices over TCP with no third-party runtime."""

    def __init__(
        self,
        endpoint: TransferEndpoint,
        timeout_s: float = 30.0,
        start_server: bool = True,
    ) -> None:
        super().__init__(endpoint)
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self.timeout_s = timeout_s
        self._registered: Dict[str, torch.Tensor] = {}
        self._descriptors: Dict[str, RegisteredBuffer] = {}
        self._handles: Dict[str, TransferHandle] = {}
        self._lock = RLock()
        self._closed = False
        self._server = None
        self._server_thread = None
        if start_server:
            self._start_server()

    def _start_server(self) -> None:
        host, port = _split_host_port(self.endpoint.hostname)
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, port))
        server.listen()
        server.settimeout(0.2)
        self._server = server
        self._server_thread = Thread(
            target=self._serve,
            name=f"pd-transfer-{self.endpoint.endpoint_id}",
            daemon=True,
        )
        self._server_thread.start()

    def _serve(self) -> None:
        while not self._closed:
            try:
                connection, _ = self._server.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            Thread(
                target=self._receive_transfer,
                args=(connection,),
                daemon=True,
            ).start()

    def _receive_transfer(self, connection: socket.socket) -> None:
        with connection:
            connection.settimeout(self.timeout_s)
            try:
                header = _recv_message(connection)
                slices = header.get("slices")
                if not isinstance(slices, list) or not slices:
                    raise ValueError("transfer requires at least one slice")
                validated = []
                with self._lock:
                    for item in slices:
                        name = str(item["target_name"])
                        try:
                            tensor = self._registered[name]
                            descriptor = self._descriptors[name]
                        except KeyError as exc:
                            raise ValueError(
                                f"target buffer {name!r} is not registered"
                            ) from exc
                        offset = int(item["target_offset"])
                        length = int(item["length"])
                        if int(item["target_address"]) != descriptor.address:
                            raise ValueError(
                                f"target buffer {name!r} address changed"
                            )
                        if item["dtype"] != descriptor.dtype:
                            raise ValueError(
                                f"target buffer {name!r} dtype changed"
                            )
                        if tuple(item["shape"]) != descriptor.shape:
                            raise ValueError(
                                f"target buffer {name!r} shape changed"
                            )
                        if offset < 0 or length <= 0:
                            raise ValueError("slice ranges must be positive")
                        if offset + length > descriptor.nbytes:
                            raise ValueError(
                                f"slice exceeds target buffer {name!r}"
                            )
                        validated.append((tensor, offset, length))

                    # The lock pins descriptors while bytes are copied into them.
                    for tensor, offset, length in validated:
                        payload = bytearray(_recv_exact(connection, length))
                        source = torch.frombuffer(payload, dtype=torch.uint8)
                        target = tensor.view(torch.uint8).reshape(-1).narrow(
                            0, offset, length
                        )
                        target.copy_(source, non_blocking=False)
                _send_message(
                    connection,
                    {
                        "status": "completed",
                        "transfer_id": str(header["transfer_id"]),
                    },
                )
            except Exception as exc:
                try:
                    _send_message(
                        connection,
                        {"status": "failed", "error": str(exc)},
                    )
                except Exception:
                    pass

    def register_tensor(
        self,
        name: str,
        tensor: Any,
    ) -> RegisteredBuffer:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("P2P registration requires a torch tensor")
        if not tensor.is_contiguous():
            raise ValueError("registered tensors must be contiguous")
        if tensor.numel() == 0:
            raise ValueError("registered tensors must not be empty")
        with self._lock:
            if self._closed:
                raise RuntimeError("transfer backend is closed")
            if name in self._registered:
                raise ValueError(f"buffer {name!r} is already registered")
            descriptor = RegisteredBuffer(
                endpoint=self.endpoint,
                name=name,
                address=tensor.data_ptr(),
                nbytes=tensor.numel() * tensor.element_size(),
                dtype=str(tensor.dtype),
                shape=tuple(tensor.shape),
                device=str(tensor.device),
            )
            self._registered[name] = tensor
            self._descriptors[name] = descriptor
            return descriptor

    def unregister_buffer(self, name: str) -> None:
        with self._lock:
            if name not in self._registered:
                raise ValueError(f"buffer {name!r} is not registered")
            if any(
                not handle.status.is_terminal
                for handle in self._handles.values()
            ):
                raise RuntimeError(
                    "cannot unregister memory during an active transfer"
                )
            del self._registered[name]
            del self._descriptors[name]

    def submit(self, plan: TransferPlan) -> TransferHandle:
        if plan.source_endpoint != self.endpoint:
            raise ValueError("this backend does not own the plan source")
        with self._lock:
            if self._closed:
                raise RuntimeError("transfer backend is closed")
            if plan.transfer_id in self._handles:
                raise ValueError(f"duplicate transfer id: {plan.transfer_id}")
            for item in plan.slices:
                local = self._descriptors.get(item.source.buffer.name)
                if local != item.source.buffer:
                    raise ValueError(
                        f"source buffer {item.source.buffer.name!r} is not "
                        "registered by this backend"
                    )
            handle = TransferHandle(plan.transfer_id)
            handle.transition(TransferStatus.RUNNING)
            self._handles[plan.transfer_id] = handle
        Thread(
            target=self._send_transfer,
            args=(plan, handle),
            name=f"pd-send-{plan.transfer_id}",
            daemon=True,
        ).start()
        return handle

    def _send_transfer(
        self,
        plan: TransferPlan,
        handle: TransferHandle,
    ) -> None:
        try:
            host, port = _split_host_port(plan.target_endpoint.hostname)
            with socket.create_connection(
                (host, port), timeout=self.timeout_s
            ) as connection:
                connection.settimeout(self.timeout_s)
                header = {
                    "transfer_id": plan.transfer_id,
                    "request_id": plan.request_id,
                    "slices": [
                        {
                            "target_name": item.target.buffer.name,
                            "target_address": item.target.buffer.address,
                            "target_offset": item.target.offset,
                            "length": item.length,
                            "dtype": item.target.buffer.dtype,
                            "shape": list(item.target.buffer.shape),
                        }
                        for item in plan.slices
                    ],
                }
                _send_message(connection, header)
                with self._lock:
                    for item in plan.slices:
                        tensor = self._registered[item.source.buffer.name]
                        source = tensor.view(torch.uint8).reshape(-1).narrow(
                            0, item.source.offset, item.length
                        )
                        if source.device.type != "cpu":
                            source = source.cpu()
                        connection.sendall(source.numpy().tobytes())
                response = _recv_message(connection)
                if response.get("status") != "completed":
                    raise RuntimeError(
                        response.get("error", "remote transfer failed")
                    )
                if response.get("transfer_id") != plan.transfer_id:
                    raise RuntimeError("remote acknowledged another transfer")
            handle.transition(TransferStatus.COMPLETED)
        except (TimeoutError, socket.timeout) as exc:
            handle.transition(TransferStatus.TIMED_OUT, str(exc))
        except Exception as exc:
            handle.transition(TransferStatus.FAILED, str(exc))

    def poll(self, handle: TransferHandle) -> TransferStatus:
        with self._lock:
            registered = self._handles.get(handle.transfer_id)
            if registered is not handle:
                raise ValueError("transfer handle does not belong to this backend")
            return handle.status

    def abort(self, handle: TransferHandle) -> None:
        # A blocking socket write cannot be safely interrupted from here. The
        # timeout bounds its lifetime and keeps memory registered until then.
        if not handle.status.is_terminal:
            handle.error = "cancellation requested; waiting for socket timeout"

    def close(self) -> None:
        with self._lock:
            active = [
                handle.transfer_id
                for handle in self._handles.values()
                if not handle.status.is_terminal
            ]
            if active:
                raise RuntimeError(
                    f"cannot close backend with active transfers: {active}"
                )
            self._closed = True
            self._registered.clear()
            self._descriptors.clear()
        if self._server is not None:
            self._server.close()
        if self._server_thread is not None:
            self._server_thread.join(timeout=1.0)
