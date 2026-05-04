import torch

from minivllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_group,
)
from .utils import split_tensor_along_last_dim


def _reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""

    # Bypass the function if we are using only 1 GPU.
    if get_tensor_model_parallel_world_size() == 1:
        return input_

    torch.distributed.all_reduce(input_, group=get_tensor_model_parallel_group())
    return input_


def _split_along_last_dim(input_: torch.Tensor) -> torch.Tensor:
    """Split the tensor along its last dimension and keep the
    corresponding slice."""

    world_size = get_tensor_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    # Split along last dimension.
    input_list = split_tensor_along_last_dim(input_, world_size)

    # Note: torch.split does not create contiguous tensors by default.
    rank = get_tensor_model_parallel_rank()
    output = input_list[rank].contiguous()
    return output


def _gather_along_last_dim(input_: torch.Tensor) -> torch.Tensor:
    """Gather tensors and concatinate along the last dimension."""

    world_size = get_tensor_model_parallel_world_size()
    if world_size == 1:
        return input_

    # Size and dimension.
    last_dim = input_.dim() - 1
    rank = get_tensor_model_parallel_rank()

    buffer_list = [torch.empty_like(input_) for _ in range(world_size)]
    buffer_list[rank] = input_
    torch.distributed.all_gather(
        buffer_list, input_, group=get_tensor_model_parallel_group())
    
    # NOTE: torch.cat already creates a contiguous tensor.
    output = torch.cat(buffer_list, dim=last_dim).contiguous()
    return output


class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """All-reduce the input from the model parallel region."""

    @staticmethod
    def symbolic(
        graph: torch.cuda.CUDAGraph,
        input_: torch.Tensor,
    ) -> torch.Tensor:
        return _reduce(input_)
    
    @staticmethod
    def forward(ctx, input_: torch.Tensor) -> torch.Tensor:
        return _reduce(input_)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class _ScatterToModelParallelRegion(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    @staticmethod
    def symbolic(graph, input_):
        return _split_along_last_dim(input_)

    @staticmethod
    def forward(ctx, input_):
        return _split_along_last_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather_along_last_dim(grad_output)


class _GatherFromModelParallelRegion(torch.autograd.Function):
    """Gather the input from model parallel region and concatinate."""

    @staticmethod
    def symbolic(graph, input_):
        return _gather_along_last_dim(input_)

    @staticmethod
    def forward(ctx, input_):
        return _gather_along_last_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _split_along_last_dim(grad_output)


def reduce_from_tensor_model_parallel_region(input_):
    return _ReduceFromModelParallelRegion.apply(input_)


def scatter_to_tensor_model_parallel_region(input_):
    return _ScatterToModelParallelRegion.apply(input_)


def gather_from_tensor_model_parallel_region(input_):
    return _GatherFromModelParallelRegion.apply(input_)