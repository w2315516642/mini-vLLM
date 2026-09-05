from typing import Callable
from minivllm.profiling import nvtx_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter

from minivllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_all_reduce_launcher,
)
from .mappings import (
    gather_from_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
    scatter_to_tensor_model_parallel_region,
)
# from .random import get_cuda_rng_tracker
from .utils import divide, VocabUtility

_MODEL_PARALLEL_ATTRIBUTE_DEFAULTS = {
    'tensor_model_parallel': False,
    'partition_dim': -1,
    'partition_stride': 1
}

# The software W8A16 kernel wins on bandwidth-bound decode/verification shapes.
# Large prefill still uses cuBLAS until its fused tile path is competitive.
FP8_FUSED_MAX_TOKENS = 512


def _config_value(config, name, default=None):
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def get_fp8_block_size(quant_config):
    """Return ``(output_block, input_block)`` for an FP8 checkpoint."""
    if quant_config is None:
        return None
    quant_method = _config_value(quant_config, "quant_method")
    quant_algo = _config_value(quant_config, "quant_algo")
    block_size = _config_value(quant_config, "weight_block_size")
    if quant_method != "fp8" and quant_algo not in {"FP8_PB_WO", "FP8"}:
        return None
    if not isinstance(block_size, (list, tuple)) or len(block_size) != 2:
        raise ValueError(
            "FP8 block quantization requires weight_block_size=[N, K]"
        )
    if any(
        not isinstance(size, int) or isinstance(size, bool) or size <= 0
        for size in block_size
    ):
        raise ValueError("FP8 weight block dimensions must be positive integers")
    return tuple(block_size)


def dequantize_fp8_block_weight(
    weight: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    block_size,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Materialize one block-scaled FP8 matrix in the compute dtype."""
    if weight.ndim != 2 or weight_scale_inv.ndim != 2:
        raise ValueError("FP8 weight and weight_scale_inv must be matrices")
    block_n, block_k = block_size
    expected_scale_shape = (
        (weight.shape[0] + block_n - 1) // block_n,
        (weight.shape[1] + block_k - 1) // block_k,
    )
    if tuple(weight_scale_inv.shape) != expected_scale_shape:
        raise ValueError(
            "weight_scale_inv has shape "
            f"{tuple(weight_scale_inv.shape)}, expected {expected_scale_shape}"
        )
    if weight.shape[0] % block_n == 0 and weight.shape[1] % block_k == 0:
        dequantized = weight.to(dtype).view(
            weight.shape[0] // block_n,
            block_n,
            weight.shape[1] // block_k,
            block_k,
        )
        dequantized.mul_(weight_scale_inv.to(dtype)[:, None, :, None])
        return dequantized.view_as(weight)

    # Tail blocks are uncommon in Qwen's quantized linears, but supporting
    # them keeps the loader usable for small test and fine-tuned checkpoints.
    expanded_scale = weight_scale_inv.to(dtype).repeat_interleave(
        block_n, dim=0
    ).repeat_interleave(block_k, dim=1)
    expanded_scale = expanded_scale[: weight.shape[0], : weight.shape[1]]
    dequantized = weight.to(dtype)
    return dequantized.mul_(expanded_scale)


def _linear_weight(module: nn.Module, input_: torch.Tensor) -> torch.Tensor:
    if not hasattr(module, "weight_scale_inv"):
        return module.weight
    return dequantize_fp8_block_weight(
        module.weight,
        module.weight_scale_inv,
        module.weight_block_size,
        input_.dtype,
    )


@nvtx_function("linear")
def _linear(module, input_, bias=None, out=None):
    # Select by the local GEMM's M, not by request count: speculative verification
    # packs several tokens per request. Weights always remain resident in FP8;
    # only large-M/CPU/FP32 calls materialize a temporary per-layer weight.
    if hasattr(module, "weight_scale_inv") and input_.is_cuda and input_.dtype in (
        torch.float16, torch.bfloat16,
    ) and input_.numel() // input_.shape[-1] <= FP8_FUSED_MAX_TOKENS:
        from minivllm.model_executor.layers.fp8_linear import fp8_linear
        return fp8_linear(input_, module.weight, module.weight_scale_inv,
                          module.weight_block_size, bias, out)
    weight = _linear_weight(module, input_)
    if out is None:
        return F.linear(input_, weight, bias)
    torch.matmul(input_, weight.t(), out=out)
    if bias is not None:
        out.add_(bias)
    return out


def set_tensor_model_parallel_attributes(
    tensor: torch.Tensor,
    is_parallel: bool,
    dim: int,
    stride: int,
) -> None:
    # Make sure the attributes are not set.
    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        assert not hasattr(tensor, attribute)
    # Set the attributes.
    setattr(tensor, 'tensor_model_parallel', is_parallel)
    setattr(tensor, 'partition_dim', dim)
    setattr(tensor, 'partition_stride', stride)


class VocabParallelEmbedding(nn.Module):
    """Embedding parallelized in the vocabulary dimension.

    This is mainly adapted from torch.nn.Embedding and all the default
    values are kept.
    Arguments:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.

    Keyword Arguments:
        init_method: method to initialize weights.
        params_dtype
        use_cpu_initialization
        perform_initialization
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        init_method: Callable = init.xavier_normal_,
        params_dtype: torch.dtype = None,
        use_cpu_initialization: bool = False,
        perform_initialization: bool = False,
    ) -> None:
        super().__init__()
        # Keep the input dimensions.
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()

        # Set the defaults for compatibility.
        self.padding_idx = None
        self.max_norm = None
        self.norm_type = 2.
        self.scale_grad_by_freq = False
        self.sparse = False
        self._weight = None
        self.tensor_model_parallel_size = get_tensor_model_parallel_world_size()
        # Divide the weight matrix along the vocaburaly dimension.
        self.vocab_start_index, self.vocab_end_index = \
            VocabUtility.vocab_range_from_global_vocab_size(
                self.num_embeddings,
                get_tensor_model_parallel_rank(),
                self.tensor_model_parallel_size,
            )
        self.num_embeddings_per_partition = (
            self.vocab_end_index - self.vocab_start_index
        )

        # Allocate weights and initialize.
        if use_cpu_initialization:
            self.weight = Parameter(torch.empty(
                self.num_embeddings_per_partition,
                self.embedding_dim,
                dtype=params_dtype,
            ))
            # TODO
            # if perform_initialization:
            #     _initialize_affine_weight_cpu()
        else:
            self.weight = Parameter(torch.empty(
                self.num_embeddings_per_partition,
                self.embedding_dim,
                device=torch.cuda.current_device(),
                dtype=params_dtype,
            ))
            # TODO
            # if perform_initialization:
    
    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        if self.tensor_model_parallel_size > 1:
            # Build the mask.
            input_mask = (
                (input_ < self.vocab_start_index) |
                (input_ >= self.vocab_end_index)
            )
            # Mask the input.
            masked_input = input_.clone() - self.vocab_start_index
            masked_input[input_mask] = 0
        else:
            masked_input = input_
        # Get the embeddings.
        output_parallel = F.embedding(masked_input, self.weight,
                                      self.padding_idx, self.max_norm,
                                      self.norm_type, self.scale_grad_by_freq,
                                      self.sparse)
        # Mask the output embedding.
        if self.tensor_model_parallel_size > 1:
            output_parallel[input_mask, :] = 0.0
        # Reduce across all the model parallel GPUs.
        output = reduce_from_tensor_model_parallel_region(output_parallel)
        return output


class ColumnParallelLinear(nn.Module):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.

    Keyword Arguments
        bias: If true, add bias
        gather_output: If true, call all-gather on output and make Y available
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
        skip_bias_add: This was added to enable performance optimations where bias
                       can be fused with other elementwise operations. we skip
                       adding bias but instead return it.
        params_dtype:
        use_cpu_initialization:
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        bias: bool = True,
        gather_output: bool = True,
        init_method: Callable = init.xavier_normal_,
        stride: int = 1,
        keep_master_weight_for_test: bool = False,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype = None,
        use_cpu_initialization: bool = False,
        perform_initialization: bool = False,
        quant_config=None,
    ) -> None:
        super().__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size 
        self.gather_output = gather_output 
        # Divide the weight matrix along the last dimension.
        world_size = get_tensor_model_parallel_world_size()
        self.output_size_per_partition = divide(output_size, world_size)
        # print(f"col output_size_per_partition: {self.output_size_per_partition}, "
        #       f"total output size: {output_size}, "
        #       f"world_size: {world_size}")
        self.skip_bias_add = skip_bias_add

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()

        # FP8 checkpoints keep the compressed matrix resident. The small scale
        # matrix is partitioned in the same direction as the weight.
        self.weight_block_size = get_fp8_block_size(quant_config)
        weight_dtype = params_dtype
        requires_grad = True
        if self.weight_block_size is not None:
            weight_dtype = getattr(torch, "float8_e4m3fn", None)
            if weight_dtype is None:
                raise RuntimeError("This PyTorch build does not support FP8")
            requires_grad = False
        weight_device = (
            None if use_cpu_initialization else torch.cuda.current_device()
        )
        self.weight = Parameter(
            torch.empty(
                self.output_size_per_partition,
                self.input_size,
                device=weight_device,
                dtype=weight_dtype,
            ),
            requires_grad=requires_grad,
        )
        if self.weight_block_size is not None:
            block_n, block_k = self.weight_block_size
            scale_shape = (
                (self.output_size_per_partition + block_n - 1) // block_n,
                (self.input_size + block_k - 1) // block_k,
            )
            self.weight_scale_inv = Parameter(
                torch.ones(
                    scale_shape,
                    device=weight_device,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
        
        if bias:
            if use_cpu_initialization:
                self.bias = Parameter(torch.empty(
                    self.output_size_per_partition, dtype=params_dtype,
                ))
            else:
                self.bias = Parameter(torch.empty(
                    self.output_size_per_partition,
                    device=torch.cuda.current_device(),
                    dtype=params_dtype,
                ))
            set_tensor_model_parallel_attributes(self.bias, True, 0, stride)
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter('bias', None)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """Forward of ColumnParallelLinear

        Args:
            input_: 3D tensor whose order of dimension is [sequence, batch, hidden]

        Returns:
            - output
            - bias
        """
        bias = self.bias if self.skip_bias_add else None

        input_parallel = input_
        # Matrix multiply.
        output_parallel = _linear(self, input_parallel, bias)
        if self.gather_output:
            # All-gather across the partitions.
            output = gather_from_tensor_model_parallel_region(output_parallel)
        else:
            output = output_parallel
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias

    
class RowParallelLinear(nn.Module):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.

    Keyword Arguments:
        bias: If true, add bias. Note that bias is not parallelized.
        input_is_parallel: If true, we assume that the input is already
                           split across the GPUs and we do not split
                           again.
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
        skip_bias_add: This was added to enable performance optimization where bias
                       can be fused with other elementwise operations. We skip
                       adding bias but instead return it.
        params_dtype:
        use_cpu_initialization:
        perform_initialization:
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        bias: bool = True,
        input_is_parallel: bool = False,
        init_method: Callable = init.xavier_normal_,
        stride: int = 1,
        keep_master_weight_for_test: bool = False,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype = None,
        use_cpu_initialization: bool = False,
        perform_initialization: bool = False,
        quant_config=None,
    ) -> None:
        super().__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()

        # Divide the weight matrix along the last dimension.
        world_size = get_tensor_model_parallel_world_size()
        self.input_size_per_partition = divide(input_size, world_size)
        self.skpi_bias_add = skip_bias_add

        self.weight_block_size = get_fp8_block_size(quant_config)
        weight_dtype = params_dtype
        requires_grad = True
        if self.weight_block_size is not None:
            weight_dtype = getattr(torch, "float8_e4m3fn", None)
            if weight_dtype is None:
                raise RuntimeError("This PyTorch build does not support FP8")
            requires_grad = False
        weight_device = (
            None if use_cpu_initialization else torch.cuda.current_device()
        )
        self.weight = Parameter(
            torch.empty(
                self.output_size,
                self.input_size_per_partition,
                device=weight_device,
                dtype=weight_dtype,
            ),
            requires_grad=requires_grad,
        )
        if self.weight_block_size is not None:
            block_n, block_k = self.weight_block_size
            scale_shape = (
                (self.output_size + block_n - 1) // block_n,
                (self.input_size_per_partition + block_k - 1) // block_k,
            )
            self.weight_scale_inv = Parameter(
                torch.ones(
                    scale_shape,
                    device=weight_device,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
        
        if bias:
            if use_cpu_initialization:
                self.bias = Parameter(torch.empty(
                    self.output_size, dtype=params_dtype,
                ))
            else:
                self.bias = Parameter(torch.empty(
                    self.output_size,
                    device=torch.cuda.current_device(),
                    dtype=params_dtype,
                ))
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter('bias', None)
        self.weight_t = self.weight.t()
    
    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """Forward of RowParallelLinear

        Args:
            input_: 3D tensor whose order of dimension is [sequence, batch, hidden]

        Returns:
            - output
            - bias
        """
        # Set up backprop all-reduce.
        if self.input_is_parallel:
            input_parallel = input_
        else:
            input_parallel = scatter_to_tensor_model_parallel_region(input_)
        # Matrix multiply.
        if get_tensor_model_parallel_world_size() == 1:
            output_ = _linear(self, input_parallel)
        else:
            all_reduce_launcher = get_all_reduce_launcher()
            num_tokens = input_parallel.shape[0]
            # 取cuda graph要用的固定显存位置
            output_buffer = all_reduce_launcher.get_buffer(num_tokens)
            _linear(self, input_parallel, out=output_buffer)
            # All-reduce across all the partitions.
            output_ = all_reduce_launcher.launch(output_buffer)

        if not self.skpi_bias_add:
            output = (output_ + self.bias) if self.bias is not None else output_
            output_bias = None
        else:
            output = output_
            output_bias = self.bias
        return output, output_bias
