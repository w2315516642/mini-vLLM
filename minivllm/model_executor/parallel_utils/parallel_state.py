import torch
from typing import Optional, Dict

# Intra-layer model parallel group that the current rank belongs to.
_TENSOR_MODEL_PARALLEL_GROUP = None
# Inter-layer model parallel group that the current rank belongs to.
_PIPELINE_MODEL_PARALLEL_GROUP = None
# Model parallel group (both intra- and pipeline) that the current rank belongs to.
_MODEL_PARALLEL_GROUP = None
# Embedding group.
_EMBEDDING_GROUP = None
# Position embedding group.
_POSITION_EMBEDDING_GROUP = None
# Data parallel group that the current rank belongs to.
_DATA_PARALLEL_GROUP = None

_VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = None
_VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
_PIPELINE_MODEL_PARALLEL_SPLIT_RANK = None

# These values enable us to change the mpu sizes on the fly.
_MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = None
_MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
_MPU_TENSOR_MODEL_PARALLEL_RANK = None
_MPU_PIPELINE_MODEL_PARALLEL_RANK = None

# A list of ranks that have a copy of the embedding.
_EMBEDDING_GLOBAL_RANKS = None

# A list of ranks that have a copy of the position embedding.
_POSITION_EMBEDDING_GLOBAL_RANKS = None

# A list of global ranks for each pipeline group to ease calculation of the source
# rank when broadcasting from the first or last pipeline stage.
_PIPELINE_GLOBAL_RANKS = None

# A list of global ranks for each data parallel group to ease calculation of the source
# rank when broadcasting weights from src to all other data parallel ranks
_DATA_PARALLEL_GLOBAL_RANKS = None

_ALL_REDUCE_LAUNCHER: Optional['GraphAllReduce'] = None


def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    virtual_pipeline_model_parallel_size: Optional[int] = None,
    pipeline_model_parallel_split_rank: Optional[int] = None,
) -> None:
    """
    Initialize model data parallel groups.

    Arguments:
        tensor_model_parallel_size: number of GPUs used for tensor model parallelism.
        pipeline_model_parallel_size: number of GPUs used for pipeline model parallelism.
        virtual_pipeline_model_parallel_size: number of virtual stages (interleaved
                                              pipeline).
        pipeline_model_parallel_split_rank: for models with both encoder and decoder,
                                            rank in pipeline with split point.

    Let's say we have a total of 16 GPUs denoted by g0 ... g15 and we
    use 2 GPUs to parallelize the model tensor, and 4 GPUs to parallelize
    the model pipeline. The present function will
    create 8 tensor model-parallel groups, 4 pipeline model-parallel groups
    and 8 data-parallel groups as:
        8 data_parallel groups:
            [g0, g2], [g1, g3], [g4, g6], [g5, g7], [g8, g10], [g9, g11], [g12, g14], [g13, g15]
        8 tensor model-parallel groups:
            [g0, g1], [g2, g3], [g4, g5], [g6, g7], [g8, g9], [g10, g11], [g12, g13], [g14, g15]
        4 pipeline model-parallel groups:
            [g0, g4, g8, g12], [g1, g5, g9, g13], [g2, g6, g10, g14], [g3, g7, g11, g15]
    Note that for efficiency, the caller should make sure adjacent ranks
    are on the same DGX box. For example if we are using 2 DGX-1 boxes
    with a total of 16 GPUs, rank 0 to 7 belong to the first box and
    ranks 8 to 15 belong to the second box.
    """
    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    world_size: int = torch.distributed.get_world_size()

    if world_size % (tensor_model_parallel_size * pipeline_model_parallel_size) != 0:
        raise RuntimeError(
            f"world_size ({world_size}) is not divisible by "
            f"tensor_model_parallel_size ({tensor_model_parallel_size}) "
            f"x pipeline_model_parallel_size ({pipeline_model_parallel_size})"
        )
    
    data_parallel_size: int = world_size // (tensor_model_parallel_size *
                                             pipeline_model_parallel_size)
    
    num_tensor_model_parallel_groups: int = \
        world_size // tensor_model_parallel_size

    # TODO

    rank = torch.distributed.get_rank()

    # Build the tensor model-parallel groups.
    global _TENSOR_MODEL_PARALLEL_GROUP
    assert _TENSOR_MODEL_PARALLEL_GROUP is None, \
        'tensor model parallel group is already initialized'
    for i in range(num_tensor_model_parallel_groups):
        ranks = range(i * tensor_model_parallel_size,
                      (i + 1) * tensor_model_parallel_size)
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            _TENSOR_MODEL_PARALLEL_GROUP = group


def initialize_all_reduce_launcher(
    max_num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    disable_graph: bool = False,
) -> None:
    global _ALL_REDUCE_LAUNCHER
    _ALL_REDUCE_LAUNCHER = GraphAllReduce(
        max_num_tokens=max_num_tokens,
        hidden_size=hidden_size,
        dtype=dtype,
        disable_graph=disable_graph,
    )


def get_tensor_model_parallel_group():
    """Get the tensor model parallel group the caller rank belongs to."""
    assert _TENSOR_MODEL_PARALLEL_GROUP is not None, \
        'intra_layer_model parallel group is not initialized'
    return _TENSOR_MODEL_PARALLEL_GROUP


def get_tensor_model_parallel_rank():
    "Return the rank for the tensor model parallel group."
    global _MPU_TENSOR_MODEL_PARALLEL_RANK
    if _MPU_TENSOR_MODEL_PARALLEL_RANK is not None:
        return _MPU_TENSOR_MODEL_PARALLEL_RANK
    return torch.distributed.get_rank(group=get_tensor_model_parallel_group())


def get_tensor_model_parallel_world_size():
    global _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    if _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE is not None:
        return _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    return torch.distributed.get_world_size(group=get_tensor_model_parallel_group())


def model_parallel_is_initialized():
    """Check if model and data parallel groups are initialized."""
    if (
        _TENSOR_MODEL_PARALLEL_GROUP is None
        or _PIPELINE_MODEL_PARALLEL_GROUP is None
        or _DATA_PARALLEL_GROUP is None
    ):
        return False
    return True


def get_all_reduce_launcher() -> 'GraphAllReduce':
    assert _ALL_REDUCE_LAUNCHER is not None, 'all reduce launcher is not initialized'
    return _ALL_REDUCE_LAUNCHER


class GraphAllReduce:
    """Reduce valid token rows using fixed-address, aligned communication buffers.

    Callers write into get_buffer(N). Only the collective sees the padding;
    launch returns the same N-row view, so model code need not align its inputs.
    """

    _TOKEN_ALIGNMENT = 8

    def __init__(
        self,
        max_num_tokens: int,
        hidden_size: int,
        dtype: torch.dtype,
        disable_graph: bool = False,
    ) -> None:
        self.max_num_tokens = max_num_tokens
        self.hidden_size = hidden_size
        self.disable_graph = disable_graph
        self.buffer_capacity = self._aligned_num_tokens(max_num_tokens)

        tp_world_size = get_tensor_model_parallel_world_size()
        if tp_world_size == 1:
            return

        self.group = get_tensor_model_parallel_group()
        self.buffer = torch.empty(
            size=(self.buffer_capacity, hidden_size),
            dtype=dtype,
            device="cuda",
        )

        # The last bucket also covers a scheduler limit that is not aligned.
        if not self.disable_graph:
            self.graphs: Dict[int, torch.cuda.CUDAGraph] = {}
            for num_tokens in range(
                self._TOKEN_ALIGNMENT, self.buffer_capacity + 1,
                self._TOKEN_ALIGNMENT,
            ):
                self.graphs[num_tokens] = self._build_graph(num_tokens)

    @classmethod
    def _aligned_num_tokens(cls, num_tokens: int) -> int:
        alignment = cls._TOKEN_ALIGNMENT
        return (num_tokens + alignment - 1) // alignment * alignment

    def get_buffer(self, num_tokens: int) -> torch.Tensor:
        """Reserve a prefix before matmul, without resizing captured storage."""
        if not 0 <= num_tokens <= self.buffer_capacity:
            raise ValueError(
                f"All-reduce token count {num_tokens} is outside buffer capacity "
                f"[0, {self.buffer_capacity}]"
            )
        return self.buffer[:num_tokens]

    def _build_graph(self, num_tokens: int) -> torch.cuda.CUDAGraph:
        # Warm up
        torch.distributed.all_reduce(self.buffer[:num_tokens], group=self.group)
        torch.cuda.synchronize()

        # Build graph
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            torch.distributed.all_reduce(
                self.buffer[:num_tokens], group=self.group
            )
        torch.cuda.synchronize()
        return graph

    def launch(self, x: torch.Tensor) -> torch.Tensor:
        num_tokens = x.shape[0]
        expected = self.get_buffer(num_tokens)
        # Replay uses the captured pointer, not x; an arbitrary view is unsafe.
        if (x.shape != expected.shape or x.stride() != expected.stride()
                or x.dtype != expected.dtype or x.device != expected.device
                or x.data_ptr() != expected.data_ptr()):
            raise ValueError("All-reduce input must be the captured buffer prefix")
        if num_tokens == 0:
            return x
        if self.disable_graph:
            torch.distributed.all_reduce(x, group=self.group)
        else:
            graph_tokens = self._aligned_num_tokens(num_tokens)
            # Never reduce stale rows left by a previous, longer token batch.
            self.buffer[num_tokens:graph_tokens].zero_()
            self.graphs[graph_tokens].replay()
        return x
