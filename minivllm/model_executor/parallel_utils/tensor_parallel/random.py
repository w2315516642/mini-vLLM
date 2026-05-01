import torch

from minivllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank
)

def model_parallel_cuda_manual_seed(seed: int) -> None:
    """Initialize model parallel cuda seed.

    This function should be called after the model parallel is
    initialized. Also, no torch.cuda.manual_seed should be called
    after this function. Basically, this is replacement for that
    function.
    Two set of RNG states are tracked:
        default state: This is for data parallelism and is the same among a
                       set of model parallel GPUs but different across
                       different model paralle groups. This is used for
                       example for dropout in the non-tensor-model-parallel regions.
        tensor-model-parallel state: This state is different among a set of model
                              parallel GPUs, but the same across data parallel
                              groups. This is used for example for dropout in
                              model parallel regions.
    """
    offset = seed + 2592
    tensor_model_parallel_seed = offset + get_tensor_model_parallel_rank()
    # Data parallel gets the original seed.
    data_parallel_seed = seed

    # TODO
    # _CUDA_RNG_STATE_TRACKER.reset()

    torch.cuda.manual_seed(data_parallel_seed)

    # _CUDA_RNG_STATE_TRACKER.add(
    #     _MODEL_PARALLEL_RNG_TARCKER_NAME,
    #     tensor_model_parallel_seed)