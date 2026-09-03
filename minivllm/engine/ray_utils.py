import random
from dataclasses import dataclass
from typing import Optional, Tuple, List

try:
    import ray
except ImportError:
    ray = None

from minivllm.configs import ParallelConfig

# rank, Ray node-affinity resource key, node-local device id
DeviceID = Tuple[int, Optional[str], int]


@dataclass(frozen=True)
class _RayGPUNode:
    address: str
    num_gpus: int

    @property
    def resource_key(self) -> str:
        # A Ray scheduling key is not itself a network address.
        return f"node:{self.address}"


def _get_gpu_nodes() -> List[_RayGPUNode]:
    """Read each live GPU node once, using Ray's explicit network address."""
    nodes = []
    for node in ray.nodes():
        if not node["Alive"]:
            continue
        # Each worker reserves one whole GPU; CPU-only nodes omit this resource.
        num_gpus = int(node["Resources"].get("GPU", 0))
        if num_gpus <= 0:
            continue
        # Resource labels (including node:__internal_head__) do not enumerate
        # machines. NodeManagerAddress is the address advertised by the raylet.
        nodes.append(_RayGPUNode(node["NodeManagerAddress"], num_gpus))
    return nodes


def initialize_cluster(
    parallel_config: ParallelConfig,
    engine_use_ray: bool = False,
    ray_address: Optional[str] = None,
) -> Tuple[str, List[List[DeviceID]]]:
    """Initialize the distributed cluster probably with Ray.

    Args:
        parallel_config: The configurations for parallel execution.
        engine_use_ray: Whether to use Ray for async engine.
        ray_address: The address of the Ray cluster. If None, uses
            the default Ray cluster address.

    Returns:
        A tuple of (`distributed_init_method`, `all_stage_devices`). The
        `distributed_init_method` is the address for initializing the
        distributed backend. `all_stage_devices` includes device IDs for
        each worker in each pipeline stage. Each device ID is a tuple of
        (rank, node resource, device id).
    """
    if parallel_config.worker_use_ray or engine_use_ray:
        if ray is None:
            raise ImportError(
                "Ray is not installed. Please install Ray to use distributed "
                "serving."
            )
        # Connect to a ray cluster.
        ray.init(address=ray_address)

    if not parallel_config.worker_use_ray:
        # Initialize cluster locally.
        port = random.randint(10000, 20000)
        # We need to setup the distributed init method to make sure
        # the distributed megatron code (e.g., get world size) works correctly.
        distributed_init_method = f"tcp://localhost:{port}"
        all_stage_devices = [[(0, None, 0)]]
        return distributed_init_method, all_stage_devices

    nodes = _get_gpu_nodes()
    if not nodes:
        raise ValueError("No alive Ray nodes with GPUs are available.")

    # The existing rank layout requires equal GPU capacity on every node.
    num_devices_per_node = nodes[0].num_gpus
    if any(node.num_gpus != num_devices_per_node for node in nodes):
        raise ValueError("The number of GPUs per node is not uniform.")
    if parallel_config.world_size > len(nodes) * num_devices_per_node:
        raise ValueError(
            "The number of required GPUs exceeds the total number of "
            "available GPUs."
        )
    if parallel_config.tensor_parallel_size >= num_devices_per_node:
        if parallel_config.tensor_parallel_size % num_devices_per_node != 0:
            raise ValueError(
                "The number of tensor parallelism is not divisible by the "
                "number of GPUs per node."
            )
    else:
        if num_devices_per_node % parallel_config.tensor_parallel_size != 0:
            raise ValueError(
                "The number of GPUs per node is not divisible by the number "
                "of tensor parallelism.")

    # Rank 0 is placed on the first GPU node and hosts the rendezvous store.
    port = random.randint(10000, 20000)
    distributed_init_method = f"tcp://{nodes[0].address}:{port}"

    # Assign contiguous ranks within each node, preserving the DeviceID API.
    rank = 0
    all_stage_devices = []

    for _ in range(parallel_config.pipeline_parallel_size):
        stage_devices = []
        for _ in range(parallel_config.tensor_parallel_size):
            node_index, device_id = divmod(rank, num_devices_per_node)
            node = nodes[node_index]
            stage_devices.append((rank, node.resource_key, device_id))
            rank += 1
        all_stage_devices.append(stage_devices)

    return distributed_init_method, all_stage_devices
