import torch
import psutil
from enum import Enum, auto

class Device(Enum):
    CPU = auto()
    GPU = auto()

def get_gpu_memory(rank: int = 0) -> int:
    "Return the total memory of the GPU in bytes."
    return torch.cuda.get_device_properties(rank).total_memory

def get_cpu_memory() -> int:
    "Return the total memory of the CPU node in bytes."
    return psutil.virtual_memory().total