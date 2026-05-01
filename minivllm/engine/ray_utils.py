from typing import Optional, Tuple

try:
    import ray
except ImportError:
    ray = None

# rank, node resource (node IP), device id
DeviceID = Tuple[int, Optional[str], int]