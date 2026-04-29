from typing import List

from minivllm.utils.device import Device

class LogicalTokenBlock:
    "虚拟内存块，不代表真实内存情况"
    def __init__(
        self,
        block_id: int,      # 这是id吧
        block_size: int,    # 一个block能容纳多少个token
    ) -> None:
        self.block_id = block_id
        self.block_size = block_size    

        self.token_ids = [-1] * block_size
        self.num_tokens = 0

    def is_empty(self) -> bool:
        return self.num_tokens == 0
    
    def is_full(self) -> bool:
        return self.num_tokens == self.block_size

    def get_num_empty_slots(self) -> int:
        return self.block_size - self.num_tokens

    def append_tokens(self, token_ids: List[int]) -> None:
        if len(token_ids) > self.get_num_empty_slots():
            print(f"Failed to allocate token ids in block {self.block_id}")
            return
        self.token_ids[self.num_tokens:self.num_tokens + len(token_ids)] = token_ids
        self.num_tokens += len(token_ids)
        
    def get_token_ids(self) -> List[int]:
        return self.token_ids[:self.num_tokens] 
    
    def get_last_token_id(self) -> int:
        if self.num_tokens <= 0:
            print(f"Block {self.block_id} is empty!")
            return
        return self.token_ids[self.num_tokens - 1]

class PhysicalTokenBlock:
    "表示kvcache里面的物理块"
    def __init__(
        self,
        device: Device,
        block_id: int,
        block_size: int,
    ) -> None:
        self.device = device
        self.block_id = block_id
        self.block_size = block_size

        self.ref_count = 0

    def __repr__(self) -> str:
        return (f'PhysicalTokenBlock(device={self.device}, '
                f'block_number={self.block_number}, '
                f'ref_count={self.ref_count})')
