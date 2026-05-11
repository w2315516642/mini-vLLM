from typing import Callable, Any, NewType, List, Optional
import hashlib
import pickle
import os

from minivllm.sequence import Sequence


BlockHash = NewType("BlockHash", bytes)
BlockHasher = Callable[[Sequence], List[BlockHash]]

NONE_HASH: BlockHash

def init_none_hash(hash_fn: Callable[[Any], bytes]) -> None:
    global NONE_HASH

    hash_seed = os.getenv("PYTHONHASHSEED")
    if hash_seed is None:
        NONE_HASH = BlockHash(os.urandom(42))
    else:
        NONE_HASH = BlockHash(hash_fn(hash_seed))


def sha256(input_: Any) -> bytes:
    """Hash any picklable Python object using SHA-256.

    The input is serialized using pickle before hashing, which allows
    arbitrary Python objects to be used. Note that this function does
    not use a hash seed—if you need one, prepend it explicitly to the input.

    Args:
        input: Any picklable Python object.

    Returns:
        Bytes representing the SHA-256 hash of the serialized input.
    """
    input_bytes = pickle.dumps(input_, protocol=pickle.HIGHEST_PROTOCOL)
    return hashlib.sha256(input_bytes).digest()


def get_hash_fn_by_name(hash_fn_name: str) -> Callable[[Any], bytes]:
    if hash_fn_name == "sha256":
        return sha256

    raise NotImplementedError(f"Not supported hash function {hash_fn_name}")

    
def hash_block_tokens(
    hash_function: Callable[[Any], bytes],
    parent_block_hash: Optional[BlockHash],
    block_tokens_ids: List[int],
) -> BlockHash:
    if parent_block_hash is None:
        parent_block_hash = NONE_HASH
    
    block_tokens_ids_tuple = tuple(block_tokens_ids)
    return BlockHash(
        hash_function((parent_block_hash, block_tokens_ids_tuple))
    )


def get_seq_block_hasher(
    caching_hash_fn: Callable[[Any], bytes],
) -> BlockHasher:
    """
    Returns a function which computes the list of un-computed block hashes
    of a sequence group."""

    def seq_block_hasher(seq: Sequence) -> List[BlockHash]:

        if seq.is_finished():
            return []

        start_token_idx = len(seq.block_hashes) * seq.block_size
        end_token_idx = start_token_idx + seq.block_size
        num_tokens = seq.get_len()

        new_block_hashes = []
        prev_block_hash = seq.block_hashes[-1] if seq.block_hashes else None
        while end_token_idx <= num_tokens:
            all_token_ids = seq.get_token_ids()
            block_tokens = all_token_ids[start_token_idx:end_token_idx]

            block_hash = hash_block_tokens(
                caching_hash_fn, block_tokens, prev_block_hash
            )

            new_block_hashes.append(block_hash)

            start_token_idx += seq.block_size
            end_token_idx = start_token_idx + seq.block_size
            prev_block_hash = block_hash

        return new_block_hashes

    return seq_block_hasher

