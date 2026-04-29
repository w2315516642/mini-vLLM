from typing import Optional, Union, List

class SamplingParams:
    """ 推理过程中涉及的参数 
    
    Args:
        n: 输出序列的数量
        best_of: 推理过程中保留的序列数量，输出序列从这里面选n条，所以要保证best of大于等于n
        temperature: 温度参数
        top_p:
        top_k:
        use_beam_search:
        stop: 字符串列表，生成过程中遇到这些字符串就会停止生成。返回的prompts中不会包含这些字符串
        ignore_eos: 是否无视 EOS token继续生成
        max_tokens: 最大生成token数量
        logprobs: 每个输出token返回的log概率数量
    """
    def __init__(
        self,
        n: int = 1,
        best_of: Optional[int] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        use_beam_search: bool = False,
        stop: Union[str, List[str]] = [],
        ignore_eos: bool = False,
        max_tokens: int = 16,
        logprobs: Optional[int] = None,
    ) -> None:
        self.n = n
        self.best_of = best_of if best_of is not None else n
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.use_beam_search = use_beam_search
        self.stop = [stop] if isinstance(stop, str) else stop
        self.ignore_eos = ignore_eos 
        self.max_tokens = max_tokens
        self.logprobs = logprobs

        self._verify_args()


    def _verify_args(self) -> None:
        assert self.n >= 1, f"n must be at least 1, got {self.n}"
        assert self.best_of >= self.n, (f"best_of must be greater than or equal to n" + 
                                        f"got n={self.n} and best_of={self.best_of}")
        assert self.temperature >= 0.0, f"temperature must be non-negetive, got {self.temperature}"
        assert self.top_p > 0.0 and self.top_p <= 1, (
            f"top_p must be in (0, 1], got {self.top_p}")
        assert self.top_k == -1 or self.top_k > 0, (
            f"top_k must be -1 (disable), or at least 1, got {self.top_k}")
        assert self.max_tokens >= 1, f"max_tokens must be at least 1, got {self.max_tokens}"
        assert self.logprobs >= 0 if self.logprobs is not None else True, (
            f"logprobs must be non-negetive, got {self.logprobs}")

    def __repr__(self) -> str:
        return (f"SamplingParams(n={self.n}, "
                f"best_of={self.best_of}, "
                f"temperature={self.temperature}, "
                f"top_p={self.top_p}, "
                f"top_k={self.top_k}, "
                f"use_beam_search={self.use_beam_search}, "
                f"stop={self.stop}, "
                f"ignore_eos={self.ignore_eos}, "
                f"max_tokens={self.max_tokens}, "
                f"logprobs={self.logprobs})")