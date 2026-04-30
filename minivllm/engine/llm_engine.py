from typing import List

from .model_runner import ModelRunner
from minivllm.sequence import Sequence, SequenceGroup, SequenceStatus
from minivllm.utils.counter import Counter
from minivllm.worker.worker import Worker

class LLMEngine:
    """ 核心，负责模型初始化、资源分配和推理 """
    def __init__(
        self, 
        model_config,
        cache_config,
        parallel_config,
        scheduler_config,
        distributed_init_method: str,
        stage_devices,
        log_stats: bool,
    ) -> None:

        self.model_config = model_config
        self.cache_config = cache_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.log_stats = log_stats

        self.seq_counter = Counter()

        self.workers: List[Worker] = []

        self.max_model_len = max_model_len

        # 1. 获取模型配置

        # 2. 加载模型
        self.runner = ModelRunner(model)

        # 3. 分配 KVCache 空间

        # 4. 初始化推理相关的成员
        self.eos_id_list = [151643]

        # 运行池
        self.running: List[Sequence] = []
    
    def step(self):
        # 1. 找到需要处理的请求（应该是schedular负责的，先放这）
        request = None
        for req in self.running:
            if not req.is_finished:
                request = req
                break
        
        if request is None:
            return []
        
        # 2. 进行一次推理
        next_token_ids, past_kv = self.runner.forward(
            input_ids=request.next_token,
            is_prefill=request.is_prefill,
            past_kv=request.kv_cache,
            attention_mask=request.attention_mask,
        )
        request.append_token(next_token_ids[-1])

        # 3. 更新请求状态
        if request.is_prefill:
            request.decoding()
        if (
            next_token_ids[-1] in self.eos_id_list 
            or len(request) >= self.max_model_len
        ):
            request.finished()
            return [(self.next_request_id, request.output_token_ids)]
        else:
            request.kv_cache = past_kv
            return []


    def generate(self, prompts: str):
        # 1. 预处理转成token-ids
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompts}
        ]
        text = self.runner.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        ) + "</think>\n"
        print(f"prompts:\n {text}")
        # 对prompts进行tokenize
        encoded = self.runner.tokenizer(
            [text], return_tensor='pt'
            ).to(self.runner.device)
        
        # 2. 把新输入的token-ids转换成一个请求（要根据请求参数设置seqg）

        request = Sequence(
            token_ids=encoded.input_ids[0], 
            attention_mask=encoded.attention_mask
        )
        self.next_request_id += 1

        # 2. 添加请求
        self.running.append(request)

        outputs = {}
        while not self.is_finished():
            output = self.step()
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids

        # 解码并整成列表输出
        outputs = [{
            "request_id": seq_id, 
            "text": self.runner.tokenizer.decode(outputs[seq_id])} 
            for seq_id in sorted(outputs.keys())]

        return outputs

    def is_finished(self):
        for request in self.running:
            if not request.is_finished:
                return False
        return True 


if __name__ == "__main__":
    from pathlib import Path

    model_path = Path(__file__).parent.parent.parent / "models" / "Qwen3-0.6B"
    prompts = "Hello?"
    engine = LLMEngine(model_path, 64)
    outputs = engine.generate(prompts)
    print(outputs)