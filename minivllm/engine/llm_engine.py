from typing import Any, List


from loguru import logger

from .model_runner import ModelRunner

from minivllm.configs import (
    ModelConfig, ParallelConfig, CacheConfig, SchedulerConfig
)

from minivllm.engine.arg_utils import EngineArgs
from minivllm.engine.ray_utils import DeviceID, ray
from minivllm.outputs import RequestOutput
from minivllm.sampling_params import SamplingParams
from minivllm.sequence import Sequence, SequenceGroup, SequenceStatus
from minivllm.utils import Counter
from minivllm.worker.worker import Worker

class LLMEngine:
    """ 核心，负责模型初始化、资源分配和推理 """
    def __init__(
        self, 
        model_config: ModelConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        distributed_init_method: str,
        stage_devices: List[List[DeviceID]],
        log_stats: bool,
    ) -> None:
        self.model_config = model_config
        self.cache_config = cache_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.log_stats = log_stats
        self._verify_args()

        self.tokenizer = get_tokenizer(model_config.model)
        self.seq_counter = Counter()

        # Create the parallel GPU workers.
        self.workers: List[Worker] = []
        assert len(stage_devices) == 1, "Only support one stage for now."
        for rank, node_resource, _ in stage_devices[0]:
            worker_cls = Worker
            if self.parallel_config.worker_use_ray:
                worker_cls = ray.remote(
                    num_cpus=0,
                    num_gpus=1,
                    resources={node_resource: 1e-5}
                )(worker_cls).remote
            
            worker = worker_cls(
                model_config,
                parallel_config,
                scheduler_config,
                rank,
                distributed_init_method
            )
            self.workers.append(worker)
        
        self._init_cache()

        # TODO: Create scheduler
        self.scheduler = Scheduler(scheduler_config, cache_config, log_stats)
    
    def _verify_args(self) -> None:
        self.model_config.verify_with_parallel_config(self.parallel_config)
        self.cache_config.verify_with_parallel_config(self.parallel_config)

    def _init_cache(self) -> None:
        """Profiles the memory usage and initializes the KV cache."""
        # Get the maximum number of blocks that can be allocated on GPU and CPU.
        num_blocks = self._run_workers(
            "profile_num_available_blocks",
            get_all_outputs=True,
            block_size=self.cache_config.block_size,
            gpu_memory_utilization=self.cache_config.gpu_memory_utilization,
            cpu_swap_space=self.cache_config.swap_space_bytes,
        )

        # Since we use a shared centralized controller, we take the minimum
        # number of blocks across all workers to make sure all the memory
        # operators can be applied to all workers.
        num_gpu_blocks = min(b[0] for b in num_blocks)
        num_cpu_blocks = min(b[1] for b in num_blocks)
        # Log
        logger.info(f'# GPU blocks: {num_gpu_blocks}, '
                    f'# CPU blocks: {num_cpu_blocks}')
        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks
        
        # Initialize the cache.
        self._run_workers("init_cache_engine", cache_config=self.cache_config)

    @classmethod
    def from_engine_args(cls, engine_args: EngineArgs) -> "LLMEngine":
        """Creates an LLM engine from the engine arguments."""
        # Create the engine configs.
        engine_configs = engine_args.create_engine_configs()
        parallel_config = engine_configs[2]
        # Initialize the cluster.
        distributed_init_method, devices = initialize_cluster(parallel_config)
        # Create the LLM engine.
        engine = cls(*engine_configs, distributed_init_method, devices,
                    log_stats=not engine_args.disable_log_stats)
        return engine


    def step(self) -> List[RequestOutput]:
        """Performs one decoding iteration and returns newly generated results.

        This function performs one decoding iteration of the engine. It first
        schedules the sequences to be executed in the next iteration and the
        token blocks to be swapped in/out/copy. Then, it executes the model
        and updates the scheduler with the model outputs. Finally, it decodes
        the sequences and returns the newly generated results.
        """
        seq_group_metadata_list, scheduler_outputs = self.scheduler.schedule()
        if (not seq_group_metadata_list) and scheduler_outpus.is_empty():
            # Nothing to do.
            return []
        
        # Execute the model.
        output = self._run_workers(
            "execute_model",
            seq_group_metadata_list=seq_group_metadata_list,
            blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in,
            blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,
            blocks_to_copy=scheduler_outputs.blocks_to_copy
        )
        # Update the scheduler with the model outputs.
        seq_groups = self.scheduler.update(output)

        # Decode the sequences.
        self._decode_sequences(seq_groups)
        # Stop the sequences that meet the stopping criteria.
        self._stop_sequences(seq_groups)
        # Free the finished sequence groups.
        self.scheduler.free_finished_seq_groups()

        # Create the outputs.
        request_outputs: List[RequestOutput] = []
        for seq_group in seq_groups:
            request_output = RequestOutput.from_seq_group(seq_group)
            request_outputs.append(request_output)
        return request_outputs

    def _run_workers(
        self,
        method: str,
        get_all_outputs: bool = False,
        *args,
        **kwargs
    ) -> Any:
        """Runs the given method on all workers."""
        all_outputs = []
        for worker in self.workers:
            executor = getattr(worker, method)
            if self.parallel_config.worker_use_ray:
                executor = executor.remote

            output = executor(*args, **kwargs)
            all_outputs.append(output)
        
        if self.parallel_config.worker_use_ray:
            all_outputs = ray.get(all_outputs)
        
        if get_all_outputs:
            return all_outputs
        
        # Make sure all workers have the same results.
        output = all_outputs[0]
        for other_output in all_outputs[1]:
            assert output == other_output
        return output


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