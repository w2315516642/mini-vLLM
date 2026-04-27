import torch
from typing import Tuple, List, Any
from transformers import AutoTokenizer, AutoModelForCausalLM


class ModelRunner:
    """使用 model() 重写transformers的generate方法"""
    def __init__(self, model_name="Qwen/Qwen3-0.6B"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        # self.model.to(self.device)
        self.model.eval()

    def forward(
        self, 
        input_ids: List[int], 
        is_prefill: bool, 
        past_kv: Any | None = None,
        attention_mask: List[int] = None
    ):
        input_batch = torch.tensor(input_ids, dtype=torch.int, device=self.device)

        if is_prefill:
            attn_mask = torch.tensor(attention_mask, dtype=torch.int, device=self.device)
            next_token_ids, new_kv = self.forward_prefill(input_batch, attn_mask)
        else:
            next_token_ids, new_kv = self.forward_decode(input_batch, past_kv)

        return next_token_ids.tolist(), new_kv

    def forward_prefill(
        self, 
        input_batch: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        "接收prompts，计算并返回所有KVCache-> next_is, past_kv"
        # encoded = self.tokenizer(
        #     input_batch,
        #     return_tensors='pt',
        #     padding=True,
        #     truncation=True,
        # ).to(self.device)

        # inference
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_batch,
                attention_mask=attention_mask,
                use_cache=True
            )
        
        logits = outputs.logits
        past_kv = outputs.past_key_values

        # 采用贪婪采样
        next_ids = torch.argmax(logits[:, -1, :], dim=-1)
        return next_ids, past_kv

    def forward_decode(
        self, 
        token_ids: torch.Tensor, 
        past_kv
    ):
        "借助prefill时得到的kvcache，节省后续decode的计算时间"
        outputs = self.model(
            input_ids=token_ids,
            past_key_values=past_kv,
            use_cache=True
        )
        logits = outputs.logits
        past_kv = outputs.past_key_values

        next_ids = torch.argmax(logits[:, -1, :], dim=-1)
        return next_ids, past_kv

    # @property
    # def eos_id(self):
    #     return self.tokenizer.encode(self.tokenizer)


from pathlib import Path
if __name__ == "__main__":
    model_path = Path(__file__).parent.parent.parent / "models" / "Qwen3-0.6B"

    runner = ModelRunner(model_path)
    print(f"tokenizer: {runner.tokenizer.eos_token_id}")

    print("im_end ID:", runner.tokenizer.convert_tokens_to_ids("<|im_end|>"))
    print("eos ID:", runner.tokenizer.eos_token_id)
    prompts = ["what is your name?", "hello?"]
    next_ids_pre, past_kv_pre = runner.forward_prefill(prompts)
    next_ids, past_kv = runner.forward_decode(next_ids_pre, past_kv_pre)
    print("answer: ")
    print(f"prefill tokens: {next_ids_pre}, prefill keys shape: {past_kv_pre.layers[0].keys.shape}")
    print(f"decode token: {next_ids}, prefill keys shape: {past_kv.layers[0].keys.shape}")
    print(f"KV Type: {type(past_kv_pre)}")
    print(f"Layer type: {type(past_kv_pre.layers[0])}")
    print(f"Attribute: {dir(past_kv_pre.layers[0])}")