from typing import Dict, List
import json
from pathlib import Path

from transformers import AutoTokenizer


class TrieNode:
    def __init__(self) -> None:
        self.children: Dict[int, 'TriNode'] = {}
        self.ref_count: int = 0


class PrefixCachingCalculator:
    def __init__(self, model_name_or_path: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.max_num_tokens = self.tokenizer.model_max_length
        self.root = TrieNode()
        self.total_tokens = 0
        self.shared_tokens = 0

    def insert(self, token_ids: List[int]) -> None:
        node = self.root
        is_shared = True

        if len(token_ids) > self.max_num_tokens:
            token_ids = token_ids[-self.max_num_tokens:]
        
        for token_id in token_ids:
            if is_shared and token_id in node.children:
                self.shared_tokens += 1
            else:
                is_shared = False

            if token_id not in node.children:
                node.children[token_id] = TrieNode()
                
            node = node.children[token_id]
            node.ref_count += 1
        
        self.total_tokens += len(token_ids)

    def from_sharegpt(self, path: str, max_num_data: int | None = None, is_jsonl: bool = False) -> None:
        if is_jsonl:
            data = None
            raise NotImplementError("Not supported jsonl file yet.")
        else:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
        if max_num_data is None:
            num_data = len(data)
        else:
            num_data = max(max_num_data, len(data))
            
        for idx in range(num_data):
            conversations = data[idx]['conversations']
            prompt_history = ""
            for i in range(0, len(conversations), 2):
                content = conversations[i]['value']
                prompt = f"{prompt_history}\nhuman: {content}\nAssistant: "

                token_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
                self.insert(token_ids)

                if (i + 1) < len(conversations):
                    assistant_content = conversations[i + 1]['value']
                    prompt_history += f"\nhuman: {content}\nAssistant: {assistant_content}"

    def get_psr(self) -> float:
        return self.shared_tokens / self.total_tokens
    

DATASETS_DIR = Path(__file__).parent.parent / "datasets"
MODELS_DIR = Path(__file__).parent.parent.parent / "models"

if __name__ == "__main__":
    data_path = DATASETS_DIR / "share_gpt.json"
    model_path = MODELS_DIR / "open_llama_7b"
    calculator = PrefixCachingCalculator(model_path)
    calculator.from_sharegpt(data_path)
    print(f"PSR: {calculator.get_psr()}")

                