import torch
from typing import Dict, Any
from pathlib import Path
import re

def top_p_filter(logits: torch.Tensor, top_p: float | None = None) -> torch.Tensor:
    if not top_p:
        return logits
    
    assert top_p >= 0 and top_p <= 1, f"Invaild top-p value: {top_p}"

    # 先排序
    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
    # 为了算概率要进行softmax
    sorted_probs = softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    # 将累计概率大于p的下标移除
    sorted_indices_to_remove = cumulative_probs > top_p
    # 保留使得累计概率大于p的那个下标
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = sorted_indices_to_remove.scatter(
        -1, sorted_indices, sorted_indices_to_remove
    )
    logits[indices_to_remove] = float("-inf")

    return logits


def softmax(x: torch.Tensor, dim: int=-1) -> torch.Tensor:
    in_dtype = x.dtype
    x.to(torch.float32)
    max_val = torch.max(x, dim=dim, keepdim=True).values
    x = x - max_val
    x = torch.exp(x)
    sum_val = torch.sum(x, dim=dim, keepdim=True)
    o = x / sum_val
    return o.to(in_dtype)

def read_yaml(file_path: str) -> Dict[str, Any]:
    if not Path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} is not exists.")
    # 读取 yaml 为 str
    content = load_yaml_stably(file_path)
    if not content:
        raise IOError(f"Failed to read configuration file {file_path}")
    
    # 替换环境变量
    pattern = re.compile(r"\$\{(\w+)\}")
    def replacer(match: re.Match):
        env_var = match.group(1)
        return os.getenv(env_var, match.group(0))
    content = pattern.sub(replacer, content)

    # 读取文件
    try:
        return yaml.safe_load(content)
    except yaml.YAMLError as e:
        print(f"Error in reading yaml file: {e}")
        raise e


def load_yaml_stably(file_path: str) -> str | None:

    encodings = ["utf-8", "utf-8-sig", "gbk", "gb2312", "ascii", "cp936"]
    for encoding in encodings:
        try:
            with open(file_path, "r", encoding=encoding) as f:
                print(f"Successfully read file {file_path} with encoding {encoding}")
                return f.read()
        except UnicodeDecodeError:
            continue
    # 尝试二进制
    try:
        with open(file_path, "rb") as f:
            raw_data = f.read()
        detect = chardet.detect(raw_data)
        if detect["encoding"]:
            return raw_data.decode(detect["encoding"])
    except Exception as e:
        print(f"Can not read file {file_path} with right encoding: {e}")
    return None