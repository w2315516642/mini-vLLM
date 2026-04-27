import torch
from pathlib import Path
from typing import Tuple, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict
from minivllm.utils import read_yaml
import json
import yaml


class ModelConfig(BaseModel):
    vocab_size: int = Field(default=10000, alias="vocab_size")
    context_length: int = Field(default=256, alias="context_length")
    num_layers: int = Field(default=4, alias="num_layers")
    d_model: int = Field(default=512, alias="d_model")
    num_heads: int = Field(default=16, alias="num_heads")
    d_ff: int = Field(default=1344, alias="d_ff")
    theta: float = Field(default=10000.0, alias="theta")


class OptimizerConfig(BaseModel):
    # 优化器（AdamW）
    max_lr: float = Field(default=1e-3)
    min_lr: float = Field(default=1e-5)
    betas: Tuple[float, float] = Field(default=(0.9, 0.95))
    weight_decay: float = Field(default=1e-5)
    eps: float = Field(default=1e-3)


class TrainConfig(BaseModel):
    batch_size: int = Field(default=64)
    # 余弦退火
    total_train_iters: int = Field(default=100, alias="total_iters")
    warmup_iters: int = Field(default=20)
    cosine_iters: int = Field(default=70)
    
    max_l2_norm: float = Field(default=1.0)

    ckpt_interval: int = Field(default=10, alias="checkpoint_interval")
    max_ckpt_to_keep: int = Field(default=3)
    output_dir: str = Field(default="./output/")
    train_data: str = Field(default="./data/owt_small.npy")
    valid_data: str = Field(default="./data/owt_small.npy")
    log_interval: int = Field(default=10)
    valid_interval: int = Field(default=10)
    valid_iters: int = Field(default=20)


class TestConfig(BaseModel):
    model: ModelConfig = Field(default=None, alias="model_config")
    optim: OptimizerConfig = Field(default=None, alias="optim_config")
    train: TrainConfig = Field(default=None, alias="train_config")

    is_checkpoint: bool = Field(default=False, alias="is_checkpoint")
    checkpoint_path: str = Field(default="", alias="checkpoint_path")
    device: torch.device = Field(default=torch.device("cuda"), alias="device")
    dtype: torch.dtype = Field(default=torch.float16, alias="dtype")

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        populate_by_name=True
    )

    @classmethod
    def from_yaml(cls, file_path: str):
        config_dict = read_yaml(file_path)
        return cls(**config_dict)

    def to_dict(self):
        config_dict = self.model_dump(by_alias=True)
        config_dict["dtype"] = str(self.dtype)
        config_dict["device"] = str(self.device)

    def to_yaml(self, file_path: str):
        config_dict = self.to_dict()
        yaml.dump(config_dict, open(file_path, "w"), default_flow_style=False, allow_unicode=True)


    @field_validator("device", "dtype", mode="before")
    @classmethod
    def check_dtype(cls, v: Any, info: Any) -> torch.device | torch.dtype:
        if info.field_name == "device":
            if isinstance(v, str):
                device = v.replace("torch.", "")
                if "cuda" in device and not torch.cuda.is_available():
                    print(f"Cuda is not available.")
                    return torch.device("cpu")
                return torch.device(device)
        if info.field_name == "dtype":
            if isinstance(v, str):
                dtype = v.replace("torch.", "")
                if hasattr(torch, dtype):
                    return getattr(torch, dtype)
                print(f"Invaild torch.dtype: {dtype}, replaced by torch.float32")
                return torch.float32
        return v

    def __str__(self) -> str:
        config_dict = self.model_dump(by_alias=True)
        config_dict["dtype"] = str(self.dtype)
        config_dict["device"] = str(self.device)
        return f"Configuration:\n{json.dumps(config_dict, indent=4)}"


if __name__ == "__main__":
    yaml_path = Path(__file__).parent.parent / "config.yaml"
    config = Config.from_yaml(yaml_path)
    print(config)