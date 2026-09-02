
import torch
import torch.nn as nn
from typing import Optional
from transformers import AutoConfig

from minivllm.configs import ModelConfig
from minivllm.model_executor.models.registry import MODEL_REGISTRY, ModelClass
from minivllm.model_executor.weight_utils import initialize_dummy_weights
from minivllm.utils import counter


def _get_model_architecture(model_config: ModelConfig) -> ModelClass:
    """Resolve the model class from the normalized root architectures."""
    return MODEL_REGISTRY.resolve_model_class(
        model_config.architecture.architectures
    )


def get_model(model_config: ModelConfig) -> nn.Module:
    model_class = _get_model_architecture(model_config)
    torch.set_default_dtype(model_config.dtype)

    # Create a model instance.
    # The weights will be initialized as empty tensors.
    model = model_class(model_config.architecture.text_config)
    if model_config.use_dummy_weights:
        model = model.cuda()
        # Set random value to the weights.
        initialize_dummy_weights(model)
    else:
        # Load the weights from the cached or downloaded files.
        model.load_weights(
            model_config.model,
            model_config.download_dir,
            model_config.use_np_weights
        )
        model = model.cuda()
    return model.eval()


def get_dspark_model(
    model_name_or_path: str,
    *,
    dtype: torch.dtype,
    cache_dir: Optional[str] = None,
    use_dummy_weights: bool = False,
) -> nn.Module:
    """Load the standalone DSpark weights without duplicating target modules."""
    from minivllm.model_executor.models.dspark import DSparkDraftModel

    config = AutoConfig.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )
    torch.set_default_dtype(dtype)
    model = DSparkDraftModel(config)
    if use_dummy_weights:
        model = model.cuda()
        initialize_dummy_weights(model)
    else:
        model.load_weights(model_name_or_path, cache_dir)
        model = model.cuda()
    return model.eval()
