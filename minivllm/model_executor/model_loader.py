
import torch
import torch.nn as nn

from minivllm.configs import ModelConfig
from minivllm.model_executor.models.registry import MODEL_REGISTRY, ModelClass
from minivllm.model_executor.weight_utils import initialize_dummy_weights
from minivllm.model_executor.layers.fp8 import resolve_fp8_config
from minivllm.utils import counter


def _get_model_architecture(model_config: ModelConfig) -> ModelClass:
    """Resolve the model class from the normalized root architectures."""
    return MODEL_REGISTRY.resolve_model_class(
        model_config.architecture.architectures
    )


def get_model(model_config: ModelConfig) -> nn.Module:
    model_class = _get_model_architecture(model_config)
    fp8_config = resolve_fp8_config(
        model_config.architecture.root_config,
        model_config.architecture.text_config,
    )
    if fp8_config is not None and model_config.use_dummy_weights:
        raise ValueError("FP8 profiling requires checkpoint weights, not dummy weights")
    torch.set_default_dtype(model_config.dtype)

    # Create a model instance.
    # The weights will be initialized as empty tensors.
    model = model_class(model_config.architecture.text_config)
    if fp8_config is not None:
        if not hasattr(model, "configure_fp8"):
            raise ValueError("This model does not support block FP8 weights")
        model.configure_fp8(fp8_config)
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
