"""Lazy model registry used by the mini-vLLM model loader.

Hugging Face configs identify the top-level implementation through their
``architectures`` list. The registry keeps that external name separate from
the Python import target so model modules, CUDA extensions, and optional
dependencies are imported only when a model is actually selected.

Resolution rules
----------------
Resolution is exact: model families are not inferred from repository names,
and an unknown architecture never silently falls back to Llama.
"""

from dataclasses import dataclass
from importlib import import_module
from typing import Dict, Iterable, Mapping, Optional, Tuple, Type, Union

import torch.nn as nn


ModelClass = Type[nn.Module]
ModelTarget = Union[ModelClass, str]


@dataclass(frozen=True)
class ModelRegistration:
    """One Hugging Face architecture mapped to an eager or lazy target."""

    architecture: str
    target: ModelTarget


class ModelRegistry:
    """Resolve ordered Hugging Face architecture names to model classes.

    A target may be an already imported ``nn.Module`` subclass or a lazy
    ``"package.module:ClassName"`` string. Lazy targets must not be imported by
    registration or by ``supported_architectures`` inspection.
    """

    def __init__(
        self,
        registrations: Optional[Mapping[str, ModelTarget]] = None,
    ) -> None:
        # Built-ins are trusted constants so this constructor stays import-safe
        # while register_model remains learner-owned.
        self._registrations: Dict[str, ModelRegistration] = {
            architecture: ModelRegistration(architecture, target)
            for architecture, target in (registrations or {}).items()
        }

    @property
    def supported_architectures(self) -> Tuple[str, ...]:
        """Return registered architecture names without importing targets."""
        return tuple(self._registrations)

    def register_model(
        self,
        architecture: str,
        target: ModelTarget,
        *,
        overwrite: bool = False,
    ) -> None:
        """Register one architecture.

        Reject empty names, malformed lazy targets, non-model classes, and
        duplicate registrations unless ``overwrite`` is explicitly requested.
        """
        self._validate_architecture(architecture)
        self._validate_target(target)

        duplicated = architecture in self._registrations
        if duplicated and not overwrite:
            raise ValueError(
                f"Model architecture {architecture!r} is already registered"
            )
        self._registrations[architecture] = ModelRegistration(
            architecture,
            target,
        )

    def resolve_model_class(
        self,
        architectures: Iterable[str],
    ) -> ModelClass:
        """Resolve the first registered architecture in the given order.

        Raise a useful ``ValueError`` when none match. Import or type failures
        for a matched registration should retain the original exception as the
        cause and include both the architecture and target in the message.
        """
        requested_architectures = tuple(architectures)
        valid = False
        for architecture in requested_architectures:
            try:
                self._validate_architecture(architecture)
            except ValueError:
                continue

            valid = True
            registration = self._registrations.get(architecture)
            if registration is None:
                continue
            return self._load_model_class(registration)

        if not valid:
            raise ValueError(
                "No valid architecture names were provided: "
                f"{requested_architectures!r}"
            )
        raise ValueError(
            f"Unsupported architectures {requested_architectures!r}; "
            f"supported architectures are {self.supported_architectures!r}"
        )

    def _load_model_class(
        self,
        registration: ModelRegistration,
    ) -> ModelClass:
        """Load and validate one eager or lazy registration target."""
        if isinstance(registration.target, str):
            module_name, class_name = registration.target.split(":")
            try:
                module = import_module(module_name)
                model_class = getattr(module, class_name)
            except (ImportError, AttributeError) as e:
                raise RuntimeError(
                    "Failed to load model architecture "
                    f"{registration.architecture!r} from target "
                    f"{registration.target!r}"
                ) from e
            if (
                not isinstance(model_class, type)
                or not issubclass(model_class, nn.Module)
            ):
                raise TypeError(
                    f"Target {registration.target!r} for architecture "
                    f"{registration.architecture!r} must resolve to an "
                    "nn.Module subclass"
                )
            return model_class
        return registration.target

    @staticmethod
    def _validate_architecture(architecture: str) -> None:
        """Validate one external Hugging Face architecture name."""
        if not isinstance(architecture, str):
            raise ValueError(
                f"Model architecture must be a non-empty string: {architecture!r}"
            )
        if not architecture.strip():
            raise ValueError(
                f"Model architecture must be a non-empty string: {architecture!r}"
            )

    @staticmethod
    def _validate_target(target: ModelTarget) -> None:
        """Validate an eager model class or lazy ``module:class`` target."""
        if isinstance(target, str):
            name_list = target.split(":")
            if len(name_list) != 2 or any(not name.strip() for name in name_list):
                raise ValueError(
                    "Lazy model target must use the "
                    f"'module:ClassName' format: {target!r}"
                )
        elif not isinstance(target, type) or not issubclass(target, nn.Module):
            raise TypeError(
                f"Eager model target must be an nn.Module subclass: {target!r}"
            )


_BUILTIN_MODEL_TARGETS: Mapping[str, ModelTarget] = {
    "LlamaForCausalLM": (
        "minivllm.model_executor.models.llama:LlamaForCausalLM"
    ),
    # Qwen3.8 checkpoints retain the Qwen3.5 schema and architecture name.
    # The full model target follows after its hybrid decoder is implemented.
    "Qwen3_5ForConditionalGeneration": (
        "minivllm.model_executor.models.qwen3_5:"
        "Qwen3_5ForConditionalGeneration"
    ),
}


MODEL_REGISTRY = ModelRegistry(_BUILTIN_MODEL_TARGETS)


__all__ = [
    "MODEL_REGISTRY",
    "ModelClass",
    "ModelRegistration",
    "ModelRegistry",
    "ModelTarget",
]
