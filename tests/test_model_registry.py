import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

from minivllm.model_executor import model_loader
from minivllm.model_executor.models import registry as registry_module
from minivllm.model_executor.models.registry import (
    MODEL_REGISTRY,
    ModelRegistry,
)


class TinyModel(nn.Module):
    pass


class AlternateModel(nn.Module):
    pass


class NotAModel:
    pass


class ModelRegistryTest(unittest.TestCase):

    def test_builtin_architectures_are_listed_without_importing_targets(self):
        with patch.object(registry_module, "import_module") as import_mock:
            supported = MODEL_REGISTRY.supported_architectures

        self.assertIn("LlamaForCausalLM", supported)
        self.assertIn("Qwen3_5ForConditionalGeneration", supported)
        import_mock.assert_not_called()

    def test_register_and_resolve_eager_model_class(self):
        registry = ModelRegistry()
        registry.register_model("TinyForCausalLM", TinyModel)

        resolved = registry.resolve_model_class(["TinyForCausalLM"])
        self.assertIs(resolved, TinyModel)

    def test_duplicate_registration_requires_explicit_overwrite(self):
        registry = ModelRegistry({"TinyForCausalLM": TinyModel})

        with self.assertRaisesRegex(ValueError, "TinyForCausalLM"):
            registry.register_model("TinyForCausalLM", AlternateModel)

        registry.register_model(
            "TinyForCausalLM",
            AlternateModel,
            overwrite=True,
        )
        self.assertIs(
            registry.resolve_model_class(["TinyForCausalLM"]),
            AlternateModel,
        )

    def test_invalid_registration_inputs_are_rejected(self):
        registry = ModelRegistry()

        with self.assertRaisesRegex(ValueError, "architecture"):
            registry.register_model("  ", TinyModel)
        with self.assertRaisesRegex(ValueError, "module:ClassName"):
            registry.register_model("BrokenTarget", "missing_separator")
        with self.assertRaisesRegex(TypeError, "nn.Module"):
            registry.register_model("NotAModel", NotAModel)

    def test_lazy_target_is_imported_only_during_resolution(self):
        registry = ModelRegistry(
            {"LazyForCausalLM": "fake_models.lazy:TinyModel"}
        )
        fake_module = SimpleNamespace(TinyModel=TinyModel)

        with patch.object(
            registry_module,
            "import_module",
            return_value=fake_module,
        ) as import_mock:
            self.assertEqual(
                registry.supported_architectures,
                ("LazyForCausalLM",),
            )
            import_mock.assert_not_called()

            resolved = registry.resolve_model_class(["LazyForCausalLM"])

        self.assertIs(resolved, TinyModel)
        import_mock.assert_called_once_with("fake_models.lazy")

    def test_registering_lazy_target_does_not_import_it(self):
        registry = ModelRegistry()

        with patch.object(registry_module, "import_module") as import_mock:
            registry.register_model(
                "LazyForCausalLM",
                "fake_models.lazy:TinyModel",
            )

        self.assertEqual(
            registry.supported_architectures,
            ("LazyForCausalLM",),
        )
        import_mock.assert_not_called()

    def test_lazy_target_must_resolve_to_module_subclass(self):
        registry = ModelRegistry(
            {"BrokenForCausalLM": "fake_models.lazy:NotAModel"}
        )
        fake_module = SimpleNamespace(NotAModel=NotAModel)

        with patch.object(
            registry_module,
            "import_module",
            return_value=fake_module,
        ):
            with self.assertRaisesRegex(TypeError, "nn.Module"):
                registry.resolve_model_class(["BrokenForCausalLM"])

    def test_resolution_preserves_hugging_face_order(self):
        registry = ModelRegistry(
            {
                "FirstForCausalLM": TinyModel,
                "SecondForCausalLM": AlternateModel,
            }
        )

        resolved = registry.resolve_model_class(
            ["SecondForCausalLM", "FirstForCausalLM"]
        )
        self.assertIs(resolved, AlternateModel)

    def test_unknown_architectures_report_requested_and_supported_names(self):
        registry = ModelRegistry({"TinyForCausalLM": TinyModel})

        with self.assertRaisesRegex(ValueError, "UnknownForCausalLM") as ctx:
            registry.resolve_model_class(["UnknownForCausalLM"])

        self.assertIn("TinyForCausalLM", str(ctx.exception))

    def test_lazy_import_error_includes_architecture_and_target(self):
        target = "missing.models:MissingModel"
        registry = ModelRegistry({"MissingForCausalLM": target})

        with patch.object(
            registry_module,
            "import_module",
            side_effect=ModuleNotFoundError("missing.models"),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "MissingForCausalLM",
            ) as ctx:
                registry.resolve_model_class(["MissingForCausalLM"])

        self.assertIn(target, str(ctx.exception))
        self.assertIsInstance(ctx.exception.__cause__, ModuleNotFoundError)


class ModelLoaderIntegrationTest(unittest.TestCase):

    def test_loader_selects_from_root_and_constructs_with_text_config(self):
        root_config = SimpleNamespace(name="root")
        text_config = SimpleNamespace(name="text")
        architecture = SimpleNamespace(
            root_config=root_config,
            text_config=text_config,
            architectures=("Qwen3_5ForConditionalGeneration",),
        )
        model_config = SimpleNamespace(
            architecture=architecture,
            dtype=torch.float32,
            use_dummy_weights=True,
            model="unused",
            download_dir=None,
            use_np_weights=False,
        )

        class RecordingModel(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.received_config = config

            def cuda(self):
                return self

        with patch.object(
            model_loader.MODEL_REGISTRY,
            "resolve_model_class",
            return_value=RecordingModel,
        ) as resolve_mock, patch.object(
            model_loader,
            "initialize_dummy_weights",
        ), patch.object(
            model_loader.torch,
            "set_default_dtype",
        ):
            model = model_loader.get_model(model_config)

        resolve_mock.assert_called_once_with(architecture.architectures)
        self.assertIs(model.received_config, text_config)
        self.assertFalse(model.training)


if __name__ == "__main__":
    unittest.main()
