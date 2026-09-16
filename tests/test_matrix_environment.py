import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


@unittest.skipUnless(os.name == "posix" and shutil.which("bash"), "Linux shell launcher")
class EnvironmentTest(unittest.TestCase):
    def test_upstream_activation_preserves_child_mini_environment(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            conda = directory / "conda.sh"
            conda.write_text('conda() { export TEST_ACTIVE_ENV="$2"; }\n')
            python = directory / "python"
            python.write_text('#!/bin/bash\nprintf "%s|%s" "$TEST_ACTIVE_ENV" "$CONDA_ENV"\n')
            python.chmod(0o755)
            for configured in (None, "vllm", "/custom/mini-env"):
                with self.subTest(configured=configured):
                    env = dict(os.environ, CONDA_SH=str(conda), UPSTREAM_ENV="/custom/upstream",
                               PATH=tmp + os.pathsep + os.environ["PATH"])
                    env.pop("CONDA_ENV", None)
                    if configured is not None:
                        env["CONDA_ENV"] = configured
                    result = subprocess.run(["bash", "scripts/autodl/benchmark_matrix.sh"],
                                            cwd=root, env=env, check=True, capture_output=True, text=True)
                    self.assertEqual(result.stdout, "/custom/upstream|" + (configured or "vllm"))


if __name__ == "__main__":
    unittest.main()
