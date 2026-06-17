import logging
import os
from pathlib import Path
from typing import Any

from cli.lib.common.cli_helper import BaseRunner
from cli.lib.common.pip_helper import pip_install_packages
from cli.lib.common.utils import working_directory
from cli.lib.core.torchtitan.lib import (
    clone_torchtitan,
    load_torchtitan_test_library,
    run_test_plan,
)


logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[6]


def _pinned_commit(name: str) -> str:
    pin_path = _REPO_ROOT / ".github" / "ci_commit_pins" / f"{name}.txt"
    return pin_path.read_text().strip()


class TorchtitanTestRunner(BaseRunner):
    def __init__(self, args: Any):
        self.work_directory = "torchtitan"
        self.test_plan = args.test_plan

    def prepare(self):
        clone_torchtitan(dst=self.work_directory)
        torchao_url = (
            f"git+https://github.com/pytorch/ao.git@{_pinned_commit('torchao')}"
        )
        torchcomms_url = (
            "git+https://github.com/meta-pytorch/torchcomms.git"
            f"@{_pinned_commit('torchcomms')}"
        )
        # Build against the PyTorch wheel under test; prebuilt nightlies can
        # otherwise pull in or link against a different torch nightly.
        pip_install_packages(
            packages=[
                "--no-build-isolation",
                "--no-deps",
                torchao_url,
            ],
        )
        pip_install_packages(
            packages=[
                "--no-build-isolation",
                "--no-deps",
                torchcomms_url,
            ],
            env={
                "USE_GLOO": "1",
                "USE_NCCLX": "0",
                "USE_TRANSPORT": "0",
                "USE_NCCL": "1"
                if "cuda" in os.environ.get("BUILD_ENVIRONMENT", "")
                else "0",
            },
        )
        with working_directory(self.work_directory):
            pip_install_packages(packages=["-e", "."])
            pip_install_packages(packages=["pytest", "pytest-cov"])

    def run(self):
        self.prepare()
        with working_directory(self.work_directory):
            run_test_plan(self.test_plan, load_torchtitan_test_library())
