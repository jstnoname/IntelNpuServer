from __future__ import annotations

import argparse
import sys
from typing import Sequence

import openvino as ov
import openvino_genai as ov_genai

from npu_server.cli.arguments_cli import ModelArgumentsParser
from npu_server.load_model import OpenvinoModelPreparer
from npu_server.server import NpuServer


class CommandLineInterface:
    """Dispatch the npu-server commands provided on the command line."""

    def __init__(self) -> None:
        self.command_parser = self._build_command_parser()

    @staticmethod
    def _build_command_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            prog="npu-server",
            description="Download, export and serve OpenVINO GenAI models (primarily on Intel NPU)",
        )
        parser.add_argument(
            "command",
            nargs="?",
            choices=("run", "prepare", "serve", "doctor"),
            default="run",
            help="run: prepare if needed and start API; prepare: export only; "
            "serve: load existing IR; doctor: check environment",
        )
        parser.add_argument(
            "--force-export",
            action="store_true",
            help="re-export the model even if a valid OpenVINO IR already exists",
        )
        return parser

    @staticmethod
    def _inspect_environment() -> int:
        try:
            print(f"OpenVINO: {ov.__version__}")
            print(f"OpenVINO GenAI: {getattr(ov_genai, 'version', 'installed')}")
            devices = list(ov.Core().available_devices)
            print(f"Available devices: {devices}")
            if not any(device == "NPU" or device.startswith("NPU.") for device in devices):
                print("WARNING: OpenVINO did not detect an NPU. Check the Intel NPU driver.")
            return 0
        except Exception as exc:
            print(f"Environment check failed: {exc}", file=sys.stderr)
            return 1

    def startup(self, argv: Sequence[str] | None = None) -> int:
        command_args, config_args = self.command_parser.parse_known_args(argv)

        try:
            config = ModelArgumentsParser.get_model_params(config_args)

            if command_args.command == "doctor":
                return self._inspect_environment()

            if command_args.command == "prepare":
                OpenvinoModelPreparer().ensure(config, force=command_args.force_export)
                return 0

            if command_args.command == "serve":
                NpuServer(config, prepare_if_missing=False).run()
            else:
                NpuServer(
                    config,
                    prepare_if_missing=True,
                    force_export=command_args.force_export,
                ).run()
            return 0

        except KeyboardInterrupt:
            print("\nInterrupted by user.")
            return 130
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
