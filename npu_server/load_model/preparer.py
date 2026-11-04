from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from optimum.commands.optimum_cli import main as optimum_cli_main
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from npu_server.cli.model_config_cli import (
    MANIFEST_FILENAME,
    ModelConfig,
)


class OpenvinoModelPreparer:
    """Download (if needed) and export a checkpoint into an OpenVINO IR model directory."""

    CHECKPOINT_DIRNAME = "checkpoint"
    EXPORTED_DIRNAME = "exported"
    CACHE_DIRNAME = ".openvino_npu_cache"

    @classmethod
    def model_folder_name(cls, config: ModelConfig) -> str:
        """Per-model home directory under model_dir, derived from model_name."""
        safe_name = re.sub(r'[<>:"/\\|?*\s]+', "-", config.model_name or "model").strip("-.")
        return safe_name or "model"

    @classmethod
    def model_home(cls, config: ModelConfig) -> Path:
        return config.model_dir / cls.model_folder_name(config)

    @classmethod
    def checkpoint_dir(cls, config: ModelConfig) -> Path:
        """Local copy of the source checkpoint, kept inside the model home."""
        return cls.model_home(config) / cls.CHECKPOINT_DIRNAME

    @classmethod
    def exported_dir(cls, config: ModelConfig) -> Path:
        """Directory holding the exported OpenVINO IR, tokenizer and manifest."""
        return cls.model_home(config) / cls.EXPORTED_DIRNAME

    @staticmethod
    def has_openvino_ir(exported_dir: Path) -> bool:
        if not exported_dir.is_dir():
            return False
        return (exported_dir / "openvino_model.xml").is_file() and (exported_dir / "openvino_model.bin").is_file()

    @classmethod
    def _resolve_hf_model(cls, model_id: str, checkpoint_dir: Path) -> Path:
        """Download a Hub repo into checkpoint_dir, or use the provided local folder."""
        local_path = Path(model_id).expanduser()
        if local_path.exists() and local_path.is_dir():
            return local_path.resolve()

        print(f"[download] Downloading Hugging Face checkpoint into: {checkpoint_dir}", flush=True)
        snapshot_download(repo_id=model_id, local_dir=checkpoint_dir)
        return checkpoint_dir

    @classmethod
    def _migrate_legacy_layout(cls, config: ModelConfig) -> None:
        """Move a pre-0.6 flat model_dir layout into model_home/checkpoint + exported."""
        model_root = config.model_dir
        model_home = cls.model_home(config)
        if model_home.exists():
            return
        legacy_files = model_root.exists() and any(item.is_file() for item in model_root.iterdir())
        legacy_checkpoint = (model_root / cls.CHECKPOINT_DIRNAME).is_dir()
        if not legacy_files and not legacy_checkpoint:
            return

        print(f"[migrate] Restructuring legacy layout: {model_root} -> {model_home}", flush=True)
        exported_dir = model_home / cls.EXPORTED_DIRNAME
        exported_dir.mkdir(parents=True, exist_ok=True)
        for item in model_root.iterdir():
            if item == model_home or item.name == cls.CACHE_DIRNAME:
                continue
            if item.is_dir():
                if item.name == cls.CHECKPOINT_DIRNAME:
                    continue
                shutil.move(str(item), str(exported_dir / item.name))
            else:
                shutil.move(str(item), str(exported_dir / item.name))
        if legacy_checkpoint:
            shutil.move(str(model_root / cls.CHECKPOINT_DIRNAME), str(model_home / cls.CHECKPOINT_DIRNAME))

    @classmethod
    def _clear_exported(cls, exported_dir: Path) -> None:
        """Delete the exported IR directory but keep the source checkpoint sibling."""
        resolved = exported_dir.resolve()
        if exported_dir.name != cls.EXPORTED_DIRNAME or len(resolved.parts) < 4:
            raise RuntimeError(f"Refusing to remove unsafe exported directory: {resolved}")
        shutil.rmtree(resolved)

    @staticmethod
    def _load_tokenizer(source_dir: Path) -> PreTrainedTokenizerBase:
        print("[tokenizer] Loading tokenizer from the downloaded checkpoint ...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(str(source_dir), trust_remote_code=True)
        if tokenizer is None:
            raise RuntimeError(f"Tokenizer could not be loaded from '{source_dir}'")
        return tokenizer

    @staticmethod
    def _export_ir(source_dir: Path, exported_dir: Path) -> None:
        export_args = [
            "export",
            "openvino",
            "--model",
            str(source_dir),
            "--task",
            "text-generation-with-past",
            "--trust-remote-code",
            "--weight-format",
            "int4",
            "--sym",
            "--group-size",
            "128",
            "--ratio",
            "1.0",
            str(exported_dir),
        ]

        print("[export] Exporting checkpoint to OpenVINO INT4 IR:", flush=True)
        print("optimum-cli " + " ".join(export_args), flush=True)

        previous_argv = sys.argv
        try:
            sys.argv = ["optimum-cli", *export_args]
            optimum_cli_main()
        finally:
            sys.argv = previous_argv

    @staticmethod
    def _unify_attention_layers(source_dir: Path) -> bool:
        """Replace sliding_attention layer types with full_attention to satisfy NPUW."""
        config_path = source_dir / "config.json"
        if not config_path.is_file():
            return False
        checkpoint_cfg = json.loads(config_path.read_text(encoding="utf-8"))
        layer_types = checkpoint_cfg.get("layer_types")
        if not isinstance(layer_types, list) or "sliding_attention" not in layer_types:
            return False
        checkpoint_cfg["layer_types"] = [
            "full_attention" if layer_type == "sliding_attention" else layer_type for layer_type in layer_types
        ]
        rope_parameters = checkpoint_cfg.get("rope_parameters")
        if isinstance(rope_parameters, dict) and "full_attention" in rope_parameters:
            checkpoint_cfg["rope_parameters"] = {"full_attention": rope_parameters["full_attention"]}
        checkpoint_cfg.pop("sliding_window", None)
        config_path.write_text(json.dumps(checkpoint_cfg, indent=2), encoding="utf-8")
        return True

    @classmethod
    def ensure(cls, config: ModelConfig, *, force: bool = False) -> Path:
        """Make sure the model home contains a complete IR pair and a tokenizer."""
        cls._migrate_legacy_layout(config)
        checkpoint_dir = cls.checkpoint_dir(config)
        exported_dir = cls.exported_dir(config)

        if cls.has_openvino_ir(exported_dir) and not force:
            print(f"[skip] OpenVINO IR already exists: {exported_dir}", flush=True)
            return exported_dir

        if exported_dir.exists() and any(exported_dir.iterdir()):
            if not force:
                raise RuntimeError(
                    f"Directory exists but does not contain a complete OpenVINO IR: {exported_dir}. "
                    "Use --force-export to recreate it."
                )
            cls._clear_exported(exported_dir)

        source_dir = cls._resolve_hf_model(config.model_id, checkpoint_dir)
        if config.full_attention_only:
            if cls._unify_attention_layers(source_dir):
                print("[config] sliding_attention layers converted to full_attention (full_attention_only)", flush=True)
            else:
                print("[config] full_attention_only enabled, but no sliding_attention layers found", flush=True)
        tokenizer = cls._load_tokenizer(source_dir)
        cls._export_ir(source_dir, exported_dir)

        if not cls.has_openvino_ir(exported_dir):
            raise RuntimeError(
                "Optimum Intel finished without producing an OpenVINO IR pair (.xml + .bin). "
                f"Inspect the export log and directory: {exported_dir}"
            )

        print("[tokenizer] Saving Hugging Face tokenizer alongside the OpenVINO IR ...", flush=True)
        tokenizer.save_pretrained(exported_dir)

        manifest = {
            "model_id": config.model_id,
            "model_name": config.model_name,
            "device": config.device,
            "format": "OpenVINO IR",
            "weight_format": "int4",
            "symmetric": True,
            "group_size": 128,
            "full_attention_only": config.full_attention_only,
        }
        (exported_dir / MANIFEST_FILENAME).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        print(f"[done] OpenVINO IR and tokenizer saved to: {exported_dir}", flush=True)
        return exported_dir
