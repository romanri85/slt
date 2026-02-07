import os
import logging
import toml

from .model_registry import ModelRegistry

logger = logging.getLogger(__name__)

# Keys that can be overridden from training_params
TRAINING_OVERRIDES = {
    "epochs": "epochs",
    "learning_rate": "optimizer.lr",
    "lora_rank": "adapter.rank",
    "save_every_n_epochs": "save_every_n_epochs",
    "resolution": None,  # handled separately in dataset.toml
    "micro_batch_size_per_gpu": "micro_batch_size_per_gpu",
    "gradient_accumulation_steps": "gradient_accumulation_steps",
    "gradient_clipping": "gradient_clipping",
    "warmup_steps": "warmup_steps",
}


def build_configs(model_type, network_volume, work_dir, dataset_info, training_params=None):
    """
    Build training TOML + dataset TOML from template and job params.

    Returns dict with paths to generated config files:
      {"training_toml": ..., "dataset_toml": ..., "eval_dataset_toml": ...}
    """
    training_params = training_params or {}
    registry = ModelRegistry(network_volume)
    config = registry.get_config(model_type)

    # Read the TOML template
    template_name = config["toml_template"]
    template_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "toml_templates", template_name)

    logger.info(f"Loading TOML template: {template_name}")
    with open(template_path, "r") as f:
        toml_content = f.read()

    # Apply network volume path replacements
    for old, new in config["path_replacements"].items():
        replacement = new.replace("{nv}", network_volume)
        toml_content = toml_content.replace(old, replacement)

    # Fix output_dir to use work_dir
    output_dir = os.path.join(work_dir, "training_outputs", f"{model_type}_lora")
    os.makedirs(output_dir, exist_ok=True)

    # Parse the TOML
    training_config = toml.loads(toml_content)

    # Set output dir
    training_config["output_dir"] = output_dir

    # Point to our generated dataset configs
    dataset_toml_path = os.path.join(work_dir, "dataset.toml")
    training_config["dataset"] = dataset_toml_path

    # Set up eval dataset if available
    eval_dataset_toml_path = os.path.join(work_dir, "eval_dataset.toml")
    if dataset_info.get("has_eval"):
        training_config["eval_datasets"] = [
            {"name": "eval", "config": eval_dataset_toml_path}
        ]
    else:
        # Remove eval settings if no eval data
        training_config.pop("eval_datasets", None)
        training_config.pop("eval_every_n_epochs", None)
        training_config.pop("eval_before_first_step", None)
        training_config.pop("eval_micro_batch_size_per_gpu", None)
        training_config.pop("eval_gradient_accumulation_steps", None)

    # Apply training param overrides
    _apply_overrides(training_config, training_params)

    # Write training TOML
    training_toml_path = os.path.join(work_dir, "training_config.toml")
    with open(training_toml_path, "w") as f:
        toml.dump(training_config, f)
    logger.info(f"Training config written to {training_toml_path}")

    # Build dataset TOML
    _build_dataset_toml(
        dataset_toml_path, dataset_info, training_params, config["supports_video"]
    )

    # Build eval dataset TOML if eval data exists
    if dataset_info.get("has_eval"):
        _build_eval_dataset_toml(eval_dataset_toml_path, dataset_info, training_params)

    return {
        "training_toml": training_toml_path,
        "dataset_toml": dataset_toml_path,
        "eval_dataset_toml": eval_dataset_toml_path if dataset_info.get("has_eval") else None,
        "output_dir": output_dir,
    }


def _apply_overrides(config, params):
    """Apply user training_params overrides to the parsed TOML config."""
    for param_key, toml_path in TRAINING_OVERRIDES.items():
        if param_key not in params or toml_path is None:
            continue

        value = params[param_key]
        parts = toml_path.split(".")

        if len(parts) == 1:
            config[parts[0]] = value
        elif len(parts) == 2:
            section, key = parts
            if section not in config:
                config[section] = {}
            config[section][key] = value

        logger.info(f"Override: {toml_path} = {value}")


def _build_dataset_toml(path, dataset_info, training_params, supports_video):
    """Build the dataset.toml matching diffusion-pipe's expected format.

    Format: top-level resolution/AR settings + [[directory]] sections.
    """
    resolution = training_params.get("resolution", 1024)

    # Top-level settings (matching diffusion-pipe examples/dataset.toml)
    lines = []
    lines.append(f"resolutions = [{resolution}]")
    lines.append("")
    lines.append("enable_ar_bucket = true")
    lines.append("min_ar = 0.5")
    lines.append("max_ar = 2.0")
    lines.append("num_ar_buckets = 7")

    # Add frame_buckets for video-capable models
    if supports_video and dataset_info.get("has_videos"):
        lines.append("frame_buckets = [1, 33]")

    lines.append("")

    # Image directory
    if dataset_info.get("has_images"):
        lines.append("[[directory]]")
        lines.append(f"path = '{dataset_info['images']}'")
        lines.append("num_repeats = 1")
        lines.append("")

    # Video directory (same [[directory]] section, diffusion-pipe auto-detects media type)
    if dataset_info.get("has_videos") and supports_video:
        lines.append("[[directory]]")
        lines.append(f"path = '{dataset_info['videos']}'")
        lines.append("num_repeats = 1")
        lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Dataset config written to {path}")


def _build_eval_dataset_toml(path, dataset_info, training_params=None):
    """Build the eval_dataset.toml matching diffusion-pipe's expected format."""
    training_params = training_params or {}
    resolution = training_params.get("resolution", 1024)
    lines = []
    lines.append(f"resolutions = [{resolution}]")
    lines.append("")
    lines.append("enable_ar_bucket = true")
    lines.append("min_ar = 0.5")
    lines.append("max_ar = 2.0")
    lines.append("num_ar_buckets = 7")
    lines.append("")
    lines.append("[[directory]]")
    lines.append(f"path = '{dataset_info['eval']}'")
    lines.append("num_repeats = 1")
    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Eval dataset config written to {path}")
