import os
import logging
from huggingface_hub import HfApi

logger = logging.getLogger(__name__)

MODEL_CARD_TEMPLATE = """---
tags:
- lora
- diffusion-pipe
- {model_type}
base_model: {base_model}
---

# {repo_id}

LoRA trained with [diffusion-pipe](https://github.com/tdrussell/diffusion-pipe) via serverless RunPod worker.

## Training Details

- **Model type**: {model_type}
- **Epochs**: {epochs}
- **LoRA rank**: {rank}
- **Learning rate**: {lr}
"""

BASE_MODEL_MAP = {
    "flux": "black-forest-labs/FLUX.1-dev",
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
    "wan13": "Wan-AI/Wan2.1-T2V-1.3B",
    "wan14b_t2v": "Wan-AI/Wan2.1-T2V-14B",
    "wan14b_i2v": "Wan-AI/Wan2.1-I2V-14B-480P",
    "qwen": "Qwen/Qwen-Image",
    "z_image_turbo": "z-image/z-image-turbo",
    "qwen_image_edit": "Qwen/Qwen-Image-Edit",
    "z_image_base": "z-image/z-image-base",
    "ltx_video": "Lightricks/LTX-Video",
}


def upload_lora(checkpoint_dir, hf_output_repo, hf_token, model_type, training_params=None, private=False):
    """
    Upload trained LoRA to HuggingFace Hub.

    Args:
        checkpoint_dir: Path to the epoch directory with trained LoRA files
        hf_output_repo: HF repo ID (e.g. 'username/my-lora')
        hf_token: HuggingFace token with write access
        model_type: Model type string (e.g. 'flux', 'sdxl')
        training_params: Optional dict of training params for model card
        private: Whether to create the repo as private

    Returns:
        URL of the uploaded repo
    """
    training_params = training_params or {}
    api = HfApi(token=hf_token)

    logger.info(f"Creating/checking repo: {hf_output_repo}")
    api.create_repo(repo_id=hf_output_repo, exist_ok=True, private=private)

    # Generate model card
    model_card = MODEL_CARD_TEMPLATE.format(
        model_type=model_type,
        base_model=BASE_MODEL_MAP.get(model_type, "unknown"),
        repo_id=hf_output_repo,
        epochs=training_params.get("epochs", "N/A"),
        rank=training_params.get("lora_rank", 32),
        lr=training_params.get("learning_rate", "N/A"),
    )

    # Write model card to checkpoint dir
    readme_path = os.path.join(checkpoint_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(model_card)

    logger.info(f"Uploading LoRA from {checkpoint_dir} to {hf_output_repo}...")
    api.upload_folder(
        folder_path=checkpoint_dir,
        repo_id=hf_output_repo,
        commit_message=f"Upload {model_type} LoRA trained with diffusion-pipe",
    )

    repo_url = f"https://huggingface.co/{hf_output_repo}"
    logger.info(f"Upload complete: {repo_url}")
    return repo_url
