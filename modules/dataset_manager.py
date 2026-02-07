import os
import logging
from pathlib import Path
from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}


def download_dataset(hf_dataset_repo, hf_token, work_dir):
    """
    Download a HuggingFace dataset repo and organize into image/video/eval dirs.

    Expected HF dataset layout (flexible):
      images/    → training images (.jpg, .png, etc.) + optional .txt captions
      videos/    → training videos (.mp4, etc.) + optional .txt captions
      eval/      → eval images/videos + captions

    If the repo has flat files (no subdirs), they are sorted by extension.

    Returns dict with paths: {"images": ..., "videos": ..., "eval": ...}
    """
    download_dir = os.path.join(work_dir, "hf_dataset")

    logger.info(f"Downloading dataset from {hf_dataset_repo}...")
    snapshot_download(
        repo_id=hf_dataset_repo,
        repo_type="dataset",
        local_dir=download_dir,
        token=hf_token,
    )
    logger.info(f"Dataset downloaded to {download_dir}")

    # Set up working directories
    image_dir = os.path.join(work_dir, "image_dataset")
    video_dir = os.path.join(work_dir, "video_dataset")
    eval_dir = os.path.join(work_dir, "eval_dataset")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    # Check for standard subdirectories
    hf_images = os.path.join(download_dir, "images")
    hf_videos = os.path.join(download_dir, "videos")
    hf_eval = os.path.join(download_dir, "eval")

    if os.path.isdir(hf_images):
        _symlink_contents(hf_images, image_dir)
    if os.path.isdir(hf_videos):
        _symlink_contents(hf_videos, video_dir)
    if os.path.isdir(hf_eval):
        _symlink_contents(hf_eval, eval_dir)

    # If no standard subdirs found, sort flat files by extension
    if not os.path.isdir(hf_images) and not os.path.isdir(hf_videos):
        logger.info("No images/ or videos/ subdirs found, sorting flat files by extension")
        _sort_flat_files(download_dir, image_dir, video_dir)

    # Count files
    img_count = _count_media_files(image_dir)
    vid_count = _count_media_files(video_dir)
    eval_count = _count_media_files(eval_dir)

    logger.info(f"Dataset ready: {img_count} images, {vid_count} videos, {eval_count} eval files")

    if img_count == 0 and vid_count == 0:
        raise FileNotFoundError(
            f"No image or video files found in dataset '{hf_dataset_repo}'. "
            "Expected images/ or videos/ subdirectories, or flat media files."
        )

    return {
        "images": image_dir,
        "videos": video_dir,
        "eval": eval_dir,
        "has_images": img_count > 0,
        "has_videos": vid_count > 0,
        "has_eval": eval_count > 0,
        "image_count": img_count,
        "video_count": vid_count,
    }


def _symlink_contents(src_dir, dst_dir):
    """Symlink all files from src_dir into dst_dir."""
    for item in Path(src_dir).iterdir():
        if item.is_file():
            dst = Path(dst_dir) / item.name
            if not dst.exists():
                os.symlink(item, dst)


def _sort_flat_files(download_dir, image_dir, video_dir):
    """Sort flat media files into image_dir and video_dir."""
    for item in Path(download_dir).iterdir():
        if not item.is_file():
            continue
        ext = item.suffix.lower()
        if ext in IMAGE_EXTENSIONS:
            dst = Path(image_dir) / item.name
            if not dst.exists():
                os.symlink(item, dst)
        elif ext in VIDEO_EXTENSIONS:
            dst = Path(video_dir) / item.name
            if not dst.exists():
                os.symlink(item, dst)
        elif ext == ".txt":
            # Caption file — check if matching media exists
            stem = item.stem
            for media_ext in IMAGE_EXTENSIONS:
                if (Path(download_dir) / f"{stem}{media_ext}").exists():
                    dst = Path(image_dir) / item.name
                    if not dst.exists():
                        os.symlink(item, dst)
                    break
            else:
                for media_ext in VIDEO_EXTENSIONS:
                    if (Path(download_dir) / f"{stem}{media_ext}").exists():
                        dst = Path(video_dir) / item.name
                        if not dst.exists():
                            os.symlink(item, dst)
                        break


def _count_media_files(directory):
    """Count image and video files in a directory."""
    count = 0
    all_ext = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
    for item in Path(directory).iterdir():
        if item.is_file() and item.suffix.lower() in all_ext:
            count += 1
    return count
