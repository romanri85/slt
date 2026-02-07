import subprocess
import os
import logging

logger = logging.getLogger(__name__)

FLASH_ATTN_WHEEL_URL = (
    "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/"
    "v0.7.16/flash_attn-2.8.3+cu128torch2.10-cp312-cp312-linux_x86_64.whl"
)


def detect_cuda_arch():
    """Detect GPU and return CUDA architecture string (e.g. '90' for H100)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        gpu_name = result.stdout.strip().split("\n")[0].strip()
    except Exception:
        logger.warning("Could not detect GPU via nvidia-smi")
        return "80;86;89;90", "unknown"

    gpu_map = {
        ("B100", "B200", "GB200"): ("100", "blackwell"),
        ("5090", "5080", "5070", "5060"): ("120", "blackwell"),
        ("H100", "H200"): ("90", "hopper"),
        ("L4", "L40", "4090", "4080", "4070", "4060"): ("89", "ada"),
        ("A10", "A40", "A6000", "A5000", "A4000", "3090", "3080", "3070", "3060"): ("86", "ampere"),
        ("A100",): ("80", "ampere"),
        ("T4", "2080", "2070", "2060"): ("75", "turing"),
        ("V100",): ("70", "volta"),
    }

    for keywords, (arch, family) in gpu_map.items():
        for kw in keywords:
            if kw in gpu_name:
                logger.info(f"Detected GPU: {gpu_name} → sm_{arch} ({family})")
                return arch, family

    logger.warning(f"Unknown GPU: {gpu_name}, using multi-arch build")
    return "80;86;89;90", "unknown"


def ensure_flash_attn():
    """Install flash-attn if not already present. Try wheel first, then source."""
    try:
        import flash_attn  # noqa: F401
        logger.info(f"flash-attn already installed: {flash_attn.__version__}")
        return
    except ImportError:
        pass

    logger.info("flash-attn not found, installing...")

    # Try prebuilt wheel
    if _try_wheel():
        return

    # Fall back to source build
    _build_from_source()


def _try_wheel():
    """Try installing flash-attn from prebuilt wheel. Returns True on success."""
    logger.info("Trying prebuilt flash-attn wheel...")
    try:
        result = subprocess.run(
            ["pip", "install", FLASH_ATTN_WHEEL_URL],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode == 0:
            logger.info("flash-attn installed from prebuilt wheel")
            return True
        logger.warning(f"Wheel install failed: {result.stderr[:500]}")
    except Exception as e:
        logger.warning(f"Wheel install error: {e}")
    return False


def _build_from_source():
    """Build flash-attn from source."""
    logger.info("Building flash-attn from source (this may take 5-10 minutes)...")
    arch, _ = detect_cuda_arch()

    env = os.environ.copy()
    env["FLASH_ATTN_CUDA_ARCHS"] = arch
    env["MAX_JOBS"] = str(max(4, os.cpu_count() - 2))
    env["NVCC_THREADS"] = "4"

    # Ensure ninja is available
    subprocess.run(["pip", "install", "ninja", "packaging"], capture_output=True)

    result = subprocess.run(
        ["pip", "install", "flash-attn", "--no-build-isolation"],
        env=env, capture_output=True, text=True, timeout=1200,
    )
    if result.returncode != 0:
        logger.error(f"flash-attn source build failed:\n{result.stderr[-1000:]}")
        raise RuntimeError("Failed to install flash-attn from source")

    logger.info("flash-attn built and installed from source")
