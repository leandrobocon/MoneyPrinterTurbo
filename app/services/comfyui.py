import json
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import quote_plus

import requests
from loguru import logger

from app.config import config
from app.models.schema import MaterialInfo, VideoAspect


DEFAULT_BASE_URL = "http://127.0.0.1:8188"
DEFAULT_WAN_MODEL = "wan2.1_t2v_1.3B_bf16.safetensors"
DEFAULT_CLIP_NAME = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
DEFAULT_VAE_NAME = "wan_2.1_vae.safetensors"
DEFAULT_STEPS = 12
DEFAULT_CFG = 4.0
DEFAULT_FPS = 16
DEFAULT_SAMPLER = "euler"
DEFAULT_SCHEDULER = "simple"
DEFAULT_PROMPT_TEMPLATE = (
    "{term}, detailed photorealistic wildlife documentary footage, natural lighting, sharp textures, high resolution"
)
DEFAULT_NEGATIVE_PROMPT = (
    "bright tones, overexposed, static, blurry details, cartoon, anime, 3d render, painting, drawing, "
    "illustration, plastic, fake, low quality, worst quality, artifacts, watermark"
)
DEFAULT_POLL_INTERVAL_SECONDS = 3.0
DEFAULT_RUN_TIMEOUT_SECONDS = 14400.0


class ComfyUIError(RuntimeError):
    """ComfyUI request or execution error."""

    def __init__(self, message: str, prompt_id: str = ""):
        super().__init__(message)
        self.prompt_id = prompt_id


def is_enabled(settings: Mapping[str, Any] | None = None) -> bool:
    """Check if ComfyUI service is configured and enabled."""
    settings = config.app if settings is None else settings
    base_url = str(settings.get("comfyui_base_url", DEFAULT_BASE_URL) or "").strip()
    return bool(base_url)


def _base_url() -> str:
    return str(
        config.app.get("comfyui_base_url", DEFAULT_BASE_URL) or DEFAULT_BASE_URL
    ).rstrip("/")


def _wan_model() -> str:
    return str(
        config.app.get("comfyui_wan_model", DEFAULT_WAN_MODEL) or DEFAULT_WAN_MODEL
    ).strip()


def _clip_name() -> str:
    return str(
        config.app.get("comfyui_clip_name", DEFAULT_CLIP_NAME) or DEFAULT_CLIP_NAME
    ).strip()


def _vae_name() -> str:
    return str(
        config.app.get("comfyui_vae_name", DEFAULT_VAE_NAME) or DEFAULT_VAE_NAME
    ).strip()


def _steps() -> int:
    try:
        val = int(config.app.get("comfyui_steps", DEFAULT_STEPS))
        return max(1, min(val, 100))
    except (TypeError, ValueError):
        return DEFAULT_STEPS


def _cfg() -> float:
    try:
        val = float(config.app.get("comfyui_cfg", DEFAULT_CFG))
        return max(1.0, min(val, 20.0))
    except (TypeError, ValueError):
        return DEFAULT_CFG


def _fps() -> int:
    try:
        val = int(config.app.get("comfyui_fps", DEFAULT_FPS))
        return max(8, min(val, 60))
    except (TypeError, ValueError):
        return DEFAULT_FPS


def _sampler() -> str:
    return str(
        config.app.get("comfyui_sampler", DEFAULT_SAMPLER) or DEFAULT_SAMPLER
    ).strip()


def _scheduler() -> str:
    return str(
        config.app.get("comfyui_scheduler", DEFAULT_SCHEDULER) or DEFAULT_SCHEDULER
    ).strip()


def _prompt_template() -> str:
    return str(
        config.app.get("comfyui_prompt_template", DEFAULT_PROMPT_TEMPLATE)
        or DEFAULT_PROMPT_TEMPLATE
    ).strip()


def _negative_prompt() -> str:
    return str(
        config.app.get("comfyui_negative_prompt", DEFAULT_NEGATIVE_PROMPT)
        or DEFAULT_NEGATIVE_PROMPT
    ).strip()


def _format_prompt(term: str) -> str:
    template = _prompt_template()
    if "{term}" in template:
        return template.replace("{term}", term)
    return f"{term}, {template}" if template else term


def _resolution_for_aspect(aspect: VideoAspect) -> tuple[int, int]:
    """Return width, height for Wan 2.1 depending on aspect ratio."""
    if aspect == VideoAspect.landscape:
        return 832, 480
    elif aspect == VideoAspect.portrait:
        return 480, 832
    elif aspect == VideoAspect.square:
        return 512, 512
    return 480, 832


def _calculate_wan_frame_count(duration_seconds: int, fps: int = DEFAULT_FPS) -> int:
    """
    Wan 2.1 uses 4*k + 1 frames.
    For example: 17 frames (~1s), 33 frames (~2s), 49 frames (~3s), 65 frames (~4s), 81 frames (~5s).
    """
    target_frames = max(1, duration_seconds) * fps
    k = max(1, round((target_frames - 1) / 4))
    return int(4 * k + 1)


def build_wan_workflow(
    positive_prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    frame_count: int,
    steps: int,
    cfg: float,
    seed: int,
    fps: int,
) -> dict[str, Any]:
    """Constructs the prompt graph for ComfyUI Wan 2.1 Text-to-Video."""
    return {
        "10": {
            "inputs": {
                "unet_name": _wan_model(),
                "weight_dtype": "default",
            },
            "class_type": "UNETLoader",
        },
        "11": {
            "inputs": {
                "clip_name": _clip_name(),
                "type": "wan",
            },
            "class_type": "CLIPLoader",
        },
        "12": {
            "inputs": {
                "vae_name": _vae_name(),
            },
            "class_type": "VAELoader",
        },
        "13": {
            "inputs": {
                "shift": 8.0,
                "model": ["10", 0],
            },
            "class_type": "ModelSamplingSD3",
        },
        "14": {
            "inputs": {
                "text": positive_prompt,
                "clip": ["11", 0],
            },
            "class_type": "CLIPTextEncode",
        },
        "15": {
            "inputs": {
                "text": negative_prompt,
                "clip": ["11", 0],
            },
            "class_type": "CLIPTextEncode",
        },
        "16": {
            "inputs": {
                "width": width,
                "height": height,
                "length": frame_count,
                "batch_size": 1,
            },
            "class_type": "EmptyHunyuanLatentVideo",
        },
        "17": {
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": _sampler(),
                "scheduler": _scheduler(),
                "denoise": 1.0,
                "model": ["13", 0],
                "positive": ["14", 0],
                "negative": ["15", 0],
                "latent_image": ["16", 0],
            },
            "class_type": "KSampler",
        },
        "18": {
            "inputs": {
                "samples": ["17", 0],
                "vae": ["12", 0],
            },
            "class_type": "VAEDecode",
        },
        "19": {
            "inputs": {
                "fps": float(fps),
                "images": ["18", 0],
            },
            "class_type": "CreateVideo",
        },
        "20": {
            "inputs": {
                "filename_prefix": "video/MPT_Wan",
                "format": "auto",
                "video": ["19", 0],
            },
            "class_type": "SaveVideo",
        },
    }


def generate_video(
    search_term: str,
    video_aspect: VideoAspect,
    duration: int,
    save_dir: str,
    seed: int | None = None,
) -> tuple[str | None, int]:
    """
    Submits a Wan 2.1 video generation request to ComfyUI, waits for completion,
    and returns (saved_local_path, actual_duration_seconds).
    """
    base_url = _base_url()
    steps = _steps()
    cfg = _cfg()
    fps = _fps()
    positive_prompt = _format_prompt(search_term)
    negative_prompt = _negative_prompt()
    width, height = _resolution_for_aspect(video_aspect)
    frame_count = _calculate_wan_frame_count(duration, fps)
    actual_duration = max(1, round(frame_count / fps))
    actual_seed = random.randint(1, 10**14) if seed is None else seed

    logger.info(
        f"ComfyUI Wan 2.1 video request: term={search_term!r}, prompt={positive_prompt!r}, "
        f"resolution={width}x{height}, frames={frame_count}, steps={steps}, fps={fps}"
    )

    workflow = build_wan_workflow(
        positive_prompt=positive_prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        frame_count=frame_count,
        steps=steps,
        cfg=cfg,
        seed=actual_seed,
        fps=fps,
    )

    try:
        resp = requests.post(
            f"{base_url}/prompt",
            json={"prompt": workflow},
            timeout=30,
        )
        resp.raise_for_status()
        res_data = resp.json()
        prompt_id = res_data.get("prompt_id")
        if not prompt_id:
            raise ComfyUIError(f"No prompt_id returned by ComfyUI: {res_data}")
    except Exception as e:
        logger.error(f"Failed to submit prompt to ComfyUI: {e}")
        raise ComfyUIError(f"Failed to submit prompt to ComfyUI: {e}")

    logger.info(f"ComfyUI prompt submitted: prompt_id={prompt_id}. Waiting for generation...")

    start_time = time.time()
    poll_interval = float(
        config.app.get("comfyui_poll_interval_seconds", DEFAULT_POLL_INTERVAL_SECONDS)
        or DEFAULT_POLL_INTERVAL_SECONDS
    )
    timeout = float(
        config.app.get("comfyui_timeout_seconds", DEFAULT_RUN_TIMEOUT_SECONDS)
        or DEFAULT_RUN_TIMEOUT_SECONDS
    )

    output_files: list[dict[str, Any]] = []
    last_log_time = start_time
    last_active_time = start_time
    max_inactivity_seconds = 1800.0  # 30 minutes without being in queue or history before timing out

    while True:
        time.sleep(poll_interval)
        elapsed = time.time() - start_time
        
        # Check history first
        try:
            h_resp = requests.get(f"{base_url}/history/{prompt_id}", timeout=15)
            if h_resp.status_code == 200:
                h_data = h_resp.json()
                if prompt_id in h_data:
                    prompt_info = h_data[prompt_id]
                    status_info = prompt_info.get("status", {})
                    if status_info.get("status_str") == "error":
                        messages = status_info.get("messages", [])
                        raise ComfyUIError(f"ComfyUI execution error: {messages}", prompt_id=prompt_id)

                    outputs = prompt_info.get("outputs", {})
                    for node_id, node_output in outputs.items():
                        if "images" in node_output:
                            output_files.extend(node_output["images"])
                        if "videos" in node_output:
                            output_files.extend(node_output["videos"])
                        if "gifs" in node_output:
                            output_files.extend(node_output["gifs"])

                    if output_files or "outputs" in prompt_info:
                        logger.info(f"ComfyUI generation completed for prompt_id={prompt_id} in {int(elapsed)}s")
                        break
        except ComfyUIError:
            raise
        except Exception as e:
            logger.warning(f"Error checking ComfyUI history for {prompt_id}: {e}")

        # Check queue status if not yet in history
        is_in_queue = False
        queue_state = "waiting"
        try:
            q_resp = requests.get(f"{base_url}/queue", timeout=15)
            if q_resp.status_code == 200:
                q_data = q_resp.json()
                running_ids = [item[1] for item in q_data.get("queue_running", []) if len(item) > 1]
                pending_ids = [item[1] for item in q_data.get("queue_pending", []) if len(item) > 1]
                if prompt_id in running_ids:
                    is_in_queue = True
                    queue_state = "running (generating frames)"
                    last_active_time = time.time()
                elif prompt_id in pending_ids:
                    is_in_queue = True
                    queue_state = f"pending in queue (position {pending_ids.index(prompt_id) + 1}/{len(pending_ids)})"
                    last_active_time = time.time()
        except Exception as q_err:
            logger.debug(f"Error checking ComfyUI queue: {q_err}")

        if is_in_queue:
            last_active_time = time.time()

        if elapsed - (last_log_time - start_time) >= 30.0:
            last_log_time = time.time()
            logger.info(
                f"ComfyUI Wan 2.1 video: prompt_id={prompt_id}, state={queue_state}, "
                f"elapsed={int(elapsed)}s"
            )

        # Timeout only if prompt has disappeared from both queue and history for too long
        if not is_in_queue and (time.time() - last_active_time > max_inactivity_seconds):
            logger.error(
                f"ComfyUI prompt {prompt_id} disappeared from queue and did not appear in history for {int(max_inactivity_seconds)}s"
            )
            break

    target_dir = Path(save_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / f"comfyui_wan_{prompt_id}.mp4"

    if output_files:
        file_meta = output_files[0]
        filename = file_meta.get("filename", "")
        subfolder = file_meta.get("subfolder", "")
        file_type = file_meta.get("type", "output")

        view_url = f"{base_url}/view?filename={quote_plus(filename)}&subfolder={quote_plus(subfolder)}&type={quote_plus(file_type)}"
        logger.info(f"Fetching generated video from ComfyUI: {view_url}")
        try:
            v_resp = requests.get(view_url, timeout=120)
            v_resp.raise_for_status()
            with open(target_file, "wb") as f:
                f.write(v_resp.content)
            logger.info(f"Saved ComfyUI video to {target_file}")
            return str(target_file), actual_duration
        except Exception as e:
            logger.error(f"Failed to download video from {view_url}: {e}")
            raise ComfyUIError(f"Failed to download video from ComfyUI: {e}", prompt_id=prompt_id)

    raise ComfyUIError(f"ComfyUI did not produce output video for prompt_id={prompt_id}", prompt_id=prompt_id)
