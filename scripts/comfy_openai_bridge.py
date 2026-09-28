"""
ComfyUI to OpenAI Images API Bridge for MoneyPrinterTurbo
Translates POST /v1/images/generations into ComfyUI (Qwen Image 2.1 / FLUX.1 / Z-Image) workflow executions.
"""

import argparse
import base64
import json
import os
import sys
import time
import urllib.request
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler

COMFY_HOST = "127.0.0.1"
COMFY_PORT = 8188

DEFAULT_MODEL = "qwen-image-2.1"


def build_qwen_image_workflow(
    prompt_text: str,
    width: int = 768,
    height: int = 1344,
    seed: int = None,
    steps: int = 8,
    cfg: float = 4.0,
):
    """
    Constructs the prompt graph for Qwen-Image 2.1 / Z-Image / Wan in ComfyUI.
    """
    if seed is None:
        seed = int(time.time() * 1000) % 2147483647

    # Qwen-Image workflow definition for ComfyUI
    return {
        "1": {
            "inputs": {
                "unet_name": "qwen_image_2.1_int8_convrot.safetensors",
                "weight_dtype": "default",
            },
            "class_type": "UNETLoader",
        },
        "2": {
            "inputs": {
                "clip_name": "qwen3vl_8b_int8_convrot.safetensors",
                "type": "sdxl",
            },
            "class_type": "CLIPLoader",
        },
        "3": {
            "inputs": {
                "vae_name": "qwen_image_2.1_vae_bf16.safetensors",
            },
            "class_type": "VAELoader",
        },
        "4": {
            "inputs": {
                "clip": ["2", 0],
                "text": prompt_text,
            },
            "class_type": "CLIPTextEncode",
        },
        "5": {
            "inputs": {
                "clip": ["2", 0],
                "text": "low quality, worst quality, deformed, blurry, bad anatomy, artifacts, text, watermark",
            },
            "class_type": "CLIPTextEncode",
        },
        "6": {
            "inputs": {
                "width": width,
                "height": height,
                "batch_size": 1,
            },
            "class_type": "EmptyLatentImage",
        },
        "7": {
            "inputs": {
                "model": ["1", 0],
                "positive": ["4", 0],
                "negative": ["5", 0],
                "latent_image": ["6", 0],
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0,
            },
            "class_type": "KSampler",
        },
        "8": {
            "inputs": {
                "samples": ["7", 0],
                "vae": ["3", 0],
            },
            "class_type": "VAEDecode",
        },
        "9": {
            "inputs": {
                "filename_prefix": "MPT_QWEN_IMG",
                "images": ["8", 0],
            },
            "class_type": "SaveImage",
        },
    }


def build_flux_workflow(
    prompt_text: str,
    width: int = 768,
    height: int = 1344,
    seed: int = None,
):
    if seed is None:
        seed = int(time.time() * 1000) % 2147483647

    # Standard FLUX.1 Schnell ComfyUI API workflow definition
    return {
        "1": {
            "inputs": {
                "unet_name": "flux1-schnell-fp8.safetensors",
                "weight_dtype": "default",
            },
            "class_type": "UNETLoader",
        },
        "2": {
            "inputs": {
                "clip_name1": "t5xxl_fp8_e4m3fn.safetensors",
                "clip_name2": "clip_l.safetensors",
                "type": "flux",
            },
            "class_type": "DualCLIPLoader",
        },
        "3": {
            "inputs": {
                "vae_name": "ae.safetensors",
            },
            "class_type": "VAELoader",
        },
        "4": {
            "inputs": {
                "clip": ["2", 0],
                "text": prompt_text,
            },
            "class_type": "CLIPTextEncode",
        },
        "5": {
            "inputs": {
                "width": width,
                "height": height,
                "batch_size": 1,
            },
            "class_type": "EmptySD3LatentImage",
        },
        "6": {
            "inputs": {
                "model": ["1", 0],
                "positive": ["4", 0],
                "negative": ["4", 0],
                "latent_image": ["5", 0],
                "seed": seed,
                "steps": 4,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0,
            },
            "class_type": "KSampler",
        },
        "7": {
            "inputs": {
                "samples": ["6", 0],
                "vae": ["3", 0],
            },
            "class_type": "VAEDecode",
        },
        "8": {
            "inputs": {
                "filename_prefix": "MPT_FLUX",
                "images": ["7", 0],
            },
            "class_type": "SaveImage",
        },
    }


def queue_prompt(workflow, server_url=f"http://{COMFY_HOST}:{COMFY_PORT}"):
    p = {"prompt": workflow}
    data = json.dumps(p).encode("utf-8")
    req = urllib.request.Request(
        f"{server_url}/prompt",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_history(prompt_id, server_url=f"http://{COMFY_HOST}:{COMFY_PORT}"):
    with urllib.request.urlopen(f"{server_url}/history/{prompt_id}") as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_image_data(
    filename, subfolder, folder_type, server_url=f"http://{COMFY_HOST}:{COMFY_PORT}"
):
    params = urllib.parse.urlencode(
        {"filename": filename, "subfolder": subfolder, "type": folder_type}
    )
    with urllib.request.urlopen(f"{server_url}/view?{params}") as resp:
        return resp.read()


def generate_comfy_image(
    prompt: str,
    model: str = DEFAULT_MODEL,
    width: int = 768,
    height: int = 1344,
) -> bytes:
    server_url = f"http://{COMFY_HOST}:{COMFY_PORT}"
    model_lower = str(model or "").lower()

    if "flux" in model_lower:
        workflow = build_flux_workflow(prompt, width, height)
        model_name = "FLUX.1 Schnell"
    else:
        # Default to Qwen Image 2.1
        workflow = build_qwen_image_workflow(prompt, width, height)
        model_name = "Qwen Image 2.1"

    print(
        f"[Bridge] Queueing prompt to ComfyUI ({model_name}, {width}x{height}): {prompt[:80]}..."
    )
    res = queue_prompt(workflow, server_url)
    prompt_id = res.get("prompt_id")
    if not prompt_id:
        raise RuntimeError(f"ComfyUI returned no prompt_id: {res}")
    print(f"[Bridge] Prompt queued, id={prompt_id}. Waiting for completion...")

    for _ in range(120):
        time.sleep(2)
        try:
            hist = get_history(prompt_id, server_url)
            if prompt_id in hist:
                prompt_info = hist[prompt_id]
                status_info = prompt_info.get("status", {})
                if status_info.get("status_str") == "error":
                    messages = status_info.get("messages", [])
                    raise RuntimeError(f"ComfyUI execution error: {messages}")

                outputs = prompt_info.get("outputs", {})
                for node_id, node_output in outputs.items():
                    if "images" in node_output and len(node_output["images"]) > 0:
                        img_info = node_output["images"][0]
                        print(f"[Bridge] Found output image: {img_info['filename']}")
                        return get_image_data(
                            img_info["filename"],
                            img_info.get("subfolder", ""),
                            img_info.get("type", "output"),
                            server_url,
                        )
        except Exception as e:
            if "ComfyUI execution error" in str(e):
                raise
            print(f"[Bridge] Polling history note: {e}")

    raise TimeoutError("ComfyUI did not complete generation within timeout")


class OpenAIImageHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/v1/images/generations":
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)
            req_json = json.loads(post_data.decode("utf-8"))

            prompt = req_json.get("prompt", "")
            model = req_json.get("model", DEFAULT_MODEL)
            size = req_json.get("size", "1080x1920")
            response_format = req_json.get("response_format", "b64_json")

            # Map size
            try:
                w, h = map(int, size.split("x"))
            except Exception:
                w, h = 768, 1344

            # Map dimensions to reasonable aspect
            if w > h:
                width, height = 1344, 768
            elif w == h:
                width, height = 1024, 1024
            else:
                width, height = 768, 1344

            print(
                f"\n[API] Received generation request: model={model}, prompt={prompt!r}, size={size}"
            )
            try:
                img_bytes = generate_comfy_image(
                    prompt, model=model, width=width, height=height
                )
                b64_str = base64.b64encode(img_bytes).decode("utf-8")

                response_payload = {
                    "created": int(time.time()),
                    "data": [
                        {
                            "b64_json": b64_str,
                        }
                    ],
                }

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response_payload).encode("utf-8"))
                print("[API] Successfully generated and returned image b64")
            except Exception as e:
                print(f"[API] Error generating image: {e}")
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(
                    json.dumps({"error": {"message": str(e)}}).encode("utf-8")
                )
        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        if self.path in ("/v1/models", "/v1/images"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            models_list = [
                {"id": "qwen-image-2.1"},
                {"id": "qwen-image"},
                {"id": "flux1-schnell"},
                {"id": "z_image_turbo"},
            ]
            self.wfile.write(json.dumps({"data": models_list}).encode("utf-8"))
        else:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ComfyUI Qwen Image 2.1 / FLUX OpenAI Bridge is Running")


def run(port=8000):
    server_address = ("127.0.0.1", port)
    httpd = HTTPServer(server_address, OpenAIImageHandler)
    print(
        f"Starting ComfyUI Qwen Image 2.1 / FLUX OpenAI Image Bridge on http://127.0.0.1:{port}"
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    args = parser.parse_args()
    run(port=args.port)
