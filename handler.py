"""RunPod serverless worker: BFS head swap on Qwen-Image-2.1 (same pipeline as Simzy/bfs-best-face-swap).

Weights:
  * Qwen/Qwen-Image-2.1 comes from RunPod's cached-model mount when the endpoint has it attached
    (/runpod-volume/huggingface-cache/hub), otherwise it is downloaded once per worker.
  * BFS LoRA from Simzy/BFS-Best-Face-Swap (Casey's mirror), turbo LoRA from the public Viggle repo
    (or TURBO_REPO env override). Both download at cold start into the container disk.

Input:  {"reference_face": <base64>, "target_image": <base64>, "base_model": "turbo"|"base",
         "resolution": 1024|1536|2048, "num_inference_steps": int, "lora_scale": float,
         "seed": int (-1 = random), "prompt": str (optional), "output_format": "png"|"jpeg"}
Output: {"image_base64": str, "mime_type": str, "seed": int, "width": int, "height": int,
         "elapsed": float}
"""

import base64
import io
import os
import random
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np  # noqa: E402
import runpod  # noqa: E402
import torch  # noqa: E402
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImage21Pipeline  # noqa: E402
from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_dimensions  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

BASE_MODEL_ID = os.environ.get("BASE_MODEL_ID", "Qwen/Qwen-Image-2.1")
BFS_REPO = os.environ.get("BFS_REPO", "Simzy/BFS-Best-Face-Swap")
BFS_FILE = "bfs_head_v1.1_qwen_2.1.safetensors"
TURBO_REPO = os.environ.get("TURBO_REPO", "Viggle/Qwen-Image-2.1-viggle-turbo")
TURBO_FILE = "Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r256.safetensors"
CACHE_ROOT = "/runpod-volume/huggingface-cache/hub"
LORA_DIR = "/tmp/loras"

MAX_SEED = np.iinfo(np.int32).max
FACE_SWAP_PROMPT = (
    "head_swap: start with <image1> as the base image, keeping its lighting, environment, and background. "
    "remove the head from <image1> completely and replace it with the head from <image2>, strictly preserving "
    "the hair, eye color, nose structure from <image2>. copy the direction of the eye, head rotation, micro "
    "expressions from <image1>, high quality, sharp details, 4k"
)


def resolve_cached(model_id: str):
    org, name = model_id.split("/", 1)
    root = os.path.join(CACHE_ROOT, f"models--{org}--{name}")
    snaps = os.path.join(root, "snapshots")
    ref = os.path.join(root, "refs", "main")
    if os.path.isfile(ref):
        cand = os.path.join(snaps, open(ref).read().strip())
        if os.path.isdir(cand):
            return cand
    if os.path.isdir(snaps):
        for d in sorted(os.listdir(snaps)):
            if os.path.isfile(os.path.join(snaps, d, "model_index.json")):
                return os.path.join(snaps, d)
    return None


t0 = time.time()
base_path = resolve_cached(BASE_MODEL_ID)
print(f"[load] base model from {'cache ' + base_path if base_path else 'hub ' + BASE_MODEL_ID}", flush=True)
pipe = QwenImage21Pipeline.from_pretrained(base_path or BASE_MODEL_ID, torch_dtype=torch.bfloat16)


def load_bfs_state_dict():
    """Split ComfyUI's fused img_mlp.gate_up LoRA into diffusers' gate/proj linears."""
    sd = load_file(hf_hub_download(BFS_REPO, BFS_FILE, local_dir=LORA_DIR))
    out = {}
    for k, v in sd.items():
        if ".img_mlp.gate_up." in k:
            gate_k = k.replace(".gate_up.", ".gate_layer.")
            up_k = k.replace(".gate_up.", ".proj.")
            if "lora_B" in k:
                gate, up = v.chunk(2, dim=0)
                out[gate_k], out[up_k] = gate.contiguous(), up.contiguous()
            else:
                out[gate_k], out[up_k] = v, v.clone()
        else:
            out[k] = v
    return out


print(f"[load] BFS LoRA from {BFS_REPO}", flush=True)
pipe.load_lora_weights(load_bfs_state_dict(), adapter_name="bfs")
print(f"[load] turbo LoRA from {TURBO_REPO}", flush=True)
turbo_path = hf_hub_download(TURBO_REPO, TURBO_FILE, local_dir=LORA_DIR)
pipe.load_lora_weights(turbo_path, adapter_name="turbo")

base_scheduler = pipe.scheduler
turbo_scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config, shift_terminal=None)
pipe.to("cuda")

try:
    patch_embed = pipe.text_encoder.model.visual.patch_embed

    def _patch_embed_forward(hidden_states):
        proj = patch_embed.proj
        weight = proj.weight.reshape(proj.weight.shape[0], -1)
        return hidden_states.to(weight.dtype).flatten(1) @ weight.T + proj.bias

    patch_embed.forward = _patch_embed_forward
except AttributeError as exc:
    print(f"[load] patch_embed fast path unavailable: {exc!r}", flush=True)

VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9
# Same budgets as the ZeroGPU Space: 48 GB slice -> 10 GB KV + tiled decode; big cards -> more.
KV_BUDGET = 10.0 if VRAM_GB < 75 else (20.0 if VRAM_GB < 120 else 30.0)
TILE_DECODE = VRAM_GB < 120
print(f"[load] ready in {time.time() - t0:.0f}s on {torch.cuda.get_device_name(0)} "
      f"({VRAM_GB:.0f} GB, kv={KV_BUDGET}, tile={TILE_DECODE})", flush=True)


def turbo_sigmas(steps: int):
    steps = min(max(int(steps), 4), 8)
    if steps == 4:
        return steps, [1.0, 0.75, 0.5, 0.25]
    return steps, [1.0 - 0.125 * i / (steps - 4) for i in range(steps - 4)] + [0.875, 0.75, 0.5, 0.25]


def kv_cache_fits(resolution: int, budget_gb: float) -> bool:
    prefix_tokens = 2 * (int(resolution) // 16) ** 2
    return prefix_tokens * 32 * 2 * 4096 * 2 <= budget_gb * 1e9


@torch.no_grad()
def decode_latents(latents, height, width, tile_decode):
    torch.cuda.empty_cache()
    vae = pipe.vae
    latents = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor).to(vae.dtype)
    mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(latents)
    std = torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(latents)
    latents = latents * std + mean
    if tile_decode:
        vae.enable_tiling(tile_sample_min_height=1024, tile_sample_min_width=1024,
                          tile_sample_stride_height=768, tile_sample_stride_width=768)
    try:
        image = vae.decode(latents, return_dict=False)[0][:, :, 0]
    finally:
        vae.disable_tiling()
    return pipe.image_processor.postprocess(image, output_type="pil")[0]


def decode_image(value, name):
    if not value or not isinstance(value, str):
        raise ValueError(f"{name} is required (base64 image)")
    if value.startswith("data:"):
        value = value.split(",", 1)[1]
    img = Image.open(io.BytesIO(base64.b64decode(value)))
    return ImageOps.exif_transpose(img).convert("RGB")


def handler(job):
    inp = job.get("input") or {}
    try:
        reference_face = decode_image(inp.get("reference_face"), "reference_face")
        target_image = decode_image(inp.get("target_image"), "target_image")
    except Exception as exc:  # bad input keeps the worker warm
        return {"error": f"Invalid input: {exc}"}
    turbo = str(inp.get("base_model", "turbo")).lower().startswith("turbo")
    res = int(inp.get("resolution", 1536))
    if res not in (1024, 1536, 2048):
        res = 1536
    steps_in = int(inp.get("num_inference_steps", 6 if turbo else 30))
    lora_scale = float(inp.get("lora_scale", 1.0))
    seed = int(inp.get("seed", -1))
    if seed < 0:
        seed = random.randint(0, MAX_SEED)
    prompt = (inp.get("prompt") or "").strip() or FACE_SWAP_PROMPT
    fmt = "jpeg" if str(inp.get("output_format", "png")).lower() in ("jpg", "jpeg") else "png"

    if turbo:
        pipe.set_adapters(["bfs", "turbo"], adapter_weights=[lora_scale, 1.0])
        pipe.scheduler = turbo_scheduler
        steps, sigmas = turbo_sigmas(steps_in)
    else:
        pipe.set_adapters(["bfs"], adapter_weights=[lora_scale])
        pipe.scheduler = base_scheduler
        steps, sigmas = max(1, min(60, steps_in)), None

    width, height, _ = calculate_dimensions(res * res, target_image.width / target_image.height)
    started = time.perf_counter()
    try:
        latents = pipe(
            prompt=prompt,
            image=[target_image, reference_face],
            width=width,
            height=height,
            output_resolution=res,
            num_inference_steps=steps,
            sigmas=sigmas,
            true_cfg_scale=1.0,
            use_kv_cache=kv_cache_fits(res, KV_BUDGET),
            generator=torch.Generator(device="cuda").manual_seed(seed),
            output_type="latent",
        ).images
        image = decode_latents(latents, height, width, TILE_DECODE).convert("RGB")
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {"error": "GPU out of memory; try a lower resolution."}
    elapsed = time.perf_counter() - started
    print(f"[swap] turbo={turbo} res={res} size={image.size} steps={steps} lora={lora_scale} "
          f"elapsed={elapsed:.1f}s", flush=True)
    buf = io.BytesIO()
    if fmt == "jpeg":
        image.save(buf, format="JPEG", quality=95)
    else:
        image.save(buf, format="PNG")
    return {
        "image_base64": base64.b64encode(buf.getvalue()).decode(),
        "mime_type": f"image/{fmt}",
        "seed": seed,
        "width": image.width,
        "height": image.height,
        "elapsed": round(elapsed, 2),
    }


runpod.serverless.start({"handler": handler})
