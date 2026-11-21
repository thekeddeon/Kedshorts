"""Image generation via Hugging Face Inference (free with HF_TOKEN) + local fallback."""
from __future__ import annotations

import io
import os
import struct
import zlib
from pathlib import Path

import httpx

HF_DEFAULT_MODEL = "black-forest-labs/FLUX.1-schnell"

STYLE_SUFFIX = (
    ", cinematic digital illustration, detailed scene art, strong composition, "
    "professional youtube visual quality, no text, no captions, no watermark, no logos"
)

DEFAULT_NEGATIVE = (
    "blurry, low quality, watermark, logo, text, title, signature, ugly, grainy, "
    "gore, blood, nudity, child-unsafe"
)


def full_visual_prompt(scene: str, style_suffix: str | None = None) -> str:
    """Combine the scene description with a channel-specific style suffix."""
    return f"{scene.strip()}{(style_suffix or STYLE_SUFFIX)}"


def _minimal_png_bytes(width: int = 768, height: int = 768) -> bytes:
    """Create a valid solid-color PNG without external dependencies."""
    w = max(1, int(width))
    h = max(1, int(height))
    row = b"\x00" + (b"\x18\x1c\x2a" * w)
    raw_data = row * h
    compressed = zlib.compress(raw_data, level=6)

    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", compressed)
        + _chunk(b"IEND", b"")
    )


def _local_fallback_image(index: int, width: int, height: int, out_path: Path) -> None:
    """Generate a local fallback image so downstream pipeline steps can run for free."""
    try:
        from PIL import Image, ImageDraw

        palette = [
            (28, 37, 65),
            (38, 53, 72),
            (46, 36, 59),
            (27, 52, 58),
            (54, 42, 46),
        ]
        bg = palette[index % len(palette)]
        img = Image.new("RGB", (max(1, width), max(1, height)), color=bg)
        draw = ImageDraw.Draw(img)
        draw.text((40, max(10, height // 2)), f"Scene {index + 1}", fill=(240, 244, 248))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        out_path.write_bytes(buf.getvalue())
    except ImportError:
        out_path.write_bytes(_minimal_png_bytes(width, height))


def _hf_generate(
    prompt: str,
    *,
    hf_token: str,
    width: int,
    height: int,
    model: str,
) -> bytes:
    """Generate an image via Hugging Face Inference API."""
    url = f"https://router.huggingface.co/hf-inference/models/{model}"
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Accept": "image/png",
    }
    payload = {
        "inputs": prompt,
        "parameters": {
            "width": width,
            "height": height,
        },
    }

    with httpx.Client(timeout=120.0) as client:
        resp = client.post(url, json=payload, headers=headers)
        if resp.status_code >= 400:
            raise RuntimeError(f"HF {resp.status_code} (model={model}): {resp.text}")
        content_type = resp.headers.get("content-type", "")
        if "image" not in content_type and not resp.content.startswith(b"\x89PNG"):
            raise RuntimeError(f"HF returned non-image response: {resp.text[:200]}")
        return resp.content


def save_scene_image(
    index: int,
    prompt: str,
    out_path: Path,
    *,
    width: int = 768,
    height: int = 768,
    negative: str = DEFAULT_NEGATIVE,
) -> tuple[str, str]:
    """Generate and save one image. Returns (status, detail)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    hf_token = os.environ.get("HF_TOKEN", "").strip()
    hf_model = os.environ.get("HF_IMAGE_MODEL", HF_DEFAULT_MODEL).strip() or HF_DEFAULT_MODEL

    if hf_token:
        try:
            img_bytes = _hf_generate(
                prompt,
                hf_token=hf_token,
                width=width,
                height=height,
                model=hf_model,
            )
            out_path.write_bytes(img_bytes)
            return "ok", f"hf:{hf_model}"
        except Exception as e:
            try:
                _local_fallback_image(index, width, height, out_path)
                return "ok", f"fallback-local ({e})"
            except Exception as fallback_err:
                return "fail", f"{e} | fallback failed: {fallback_err}"

    try:
        _local_fallback_image(index, width, height, out_path)
        return "ok", "fallback-local (HF_TOKEN not set)"
    except Exception as fallback_err:
        return "fail", f"fallback failed: {fallback_err}"
