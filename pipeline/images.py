"""Free image generation via Hugging Face Inference (HF_TOKEN) with visual fallback."""
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


def _minimal_png_bytes(width: int = 768, height: int = 768, index: int = 0) -> bytes:
    """Create a valid PNG without external dependencies."""
    w = max(1, int(width))
    h = max(1, int(height))
    colors = [
        b"\x1f\x3a\x5f",
        b"\x4a\x2c\x5e",
        b"\x1e\x52\x4a",
        b"\x5e\x38\x24",
        b"\x3b\x4d\x28",
    ]
    rgb = colors[index % len(colors)]
    row = b"\x00" + (rgb * w)
    compressed = zlib.compress(row * h, level=6)

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
    """Create a varied visual fallback so the video never has a plain solid screen."""
    w = max(1, int(width))
    h = max(1, int(height))
    try:
        from PIL import Image, ImageDraw

        palettes = [
            ((18, 24, 38), (46, 125, 150), (235, 175, 65)),
            ((28, 20, 42), (130, 68, 165), (90, 210, 180)),
            ((18, 36, 34), (42, 145, 115), (240, 130, 75)),
            ((42, 24, 24), (175, 75, 55), (245, 195, 85)),
            ((22, 30, 46), (75, 115, 195), (225, 105, 135)),
        ]
        bg, accent, highlight = palettes[index % len(palettes)]

        img = Image.new("RGB", (w, h), color=bg)
        draw = ImageDraw.Draw(img)

        # Subtle horizontal bands for visual depth
        bands = 8
        band_h = max(1, h // bands)
        for b_idx in range(bands):
            factor = 0.75 + (b_idx / max(1, bands - 1)) * 0.45
            r = min(255, int(bg[0] * factor))
            g = min(255, int(bg[1] * factor))
            b = min(255, int(bg[2] * factor))
            y0 = b_idx * band_h
            y1 = h if b_idx == bands - 1 else (b_idx + 1) * band_h
            draw.rectangle([0, y0, w, y1], fill=(r, g, b))

        # Geometric shapes per scene
        cx, cy = w // 2, h // 2
        radius = max(24, min(w, h) // 4)
        offset = ((index % 3) - 1) * max(20, radius // 2)
        draw.ellipse(
            [cx - radius + offset, cy - radius, cx + radius + offset, cy + radius],
            outline=accent,
            width=max(3, min(w, h) // 120),
        )
        inner_r = max(12, radius // 2)
        draw.rectangle(
            [cx - inner_r, cy - inner_r + offset, cx + inner_r, cy + inner_r + offset],
            outline=highlight,
            width=max(2, min(w, h) // 180),
        )

        margin = max(20, min(w, h) // 24)
        draw.rectangle(
            [margin, margin, w - margin, h - margin],
            outline=accent,
            width=max(2, min(w, h) // 200),
        )

        draw.text(
            (margin + 16, h - margin - max(28, h // 18)),
            f"Scene {index + 1}",
            fill=(245, 247, 250),
        )

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        out_path.write_bytes(buf.getvalue())
    except ImportError:
        out_path.write_bytes(_minimal_png_bytes(w, h, index=index))


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
    """Generate one image via Hugging Face, with a safe visual fallback."""
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
    except Exception as fallback_err:
        return "fail", f"fallback failed: {fallback_err}"
