"""Image generation via DeAPI.ai (async: submit -> poll -> download) with free local fallback."""
from __future__ import annotations

import io
import os
import random
import struct
import time
import zlib
from pathlib import Path

import httpx

DEAPI_SUBMIT_URL = "https://api.deapi.ai/api/v1/client/txt2img"
DEAPI_POLL_URL = "https://api.deapi.ai/api/v1/client/request-status"

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
    # Dark slate background byte + RGB (24, 28, 42) per pixel
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

        img = Image.new("RGB", (max(1, width), max(1, height)), color=(24, 28, 42))
        draw = ImageDraw.Draw(img)
        draw.text((40, max(10, height // 2)), f"Scene {index + 1}", fill=(240, 244, 248))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        out_path.write_bytes(buf.getvalue())
    except ImportError:
        out_path.write_bytes(_minimal_png_bytes(width, height))


def _deapi_generate(
    prompt: str,
    *,
    api_key: str,
    width: int,
    height: int,
    model: str,
    max_polls: int = 30,
    poll_interval: float = 3.0,
) -> bytes:
    """Submit image job, poll until done, download result."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    # Step 1: Submit
    payload = {
        "prompt": prompt,
        "model": model,
        "width": width,
        "height": height,
        "steps": 4,
        "seed": random.randint(1, 999999),
    }

    with httpx.Client(timeout=60.0) as client:
        # Submit with retry on 429
        for submit_try in range(5):
            resp = client.post(DEAPI_SUBMIT_URL, json=payload, headers=headers)
            if resp.status_code == 429:
                wait = 15 * (submit_try + 1)
                print(f"      DeAPI 429 on submit — waiting {wait}s (try {submit_try + 1}/5)…")
                time.sleep(wait)
                continue
            if resp.status_code >= 400:
                raise RuntimeError(f"DeAPI {resp.status_code} (model={model}): {resp.text}")
            break
        else:
            raise RuntimeError("DeAPI: 429 on submit after 5 retries")

        data = resp.json()

        request_id = data.get("data", {}).get("request_id")
        if not request_id:
            raise RuntimeError(f"No request_id in DeAPI response: {data}")
        print(f"      DeAPI submitted (id: {request_id})")

        # Step 2: Poll
        poll_headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        }

        for attempt in range(1, max_polls + 1):
            time.sleep(poll_interval)

            poll_resp = client.get(
                f"{DEAPI_POLL_URL}/{request_id}",
                headers=poll_headers,
                timeout=30.0,
            )
            poll_resp.raise_for_status()
            poll_data = poll_resp.json()

            status = poll_data.get("data", {}).get("status", "")

            if status in ("completed", "success", "done"):
                image_url = poll_data["data"].get("result_url")
                if not image_url:
                    raise RuntimeError(f"Completed but no result_url: {poll_data}")

                img_resp = client.get(image_url, timeout=60.0)
                img_resp.raise_for_status()
                print(f"      DeAPI done (polled {attempt}x)")
                return img_resp.content

            if status in ("failed", "error"):
                raise RuntimeError(f"DeAPI image failed: {poll_data}")

            # Still processing — keep polling

        raise RuntimeError(f"DeAPI timed out after {max_polls} polls for {request_id}")


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

    api_key = os.environ.get("DEAPI_TOKEN", "").strip()
    model = os.environ.get("DEAPI_MODEL", "Flux_2_Klein_4B_BF16")

    if not api_key:
        _local_fallback_image(index, width, height, out_path)
        return "ok", "fallback-local (DEAPI_TOKEN not set)"

    try:
        img_bytes = _deapi_generate(
            prompt,
            api_key=api_key,
            width=width,
            height=height,
            model=model,
        )
        out_path.write_bytes(img_bytes)
        return "ok", "deapi"
    except Exception as e:
        try:
            _local_fallback_image(index, width, height, out_path)
            return "ok", f"fallback-local ({e})"
        except Exception as fallback_err:
            return "fail", f"{e} | fallback failed: {fallback_err}"
