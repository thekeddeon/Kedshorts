HF_DEFAULT_MODEL = "black-forest-labs/FLUX.1-schnell"


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
            return "fail", f"{e} | fallback failed: {fallback_err}"
