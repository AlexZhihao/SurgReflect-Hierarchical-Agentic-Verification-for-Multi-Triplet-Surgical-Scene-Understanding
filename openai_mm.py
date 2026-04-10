from __future__ import annotations

import time
from typing import List, Optional

from openai import OpenAI


def call_openai_multimodal(
    model: str,
    prompt: str,
    image_data_urls: List[str],
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.0,
    max_output_tokens: int = 1500,
    retries: int = 3,
    sleep_sec: float = 0.8,
) -> str:
    client = OpenAI(api_key=api_key, base_url=base_url) if (api_key or base_url) else OpenAI()

    content = [{"type": "text", "text": prompt}]
    for url in image_data_urls:
        content.append({"type": "image_url", "image_url": {"url": url}})

    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            # Try max_completion_tokens first (for newer models like gpt-5.2), fallback to max_tokens
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": content}],
                    temperature=temperature,
                    max_completion_tokens=max_output_tokens,
                )
            except Exception as e:
                if "max_completion_tokens" in str(e) or "unsupported" in str(e).lower():
                    # Fallback to max_tokens for older models
                    resp = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": content}],
                        temperature=temperature,
                        max_tokens=max_output_tokens,
                    )
                else:
                    raise
            text = (resp.choices[0].message.content or "").strip()
            return text
        except Exception as e:
            last_err = e
            time.sleep(sleep_sec * attempt)

    raise RuntimeError(f"OpenAI call failed after {retries} retries: {last_err}")
