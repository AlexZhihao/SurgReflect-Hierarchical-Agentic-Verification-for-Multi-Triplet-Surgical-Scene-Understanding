from __future__ import annotations

import base64
import os
import time
from typing import List, Optional


def call_gemini_multimodal(
    model: str,
    prompt: str,
    image_data_urls: List[str],
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    max_output_tokens: int = 1500,
    retries: int = 4,
    sleep_sec: float = 3.0,
) -> str:
    """
    Call Google Gemini API with multimodal input (text + multiple images).

    Uses the google-genai SDK (preferred) with fallback to the older
    google.generativeai SDK.

    Args:
        model: Gemini model name (e.g., "gemini-2.5-pro")
        prompt: Text prompt
        image_data_urls: List of base64 data URLs for images
        api_key: Google/Gemini API key
        temperature: Sampling temperature
        max_output_tokens: Maximum tokens in response
        retries: Number of retry attempts
        sleep_sec: Sleep time between retries
    Returns:
        Model response text
    """
    key = api_key or os.getenv("GEMINI_API_KEY")

    # Build content parts: images first, then text
    parts = _build_parts(image_data_urls, prompt)

    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            text = _call_once(model, parts, key, temperature, max_output_tokens)
            return text
        except Exception as e:
            last_err = e
            if attempt < retries:
                wait = sleep_sec * (2 ** (attempt - 1))
                time.sleep(min(wait, 120))

    raise RuntimeError(f"Gemini multimodal call failed after {retries} retries: {last_err}")


def _build_parts(image_data_urls: List[str], prompt: str):
    """Build a list of content parts for the Gemini SDK."""
    parts = []
    for url in image_data_urls:
        if url.startswith("data:"):
            header, b64data = url.split(",", 1)
            mime = header.split(";")[0].split(":")[1] if ":" in header else "image/png"
            parts.append({
                "inline_data": {
                    "mime_type": mime,
                    "data": b64data,
                }
            })
    parts.append({"text": prompt})
    return parts


def _call_once(model: str, parts, api_key, temperature, max_output_tokens) -> str:
    """Try the google-genai SDK first, fall back to google.generativeai."""
    try:
        return _call_google_genai(model, parts, api_key, temperature, max_output_tokens)
    except ImportError:
        pass
    return _call_google_generativeai(model, parts, api_key, temperature, max_output_tokens)


def _call_google_genai(model, parts, api_key, temperature, max_output_tokens) -> str:
    from google import genai  # type: ignore
    from google.genai import types  # type: ignore

    client = genai.Client(api_key=api_key) if api_key else genai.Client()

    # Disable all safety filters — surgical images trigger SAFETY blocks
    safety = [
        types.SafetySetting(category="HARM_CATEGORY_HARASSMENT",          threshold="OFF"),
        types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH",         threshold="OFF"),
        types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT",   threshold="OFF"),
        types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT",   threshold="OFF"),
    ]

    # Gemini 2.5 thinking models use output tokens for internal "thinking".
    # If max_output_tokens is small, the model exhausts all tokens on thinking
    # and produces no visible text (finish_reason=MAX_TOKENS, empty response).
    # Fix: ensure generous output budget + limit thinking budget.
    is_thinking = "2.5" in model
    effective_max = max(max_output_tokens, 16384) if is_thinking else max_output_tokens

    config_kwargs = dict(
        temperature=temperature,
        max_output_tokens=effective_max,
        safety_settings=safety,
    )
    if is_thinking:
        config_kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_budget=4096,
        )

    config = types.GenerateContentConfig(**config_kwargs)
    resp = client.models.generate_content(
        model=model,
        contents=parts,
        config=config,
    )

    # For thinking models, resp.text may be None if only thinking parts exist.
    # Extract text from all non-thought parts.
    text = getattr(resp, "text", None)
    if text is None and hasattr(resp, "candidates") and resp.candidates:
        candidate = resp.candidates[0]
        if hasattr(candidate, "content") and candidate.content and candidate.content.parts:
            text_parts = []
            for p in candidate.content.parts:
                if hasattr(p, "thought") and p.thought:
                    continue  # skip thinking parts
                if hasattr(p, "text") and p.text:
                    text_parts.append(p.text)
            text = "\n".join(text_parts) if text_parts else None
    return (text or "").strip()


def _call_google_generativeai(model_name, parts, api_key, temperature, max_output_tokens) -> str:
    import google.generativeai as genai2  # type: ignore

    if api_key:
        genai2.configure(api_key=api_key)

    is_thinking = "2.5" in model_name
    effective_max = max(max_output_tokens, 16384) if is_thinking else max_output_tokens

    model = genai2.GenerativeModel(model_name)
    gen_config = genai2.types.GenerationConfig(
        temperature=temperature,
        max_output_tokens=effective_max,
    )
    # Disable all safety filters — surgical images trigger SAFETY blocks
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT",        "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH",       "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]
    resp = model.generate_content(parts, generation_config=gen_config, safety_settings=safety_settings)
    return (getattr(resp, "text", None) or "").strip()
