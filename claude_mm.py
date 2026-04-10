from __future__ import annotations

import time
from typing import List, Optional

import anthropic


def call_claude_multimodal(
    model: str,
    prompt: str,
    image_data_urls: List[str],
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    max_output_tokens: int = 1500,
    retries: int = 6,
    sleep_sec: float = 5.0,
) -> str:
    """
    Call Claude (Anthropic) API with multimodal input (text + multiple images).
    
    Args:
        model: Claude model name (e.g., "claude-opus-4-6")
        prompt: Text prompt
        image_data_urls: List of base64 data URLs for images
        api_key: Anthropic API key
        temperature: Sampling temperature
        max_output_tokens: Maximum tokens in response
        retries: Number of retry attempts
        sleep_sec: Sleep time between retries
    
    Returns:
        Model response text
    """
    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
    
    # Convert data URLs to Claude format
    content = []
    for url in image_data_urls:
        # Extract base64 data and media type from data URL
        # Format: data:image/png;base64,iVBORw0KG...
        if url.startswith("data:"):
            parts = url.split(",", 1)
            if len(parts) == 2:
                media_info = parts[0].split(";")[0].split(":")[1]  # e.g., "image/png"
                media_type = media_info if "/" in media_info else "image/png"
                base64_data = parts[1]
                
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": base64_data,
                    }
                })
    
    # Add text prompt at the end
    content.append({
        "type": "text",
        "text": prompt
    })
    
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_output_tokens,
                temperature=temperature,
                messages=[{
                    "role": "user",
                    "content": content
                }]
            )
            
            # Extract text from response
            text = ""
            for block in response.content:
                if block.type == "text":
                    text += block.text
            
            return text.strip()
            
        except Exception as e:
            last_err = e
            if attempt < retries:
                wait = sleep_sec * (2 ** (attempt - 1))  # exponential backoff
                time.sleep(min(wait, 120))  # cap at 2 min
            else:
                raise RuntimeError(f"Claude API call failed after {retries} retries: {last_err}")
    
    raise RuntimeError(f"Claude API call failed after {retries} retries: {last_err}")
