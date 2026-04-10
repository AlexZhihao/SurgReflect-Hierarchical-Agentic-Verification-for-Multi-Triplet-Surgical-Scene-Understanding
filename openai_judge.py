from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .gemini_judge import _build_prompt          # reuse the same judge prompt
from .io_utils import extract_json
from .openai_mm import call_openai_multimodal     # text-only call (no images)


@dataclass
class OpenAIJudge:
    """Judge that calls an OpenAI model (e.g. gpt-5.2) to score reports.

    Drop-in replacement for GeminiJudge — same `score()` / `score_once()` API.
    """
    model: str = "gpt-5.2"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    sleep_sec: float = 0.6
    retries: int = 3

    def score_once(
        self,
        report_text: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        prompt = _build_prompt(report_text, meta=meta)
        key = self.api_key or os.getenv("OPENAI_API_KEY")

        raw = call_openai_multimodal(
            model=self.model,
            prompt=prompt,
            image_data_urls=[],       # text-only — no images
            api_key=key,
            base_url=self.base_url,
            temperature=0.0,
            max_output_tokens=500,
            retries=1,                # we do retries ourselves
        )

        obj = extract_json(raw)

        def clip01_10(x: Any) -> int:
            try:
                v = int(round(float(x)))
                return max(0, min(v, 10))
            except Exception:
                return 0

        out = {
            "structure":         clip01_10(obj.get("structure", 0)),
            "conservativeness":  clip01_10(obj.get("conservativeness", 0)),
            "next_step":         clip01_10(obj.get("next_step", 0)),
            "professionalism":   clip01_10(obj.get("professionalism", 0)),
        }
        out["overall"] = round(sum(out.values()) / 4.0, 1)
        out["brief_reason"] = str(obj.get("brief_reason", ""))[:400]
        return out

    def score(
        self,
        report_text: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        last_err: Optional[Exception] = None
        for attempt in range(1, self.retries + 1):
            try:
                return self.score_once(report_text, meta=meta)
            except Exception as e:
                last_err = e
                time.sleep(self.sleep_sec * attempt)
        raise RuntimeError(
            f"OpenAI judge ({self.model}) failed after {self.retries} retries: {last_err}"
        )
