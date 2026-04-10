from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _build_prompt(report_text: str, meta: Optional[Dict[str, Any]] = None) -> str:
    """Build a judge prompt.

    Design goals:
    - keep the judge focused on report quality (structure / conservativeness / coherence / style)
    - avoid score saturation (10s everywhere) by giving clear anchors and caps
    - allow optional meta (canonical_ivt / canonical_phase) to judge whether the report
      is explicit and self-consistent, WITHOUT grading correctness versus ground truth.
    """

    meta_lines: str = ""
    if isinstance(meta, dict) and meta:
        civt = meta.get("canonical_ivt", [])
        cphase = meta.get("canonical_phase", None)
        try:
            civt_str = ", ".join([str(x) for x in (civt or [])])
        except Exception:
            civt_str = ""
        cphase_str = ""
        if isinstance(cphase, dict):
            cphase_str = str(cphase.get("name", ""))
        elif isinstance(cphase, str):
            cphase_str = cphase
        meta_lines = (
            "Self-declared canonical labels (NOT ground truth; do NOT grade correctness):\n"
            f"- canonical_ivt: {civt_str}\n"
            f"- canonical_phase: {cphase_str}\n\n"
        )

    # NOTE: Avoid nested triple-quotes inside an f-string.
    return (
        "You are a strict evaluator of a short intraoperative event report for laparoscopic cholecystectomy.\n\n"
        "IMPORTANT: Do NOT grade whether the IVT labels are correct versus any hidden ground truth.\n"
        "Evaluate ONLY writing quality and self-consistency. Use the full 0-10 scale; 9-10 should be rare.\n\n"
        "SCORING ANCHORS (apply these strictly):\n"
        "- 10: exceptional and rare; perfectly structured, concrete but conservative, next-step is well-justified, and very concise.\n"
        "- 7-8: good; clear structure and mostly conservative; minor verbosity or mild overreach.\n"
        "- 4-6: mediocre; generic, missing a section, unclear next-step, or noticeably verbose.\n"
        "- 0-3: poor; disorganized, speculative/hallucinated anatomy, or incoherent.\n\n"
        "HARD CAPS to prevent score inflation:\n"
        "- If any of the 3 numbered sections is missing, structure <= 6.\n"
        "- If the report contains confident claims about specific anatomy (e.g., cystic duct/artery/CBD) without clear visual justification, conservativeness <= 5.\n"
        "- If the next-step recommendation jumps ahead or contradicts the described events, next_step <= 5.\n"
        "- If the report is long, essay-like, or repetitive, professionalism <= 6.\n\n"
        + meta_lines
        + "Grade the following qualities:\n"
        "(1) Structure clarity (0-10): organized as (1) events (2) anatomy/risk (3) next-step intent; easy to read\n"
        "(2) Conservativeness (0-10): avoids over-confident unseen claims; cautious risk phrasing\n"
        "(3) Next-step coherence (0-10): next-step follows from described events/phase, no jumping\n"
        "(4) Professionalism (0-10): surgical tone, concise, no fluff\n\n"
        "Return ONLY a JSON object with this schema (numbers can be integers):\n"
        "{\n"
        "  \"structure\": 0,\n"
        "  \"conservativeness\": 0,\n"
        "  \"next_step\": 0,\n"
        "  \"professionalism\": 0,\n"
        "  \"overall\": 0,\n"
        "  \"brief_reason\": \"<= 60 words\"\n"
        "}\n\n"
        "Report to evaluate:\n"
        + (report_text or "")
        + "\n"
    )


@dataclass
class GeminiJudge:
    model: str = "gemini-2.5-pro"
    api_key: Optional[str] = None
    sleep_sec: float = 0.6
    retries: int = 3

    def _client(self):
        # Prefer the new google-genai SDK: from google import genai
        try:
            from google import genai  # type: ignore

            key = self.api_key or os.getenv("GEMINI_API_KEY")
            return ("google_genai", genai.Client(api_key=key) if key else genai.Client())
        except Exception:
            pass

        # Fallback to the older google.generativeai SDK
        import google.generativeai as genai2  # type: ignore

        key = self.api_key or os.getenv("GEMINI_API_KEY")
        if key:
            genai2.configure(api_key=key)
        return ("google_generativeai", genai2.GenerativeModel(self.model))

    def score_once(self, report_text: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        sdk, client_or_model = self._client()
        prompt = _build_prompt(report_text, meta=meta)

        if sdk == "google_genai":
            client = client_or_model
            resp = client.models.generate_content(model=self.model, contents=prompt)
            text = getattr(resp, "text", None) or ""
        else:
            model = client_or_model
            resp = model.generate_content(prompt)
            text = getattr(resp, "text", None) or ""

        from .io_utils import extract_json

        obj = extract_json(text)
        # normalize / clip
        def clip01_10(x: Any) -> int:
            try:
                v = int(round(float(x)))
                return max(0, min(v, 10))
            except Exception:
                return 0

        out = {
            "structure": clip01_10(obj.get("structure", 0)),
            "conservativeness": clip01_10(obj.get("conservativeness", 0)),
            "next_step": clip01_10(obj.get("next_step", 0)),
            "professionalism": clip01_10(obj.get("professionalism", 0)),
        }
        # IMPORTANT: To avoid inconsistency (e.g., judge "overall" not matching
        # the four dimensions), we ALWAYS compute overall as the mean of the
        # four rubric dimensions.
        out["overall"] = round(sum(out.values()) / 4.0, 1)
        out["brief_reason"] = str(obj.get("brief_reason", ""))[:400]
        return out

    def score(self, report_text: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        last_err: Optional[Exception] = None
        for attempt in range(1, self.retries + 1):
            try:
                return self.score_once(report_text, meta=meta)
            except Exception as e:
                last_err = e
                time.sleep(self.sleep_sec * attempt)
        raise RuntimeError(f"Gemini judge failed after {self.retries} retries: {last_err}")
