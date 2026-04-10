from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json(text: str) -> Dict[str, Any]:
    """Extract and parse a JSON object from model output."""
    t = text.strip()

    # remove markdown fences if any
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", t)
        t = re.sub(r"```\s*$", "", t).strip()

    # try direct
    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # regex fallback: take first {...}
    m = _JSON_RE.search(t)
    if not m:
        raise ValueError("No JSON object found in output.")

    candidate = m.group(0)
    # remove trailing junk after last }
    candidate = candidate[: candidate.rfind("}") + 1]

    return json.loads(candidate)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
