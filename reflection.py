from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import re
import json
import difflib

from ..io_utils import extract_json
from ..openai_mm import call_openai_multimodal
from ..claude_mm import call_claude_multimodal
from ..gemini_mm import call_gemini_multimodal
from ..scoring import (
    normalize,
    _parse_triplet,
    contains_token,
    ASSERTIVE_CUES,
    UNCERTAINTY_TERMS,
    RISK_ANATOMY_TERMS,
)
from .experts import ChoiceMapper
from .tools import CompatConstraintsTool, risk_terms_text, PhasePriorTool, RAGTool


def _dedup_preserve(items: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items or []:
        nx = normalize(x)
        if not nx or nx in seen:
            continue
        seen.add(nx)
        out.append(x)
    return out


def _sentence_split(text: str) -> List[str]:
    if not text:
        return []
    # keep line breaks as separators too
    parts = re.split(r"(?<=[\.\!\?\n])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _has_uncertainty(s: str) -> bool:
    s = (s or "").lower()
    return any(u in s for u in UNCERTAINTY_TERMS)


def _has_assertive(s: str) -> bool:
    s = (s or "").lower()
    return any(a in s for a in ASSERTIVE_CUES)


def _soften_assertive_cues(s: str) -> str:
    out = s
    rep = {
        "clearly": "",
        "demonstrates": "suggests",
        "shows": "suggests",
        "we see": "we may see",
        "there is": "there may be",
        "visible": "possibly visible",
        "identified": "suggested",
        "seen": "possibly seen",
    }
    low = out.lower()
    for k, v in rep.items():
        # case-insensitive replace
        out = re.sub(re.escape(k), v, out, flags=re.IGNORECASE)
    # collapse whitespace
    out = re.sub(r"\s+", " ", out).strip()
    return out


def sanitize_report(
    report_text: str,
    canonical_ivt: Sequence[str],
    vocab_i: Sequence[str],
    vocab_v: Sequence[str],
    vocab_t: Sequence[str],
) -> str:
    """
    Deterministic discouragement of major errors under the v6 heuristics:
    - avoid assertive cues when mentioning tokens not supported by canonical_ivt
    - avoid assertive mention of risky anatomy terms not in supported targets
    - ensure 3 numbered sections exist (fallback template)
    """
    report_text = (report_text or "").strip()

    # Build support sets from canonical_ivt
    supported_i = set()
    supported_v = set()
    supported_t = set()
    for x in (canonical_ivt or []):
        i, v, t = _parse_triplet(x)
        if i: supported_i.add(i)
        if v: supported_v.add(v)
        if t: supported_t.add(t)
    supported_all = supported_i | supported_v | supported_t

    vocab_all = list(vocab_i or []) + list(vocab_v or []) + list(vocab_t or [])

    if not report_text:
        report_text = ""

    # sentence-level sanitize
    sanitized: List[str] = []
    for sent in _sentence_split(report_text):
        s_low = sent.lower()

        # risky anatomy term with assertive cue and no uncertainty and unsupported
        risky_hit = False
        for term in RISK_ANATOMY_TERMS:
            if contains_token(sent, term) and normalize(term) not in supported_t:
                risky_hit = True
                break

        if risky_hit and _has_assertive(sent) and (not _has_uncertainty(sent)):
            # drop the sentence entirely (too risky)
            continue

        # if assertive sentence mentions any unsupported vocab token, soften
        if _has_assertive(sent) and (not _has_uncertainty(sent)):
            unsupported_mentioned = False
            for tok in vocab_all:
                nt = normalize(tok)
                if nt in supported_all:
                    continue
                if contains_token(sent, tok):
                    unsupported_mentioned = True
                    break
            if unsupported_mentioned:
                sent2 = _soften_assertive_cues(sent)
                # ensure uncertainty word exists
                if not _has_uncertainty(sent2):
                    sent2 = "Possibly, " + sent2[0].lower() + sent2[1:] if sent2 else sent2
                sanitized.append(sent2)
                continue

        sanitized.append(sent)

    report_text2 = "\n".join(sanitized).strip()

    # Ensure 3 numbered sections. If missing, create a safe template.
    def has_section(n: int, txt: str) -> bool:
        return bool(re.search(rf"(^|\n)\s*{n}[\)\.\:]", txt.strip()))

    if not (has_section(1, report_text2) and has_section(2, report_text2) and has_section(3, report_text2)):
        ivt_lines = "; ".join([normalize(x) for x in (canonical_ivt or [])]) if canonical_ivt else "No confident IVT identified."
        report_text2 = (
            f"1) IVT events: {ivt_lines}\n"
            f"2) Anatomy/risk: Findings are described conservatively based on the visible field; avoid over-asserting specific ducts/arteries unless clearly supported.\n"
            f"3) Next-step intent: Continue with the next appropriate step consistent with the current phase and observed maneuvers, maintaining safe exposure and hemostasis.\n"
        )

    return report_text2


def enforce_cross_task_consistency(
    i_sel: List[str],
    v_sel: List[str],
    t_sel: List[str],
    ivt_sel: List[str],
    i_choices: Sequence[Any],
    v_choices: Sequence[Any],
    t_choices: Sequence[Any],
) -> Tuple[List[str], List[str], List[str]]:
    """
    Ensure that any token appearing in ivt is included in the corresponding i/v/t selections.
    This is safe because the tokens are already in the respective choice lists.
    """
    i_map = ChoiceMapper(i_choices)
    v_map = ChoiceMapper(v_choices)
    t_map = ChoiceMapper(t_choices)

    i_norm = {normalize(x) for x in (i_sel or [])}
    v_norm = {normalize(x) for x in (v_sel or [])}
    t_norm = {normalize(x) for x in (t_sel or [])}

    add_i: List[str] = []
    add_v: List[str] = []
    add_t: List[str] = []
    for trip in (ivt_sel or []):
        inst, verb, targ = _parse_triplet(trip)
        if inst and inst not in i_norm:
            # map back to exact string choice if possible
            mapped = i_map.map_string_list([inst])
            add_i += mapped
        if verb and verb not in v_norm:
            mapped = v_map.map_string_list([verb])
            add_v += mapped
        if targ and targ not in t_norm:
            mapped = t_map.map_string_list([targ])
            add_t += mapped

    return _dedup_preserve(list(i_sel or []) + add_i), _dedup_preserve(list(v_sel or []) + add_v), _dedup_preserve(list(t_sel or []) + add_t)


def repair_ivt_with_constraints(
    ivt_sel: List[str],
    ivt_choices: Sequence[str],
    compat: Optional[CompatConstraintsTool],
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Deterministically enforce:
    - in-choice exactness (ivt_sel already mapped, but re-check)
    - tool-action compatibility
    Repair strategy for mismatches:
    - try to find an alternative triplet among choices with same instrument+target but allowed verb
    - otherwise drop the mismatching triplet
    """
    dbg: Dict[str, Any] = {"dropped": [], "repaired": []}
    allowed_set = {normalize(x) for x in (ivt_choices or []) if str(x).strip()}
    choice_by_norm = {normalize(x): str(x) for x in (ivt_choices or []) if str(x).strip()}

    out: List[str] = []
    for trip in (ivt_sel or []):
        n = normalize(trip)
        if n not in allowed_set:
            # drop unknown
            dbg["dropped"].append({"triplet": trip, "reason": "not_in_choices"})
            continue

        if compat and compat.enabled and (not compat.is_triplet_compatible(trip)):
            i, v, t = _parse_triplet(trip)
            fixed = None
            if i and t:
                # find any choice with same i,t and allowed verb
                for cand_norm, cand_str in choice_by_norm.items():
                    ci, cv, ct = _parse_triplet(cand_norm)
                    if ci == i and ct == t and compat.is_triplet_compatible(cand_str):
                        fixed = cand_str
                        break
            if fixed:
                out.append(fixed)
                dbg["repaired"].append({"from": trip, "to": fixed, "reason": "tool_action_compat"})
            else:
                dbg["dropped"].append({"triplet": trip, "reason": "tool_action_mismatch_no_fix"})
            continue

        out.append(choice_by_norm[n])

    out = _dedup_preserve(out)
    return out, dbg


def model_repair(
    *,
    model: str,
    image_data_urls: List[str],
    api_key: Optional[str],
    base_url: Optional[str],
    temperature: float,
    max_output_tokens: int,
    tasks: Dict[str, Any],
    current: Dict[str, Any],
    issues: List[str],
    extra_hints: str = "",
) -> Dict[str, Any]:
    """
    One-shot model-based repair. Keeps outputs within the original choice spaces.
    Designed for ablation (can be disabled).
    """
    i_choices = (tasks.get("i_mcq", {}) or {}).get("choices", []) or []
    v_choices = (tasks.get("v_mcq", {}) or {}).get("choices", []) or []
    t_choices = (tasks.get("t_mcq", {}) or {}).get("choices", []) or []
    ivt_choices = (tasks.get("ivt_mcq", {}) or {}).get("choices", []) or []
    p_choices = (tasks.get("phase_mcq", {}) or {}).get("choices", []) or []

    def fmt(choices: Sequence[Any]) -> str:
        if not choices:
            return "[]"
        if choices and isinstance(choices[0], dict):
            return "\n".join([f"- {c.get('id')}: {c.get('name')}" for c in choices])
        return "\n".join([f"- {c}" for c in choices])

    prompt = (
        "You are a strict surgical-video assistant. You will be given THREE frames; the 3rd is the target.\n"
        "Your job: REPAIR an existing prediction to satisfy constraints and remove inconsistencies.\n"
        "Use ONLY visual evidence. If uncertain, be conservative (omit rather than guess).\n"
        "Return ONE JSON object ONLY. No markdown.\n\n"
        "Detected issues:\n"
        + "\n".join([f"- {x}" for x in issues])
        + "\n\n"
        + (extra_hints.strip() + "\n\n" if extra_hints.strip() else "")
        + "Current prediction (may contain mistakes):\n"
        + json.dumps(current, ensure_ascii=False)
        + "\n\n"
        + "You MUST select ONLY from choices.\n\n"
        + "TASK1 Instrument choices:\n" + fmt(i_choices) + "\n\n"
        + "TASK2 Verb choices:\n" + fmt(v_choices) + "\n\n"
        + "TASK3 Target choices:\n" + fmt(t_choices) + "\n\n"
        + "TASK4 IVT choices:\n" + fmt(ivt_choices) + "\n\n"
        + "TASK5 Phase choices:\n" + fmt(p_choices) + "\n\n"
        + "Return EXACT schema (no extra keys):\n"
        + "{\n"
        + "  \"i_mcq\": {\"selected\": [\"...\"]},\n"
        + "  \"v_mcq\": {\"selected\": [\"...\"]},\n"
        + "  \"t_mcq\": {\"selected\": [\"...\"]},\n"
        + "  \"ivt_mcq\": {\"selected\": [\"...\"]},\n"
        + "  \"phase_mcq\": {\"selected\": {\"id\": 0, \"name\": \"preparation\"}},\n"
        + "  \"report_task\": {\"report_text\": \"...\"}\n"
        + "}\n"
    )

    if "claude" in model.lower():
        raw = call_claude_multimodal(
            model=model,
            prompt=prompt,
            image_data_urls=image_data_urls,
            api_key=api_key,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
    elif "gemini" in model.lower():
        raw = call_gemini_multimodal(
            model=model,
            prompt=prompt,
            image_data_urls=image_data_urls,
            api_key=api_key,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
    else:
        raw = call_openai_multimodal(
            model=model,
            prompt=prompt,
            image_data_urls=image_data_urls,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
    return extract_json(raw)
