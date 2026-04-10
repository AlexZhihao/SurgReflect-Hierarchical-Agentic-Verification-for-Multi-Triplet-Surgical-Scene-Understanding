from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import json
import difflib

from ..io_utils import extract_json
from ..scoring import normalize, _parse_triplet
from ..openai_mm import call_openai_multimodal
from ..claude_mm import call_claude_multimodal
from ..gemini_mm import call_gemini_multimodal
from .tools import RAGTool, CompatConstraintsTool, PhasePriorTool, risk_terms_text
from ..prompts import get_cot_prompt


# ----------------------------
# Choice mapping utilities
# ----------------------------

class ChoiceMapper:
    def __init__(self, choices: Sequence[Any]) -> None:
        self.choices_raw = list(choices or [])
        self.norm_to_choice: Dict[str, Any] = {}
        for c in self.choices_raw:
            if isinstance(c, dict):
                # phase option: {"id":..,"name":..}
                try:
                    key = f'{int(c.get("id"))}:{normalize(str(c.get("name","")))}'
                    self.norm_to_choice[key] = c
                except Exception:
                    pass
                # also index by name only
                self.norm_to_choice[normalize(str(c.get("name","")))] = c
            else:
                self.norm_to_choice[normalize(str(c))] = str(c)

        self.norm_keys = list(self.norm_to_choice.keys())

    def map_string_list(self, items: Any) -> List[str]:
        if items is None:
            return []
        if isinstance(items, str):
            items = [items]
        if not isinstance(items, list):
            return []
        out: List[str] = []
        for it in items:
            s = str(it).strip()
            if not s:
                continue
            ns = normalize(s)
            if ns in self.norm_to_choice and isinstance(self.norm_to_choice[ns], str):
                out.append(self.norm_to_choice[ns])
                continue
            # fuzzy match
            m = difflib.get_close_matches(ns, self.norm_keys, n=1, cutoff=0.86)
            if m:
                v = self.norm_to_choice[m[0]]
                if isinstance(v, str):
                    out.append(v)
        # de-dup while preserving order
        seen = set()
        final: List[str] = []
        for x in out:
            nx = normalize(x)
            if nx in seen:
                continue
            seen.add(nx)
            final.append(x)
        return final

    def map_phase_obj(self, obj: Any) -> Optional[Dict[str, Any]]:
        if obj is None:
            return None
        # if already exact dict with id,name, try normalize
        if isinstance(obj, dict):
            pid = obj.get("id", None)
            name = obj.get("name", None)
            if pid is not None:
                try:
                    pid_i = int(pid)
                    key = f"{pid_i}:{normalize(str(name or ''))}"
                    if key in self.norm_to_choice and isinstance(self.norm_to_choice[key], dict):
                        return self.norm_to_choice[key]
                    # fallback by id only
                    for c in self.choices_raw:
                        if isinstance(c, dict) and int(c.get("id", -999)) == pid_i:
                            return c
                except Exception:
                    pass
            if name:
                nn = normalize(str(name))
                v = self.norm_to_choice.get(nn, None)
                if isinstance(v, dict):
                    return v

        # if id provided as int/string
        try:
            pid_i = int(obj)
            for c in self.choices_raw:
                if isinstance(c, dict) and int(c.get("id", -999)) == pid_i:
                    return c
        except Exception:
            pass
        return None


def _format_choices(choices: List[Any]) -> str:
    if not choices:
        return "[]"
    if isinstance(choices[0], dict):
        return "\n".join([f"- {c.get('id')}: {c.get('name')}" for c in choices])
    return "\n".join([f"- {c}" for c in choices])


def _base_header() -> str:
    return (
        "You are an expert surgical-video assistant for laparoscopic cholecystectomy.\n"
        "You will be given THREE consecutive frames. The 3rd frame is the target frame; the first two provide temporal context.\n"
        "Use ONLY visual evidence from the frames. If uncertain, be conservative.\n"
        "Return ONE JSON object ONLY. No markdown.\n"
    )


def _call_json(
    model: str,
    prompt: str,
    image_data_urls: List[str],
    api_key: Optional[str],
    base_url: Optional[str],
    temperature: float,
    max_output_tokens: int,
) -> Dict[str, Any]:
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


def _vote_union(list_of_lists: List[List[str]], min_votes: int) -> List[str]:
    counts: Dict[str, int] = {}
    canonical: Dict[str, str] = {}
    for lst in list_of_lists:
        for x in lst:
            nx = normalize(x)
            counts[nx] = counts.get(nx, 0) + 1
            canonical[nx] = x
    # keep items >= min_votes sorted by votes desc
    items = [(nx, c) for nx, c in counts.items() if c >= min_votes]
    items.sort(key=lambda t: (-t[1], t[0]))
    return [canonical[nx] for nx, _c in items]


def _vote_mode(objs: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not objs:
        return None
    counts: Dict[str, int] = {}
    canonical: Dict[str, Dict[str, Any]] = {}
    for o in objs:
        try:
            key = f"{int(o.get('id'))}:{normalize(str(o.get('name','')))}"
        except Exception:
            key = normalize(str(o))
        counts[key] = counts.get(key, 0) + 1
        canonical[key] = o
    best = sorted(counts.items(), key=lambda x: (-x[1], x[0]))[0][0]
    return canonical[best]


# ----------------------------
# Expert outputs
# ----------------------------

@dataclass
class MultiLabelPred:
    selected: List[str]
    raw: Dict[str, Any]


@dataclass
class PhasePred:
    selected: Optional[Dict[str, Any]]
    raw: Dict[str, Any]


@dataclass
class ReportPred:
    report_text: str
    raw: Dict[str, Any]


# ----------------------------
# Experts
# ----------------------------

class InstrumentExpert:
    def __init__(self, rag: Optional[RAGTool] = None) -> None:
        self.rag = rag

    def run(self, *, model: str, tasks: Dict[str, Any], image_data_urls: List[str], api_key: Optional[str], base_url: Optional[str],
            temperature: float, max_output_tokens: int, k: int, use_cot: bool = False) -> MultiLabelPred:
        choices = (tasks.get("i_mcq", {}) or {}).get("choices", []) or []
        mapper = ChoiceMapper(choices)

        cot_block = get_cot_prompt("instrument") if use_cot else ""
        reasoning_field = '  "reasoning": "<your step-by-step reasoning>",\n' if use_cot else ""

        prompt_core = (
            _base_header()
            + "TASK: Identify instruments visible in the 3rd frame (multi-select).\n"
            + "Select ONLY from the provided choices. If uncertain, do NOT guess.\n\n"
            + cot_block
            + "Choices:\n"
            + _format_choices(list(choices))
            + "\n\nReturn JSON schema:\n"
            + "{\n" + reasoning_field + "  \"selected\": [\"<choice>\", ...],\n  \"uncertain\": [\"<choice>\", ...]\n}\n"
        )

        # Optional dataset priors
        if self.rag and self.rag.enabled:
            prior = self.rag.retrieve_text(query="instrument " + " ".join([str(c) for c in choices])[:2000], topk=3, tag_filter="instrument_prior")
            if prior:
                prompt_core = prompt_core + "\n" + prior + "\n"

        runs: List[List[str]] = []
        last_raw: Dict[str, Any] = {}
        for _ in range(max(1, int(k))):
            obj = _call_json(model=model, prompt=prompt_core, image_data_urls=image_data_urls, api_key=api_key, base_url=base_url,
                             temperature=temperature, max_output_tokens=max_output_tokens)
            last_raw = obj
            sel = mapper.map_string_list(obj.get("selected", []))
            runs.append(sel)

        min_votes = (len(runs) // 2) + 1
        selected = _vote_union(runs, min_votes=min_votes) if len(runs) > 1 else (runs[0] if runs else [])
        return MultiLabelPred(selected=selected, raw=last_raw)


class VerbExpert:
    def __init__(self, rag: Optional[RAGTool] = None) -> None:
        self.rag = rag

    def run(self, *, model: str, tasks: Dict[str, Any], image_data_urls: List[str], api_key: Optional[str], base_url: Optional[str],
            temperature: float, max_output_tokens: int, k: int, instrument_selected: Sequence[str], use_cot: bool = False) -> MultiLabelPred:
        choices = (tasks.get("v_mcq", {}) or {}).get("choices", []) or []
        mapper = ChoiceMapper(choices)

        cot_block = get_cot_prompt("verb") if use_cot else ""
        reasoning_field = '  "reasoning": "<your step-by-step reasoning>",\n' if use_cot else ""

        prompt_core = (
            _base_header()
            + "TASK: Identify actions (verbs) occurring in the 3rd frame (multi-select).\n"
            + "Use temporal context from frames 1-2 ONLY to disambiguate motion.\n"
            + "Select ONLY from the provided choices. If uncertain, do NOT guess.\n\n"
            + cot_block
            + "Instrument hints (from another module): " + ", ".join([str(x) for x in (instrument_selected or [])]) + "\n\n"
            + "Choices:\n"
            + _format_choices(list(choices))
            + "\n\nReturn JSON schema:\n"
            + "{\n" + reasoning_field + "  \"selected\": [\"<choice>\", ...],\n  \"uncertain\": [\"<choice>\", ...]\n}\n"
        )

        if self.rag and self.rag.enabled and instrument_selected:
            prior = self.rag.retrieve_text(query=" ".join([str(x) for x in instrument_selected]) + " verb", topk=4)
            if prior:
                prompt_core = prompt_core + "\n" + prior + "\n"

        runs: List[List[str]] = []
        last_raw: Dict[str, Any] = {}
        for _ in range(max(1, int(k))):
            obj = _call_json(model=model, prompt=prompt_core, image_data_urls=image_data_urls, api_key=api_key, base_url=base_url,
                             temperature=temperature, max_output_tokens=max_output_tokens)
            last_raw = obj
            sel = mapper.map_string_list(obj.get("selected", []))
            runs.append(sel)

        min_votes = (len(runs) // 2) + 1
        selected = _vote_union(runs, min_votes=min_votes) if len(runs) > 1 else (runs[0] if runs else [])
        return MultiLabelPred(selected=selected, raw=last_raw)


class TargetExpert:
    def __init__(self, rag: Optional[RAGTool] = None) -> None:
        self.rag = rag

    def run(self, *, model: str, tasks: Dict[str, Any], image_data_urls: List[str], api_key: Optional[str], base_url: Optional[str],
            temperature: float, max_output_tokens: int, k: int, instrument_selected: Sequence[str], verb_selected: Sequence[str], use_cot: bool = False) -> MultiLabelPred:
        choices = (tasks.get("t_mcq", {}) or {}).get("choices", []) or []
        mapper = ChoiceMapper(choices)

        cot_block = get_cot_prompt("target") if use_cot else ""
        reasoning_field = '  "reasoning": "<your step-by-step reasoning>",\n' if use_cot else ""

        prompt_core = (
            _base_header()
            + "TASK: Identify anatomical targets involved in the 3rd frame (multi-select).\n"
            + "Select ONLY from provided choices. If uncertain, do NOT guess.\n\n"
            + cot_block
            + "Instrument hints: " + ", ".join([str(x) for x in (instrument_selected or [])]) + "\n"
            + "Verb hints: " + ", ".join([str(x) for x in (verb_selected or [])]) + "\n\n"
            + "Choices:\n"
            + _format_choices(list(choices))
            + "\n\nReturn JSON schema:\n"
            + "{\n" + reasoning_field + "  \"selected\": [\"<choice>\", ...],\n  \"uncertain\": [\"<choice>\", ...]\n}\n"
        )

        if self.rag and self.rag.enabled and (verb_selected or instrument_selected):
            prior = self.rag.retrieve_text(query=" ".join([str(x) for x in (verb_selected or [])]) + " " + " ".join([str(x) for x in (instrument_selected or [])]) + " target", topk=4)
            if prior:
                prompt_core = prompt_core + "\n" + prior + "\n"

        runs: List[List[str]] = []
        last_raw: Dict[str, Any] = {}
        for _ in range(max(1, int(k))):
            obj = _call_json(model=model, prompt=prompt_core, image_data_urls=image_data_urls, api_key=api_key, base_url=base_url,
                             temperature=temperature, max_output_tokens=max_output_tokens)
            last_raw = obj
            sel = mapper.map_string_list(obj.get("selected", []))
            runs.append(sel)

        min_votes = (len(runs) // 2) + 1
        selected = _vote_union(runs, min_votes=min_votes) if len(runs) > 1 else (runs[0] if runs else [])
        return MultiLabelPred(selected=selected, raw=last_raw)


class IVTExpert:
    def __init__(self, rag: Optional[RAGTool] = None, compat: Optional[CompatConstraintsTool] = None) -> None:
        self.rag = rag
        self.compat = compat

    def run(self, *, model: str, tasks: Dict[str, Any], image_data_urls: List[str], api_key: Optional[str], base_url: Optional[str],
            temperature: float, max_output_tokens: int, k: int,
            instrument_selected: Sequence[str], verb_selected: Sequence[str], target_selected: Sequence[str], use_cot: bool = False) -> MultiLabelPred:
        choices = (tasks.get("ivt_mcq", {}) or {}).get("choices", []) or []
        mapper = ChoiceMapper(choices)

        constraints = ""
        if self.compat and self.compat.enabled:
            constraints = self.compat.ivt_constraints_text()

        cot_block = get_cot_prompt("ivt") if use_cot else ""
        reasoning_field = '  "reasoning": "<your step-by-step reasoning>",\n' if use_cot else ""

        prompt_core = (
            _base_header()
            + "TASK: Select ALL IVT triplets (instrument, verb, target) that occur in the 3rd frame (multi-select).\n"
            + "You MUST select ONLY from the provided IVT choices (exact strings).\n"
            + "The 3rd frame is primary; frames 1-2 are only for temporal context.\n"
            + "Be conservative: do not add triplets without clear visual support.\n\n"
            + cot_block
            + (constraints + "\n\n" if constraints else "")
            + "Hints from other modules (may be incomplete):\n"
            + "- instruments: " + ", ".join([str(x) for x in (instrument_selected or [])]) + "\n"
            + "- verbs: " + ", ".join([str(x) for x in (verb_selected or [])]) + "\n"
            + "- targets: " + ", ".join([str(x) for x in (target_selected or [])]) + "\n\n"
            + "IVT Choices:\n"
            + _format_choices(list(choices))
            + "\n\nReturn JSON schema:\n"
            + "{\n" + reasoning_field + "  \"selected\": [\"<one or more IVT choices>\", ...],\n  \"rationale\": \"<= 80 words\"\n}\n"
        )

        if self.rag and self.rag.enabled:
            q = " ".join([str(x) for x in (instrument_selected or []) + list(verb_selected or []) + list(target_selected or [])])[:1800]
            prior = self.rag.retrieve_text(query=q, topk=5)
            if prior:
                prompt_core = prompt_core + "\n" + prior + "\n"

        runs: List[List[str]] = []
        last_raw: Dict[str, Any] = {}
        for _ in range(max(1, int(k))):
            obj = _call_json(model=model, prompt=prompt_core, image_data_urls=image_data_urls, api_key=api_key, base_url=base_url,
                             temperature=temperature, max_output_tokens=max_output_tokens)
            last_raw = obj
            sel = mapper.map_string_list(obj.get("selected", []))
            runs.append(sel)

        min_votes = (len(runs) // 2) + 1
        selected = _vote_union(runs, min_votes=min_votes) if len(runs) > 1 else (runs[0] if runs else [])
        return MultiLabelPred(selected=selected, raw=last_raw)


class PhaseExpert:
    def __init__(self, rag: Optional[RAGTool] = None, phase_prior: Optional[PhasePriorTool] = None) -> None:
        self.rag = rag
        self.phase_prior = phase_prior

    def run(self, *, model: str, tasks: Dict[str, Any], image_data_urls: List[str], api_key: Optional[str], base_url: Optional[str],
            temperature: float, max_output_tokens: int, k: int,
            predicted_ivt: Sequence[str], instrument_selected: Sequence[str], verb_selected: Sequence[str], target_selected: Sequence[str], use_cot: bool = False) -> PhasePred:
        choices = (tasks.get("phase_mcq", {}) or {}).get("choices", []) or []
        mapper = ChoiceMapper(choices)

        prior_txt = ""
        if self.phase_prior and self.phase_prior.enabled and predicted_ivt:
            prior_txt = self.phase_prior.phase_prior_text(predicted_ivt)

        cot_block = get_cot_prompt("phase") if use_cot else ""
        reasoning_field = '  "reasoning": "<your step-by-step reasoning>",\n' if use_cot else ""

        prompt_core = (
            _base_header()
            + "TASK: Recognize the surgical phase for the 3rd frame. Select EXACTLY ONE option.\n"
            + "Use the IVT and temporal cues as hints, but base the decision on visual evidence.\n\n"
            + cot_block
            + "Hints:\n"
            + "- IVT: " + ", ".join([str(x) for x in (predicted_ivt or [])]) + "\n"
            + "- instruments: " + ", ".join([str(x) for x in (instrument_selected or [])]) + "\n"
            + "- verbs: " + ", ".join([str(x) for x in (verb_selected or [])]) + "\n"
            + "- targets: " + ", ".join([str(x) for x in (target_selected or [])]) + "\n\n"
            + (prior_txt + "\n\n" if prior_txt else "")
            + "Phase options (id:name):\n"
            + _format_choices(list(choices))
            + "\n\nReturn JSON schema:\n"
            + "{\n" + reasoning_field + "  \"selected\": {\"id\": 0, \"name\": \"preparation\"},\n  \"why\": \"<= 60 words\"\n}\n"
        )

        runs: List[Optional[Dict[str, Any]]] = []
        last_raw: Dict[str, Any] = {}
        for _ in range(max(1, int(k))):
            obj = _call_json(model=model, prompt=prompt_core, image_data_urls=image_data_urls, api_key=api_key, base_url=base_url,
                             temperature=temperature, max_output_tokens=max_output_tokens)
            last_raw = obj
            sel = mapper.map_phase_obj(obj.get("selected", None))
            runs.append(sel)

        selected = _vote_mode([x for x in runs if isinstance(x, dict)]) if len(runs) > 1 else (runs[0] if runs else None)
        return PhasePred(selected=selected if isinstance(selected, dict) else None, raw=last_raw)


class ReportExpert:
    def __init__(self, rag: Optional[RAGTool] = None) -> None:
        self.rag = rag

    def run(self, *, model: str, tasks: Dict[str, Any], image_data_urls: List[str], api_key: Optional[str], base_url: Optional[str],
            temperature: float, max_output_tokens: int, k: int,
            canonical_ivt: Sequence[str], canonical_phase: Optional[Dict[str, Any]], use_cot: bool = False) -> ReportPred:

        phase_str = ""
        if isinstance(canonical_phase, dict):
            phase_str = f"{canonical_phase.get('id')}:{canonical_phase.get('name')}"
        elif canonical_phase:
            phase_str = str(canonical_phase)

        cot_block = get_cot_prompt("report") if use_cot else ""

        prompt_core = (
            _base_header()
            + "TASK: Write a short intraoperative event report for the given 3 frames.\n"
            + "Write exactly 3 numbered sections:\n"
            + "1) IVT events (use the provided canonical IVT list)\n"
            + "2) anatomy/risk (be conservative; avoid naming risky ducts/arteries unless supported)\n"
            + "3) next-step intent (must be consistent with the phase and events)\n\n"
            + cot_block
            + risk_terms_text() + "\n\n"
            + "Canonical IVT (self-declared; do not add new ones):\n"
            + "\n".join([f"- {x}" for x in (canonical_ivt or [])]) + "\n\n"
            + f"Canonical phase: {phase_str}\n\n"
            + "Return JSON schema:\n"
            + "{\n  \"report_text\": \"<3 numbered sections, concise>\",\n  \"notes\": \"<= 50 words\"\n}\n"
        )

        if self.rag and self.rag.enabled and canonical_ivt:
            prior = self.rag.retrieve_text(query=" ".join([str(x) for x in canonical_ivt])[:1800], topk=3, tag_filter="phase_prior")
            if prior:
                prompt_core = prompt_core + "\n" + prior + "\n"

        texts: List[str] = []
        last_raw: Dict[str, Any] = {}
        for _ in range(max(1, int(k))):
            obj = _call_json(model=model, prompt=prompt_core, image_data_urls=image_data_urls, api_key=api_key, base_url=base_url,
                             temperature=temperature, max_output_tokens=max_output_tokens)
            last_raw = obj
            txt = str(obj.get("report_text", "") or "").strip()
            if txt:
                texts.append(txt)

        report_text = texts[0] if texts else ""
        return ReportPred(report_text=report_text, raw=last_raw)


# ----------------------------
# Joint / verification experts
# ----------------------------


@dataclass
class JointPred:
    i_selected: List[str]
    v_selected: List[str]
    t_selected: List[str]
    ivt_selected: List[str]
    phase_selected: Optional[Dict[str, Any]]
    report_text: str
    raw: Dict[str, Any]


class JointPredictExpert:
    """Single-call multitask predictor (baseline-style), used as the default first stage.

    Motivation: the legacy per-task experts were intentionally conservative and
    tended to under-predict verbs/targets/IVTs, hurting micro-F1.
    """

    def __init__(
        self,
        *,
        rag: Optional[RAGTool] = None,
        compat: Optional[CompatConstraintsTool] = None,
        phase_prior: Optional[PhasePriorTool] = None,
    ) -> None:
        self.rag = rag
        self.compat = compat
        self.phase_prior = phase_prior

    def run(
        self,
        *,
        model: str,
        tasks: Dict[str, Any],
        image_data_urls: List[str],
        api_key: Optional[str],
        base_url: Optional[str],
        temperature: float,
        max_output_tokens: int,
        k: int,
        use_cot: bool = False,
    ) -> JointPred:
        from ..prompts import build_multitask_prompt

        prompt = build_multitask_prompt(tasks, use_cot=use_cot)

        # Optional hard constraints and dataset priors (purely hints; still requires visual evidence).
        extra_parts: List[str] = []
        if self.compat and self.compat.enabled:
            extra_parts.append(self.compat.ivt_constraints_text())
        if self.rag and self.rag.enabled:
            # Provide light priors: instrument->verb/target patterns help recall without adding external knowledge.
            try:
                i_choices = (tasks.get("i_mcq", {}) or {}).get("choices", []) or []
                v_choices = (tasks.get("v_mcq", {}) or {}).get("choices", []) or []
                t_choices = (tasks.get("t_mcq", {}) or {}).get("choices", []) or []
                q = " ".join([str(x) for x in (i_choices + v_choices + t_choices)])[:1800]
                prior = self.rag.retrieve_text(query=q, topk=4)
                if prior:
                    extra_parts.append("[DATASET PRIORS]\n" + prior)
            except Exception:
                pass
        if extra_parts:
            prompt = prompt + "\n\n" + "\n\n".join(extra_parts) + "\n"

        runs: List[Dict[str, Any]] = []
        last_raw: Dict[str, Any] = {}
        for _ in range(max(1, int(k))):
            obj = _call_json(
                model=model,
                prompt=prompt,
                image_data_urls=image_data_urls,
                api_key=api_key,
                base_url=base_url,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
            last_raw = obj
            runs.append(obj)

        # If k>1: do a soft union over predictions to improve recall.
        def _get_list(o: Dict[str, Any], key: str) -> List[str]:
            return list(((o.get(key, {}) or {}).get("selected", [])) or [])

        i_lists = [_get_list(o, "i_mcq") for o in runs]
        v_lists = [_get_list(o, "v_mcq") for o in runs]
        t_lists = [_get_list(o, "t_mcq") for o in runs]
        ivt_lists = [_get_list(o, "ivt_mcq") for o in runs]
        i_sel = _vote_union([list(map(str, x)) for x in i_lists], min_votes=1) if len(runs) > 1 else list(map(str, i_lists[0] or []))
        v_sel = _vote_union([list(map(str, x)) for x in v_lists], min_votes=1) if len(runs) > 1 else list(map(str, v_lists[0] or []))
        t_sel = _vote_union([list(map(str, x)) for x in t_lists], min_votes=1) if len(runs) > 1 else list(map(str, t_lists[0] or []))
        ivt_sel = _vote_union([list(map(str, x)) for x in ivt_lists], min_votes=1) if len(runs) > 1 else list(map(str, ivt_lists[0] or []))

        # Phase: majority vote if present
        phase_objs = []
        for o in runs:
            ph = ((o.get("phase_mcq", {}) or {}).get("selected", None))
            if isinstance(ph, dict) and "id" in ph:
                phase_objs.append(ph)
        phase_sel = _vote_mode(phase_objs) if len(phase_objs) > 1 else (phase_objs[0] if phase_objs else None)

        # Report text (unused by scoring rule-channel, but kept for debug)
        report_text = str(((last_raw.get("report_task", {}) or {}).get("report_text", "")) or "").strip()

        return JointPred(
            i_selected=i_sel,
            v_selected=v_sel,
            t_selected=t_sel,
            ivt_selected=ivt_sel,
            phase_selected=phase_sel,
            report_text=report_text,
            raw=last_raw,
        )


class IVTVerifyExpert:
    """Focused verification over a short IVT candidate list to reduce misses."""

    def __init__(self, *, compat: Optional[CompatConstraintsTool] = None) -> None:
        self.compat = compat

    def run(
        self,
        *,
        model: str,
        candidates: Sequence[str],
        image_data_urls: List[str],
        api_key: Optional[str],
        base_url: Optional[str],
        temperature: float,
        max_output_tokens: int,
        k: int,
    ) -> MultiLabelPred:

        cand_list = [str(x) for x in (candidates or []) if str(x).strip()]
        mapper = ChoiceMapper(cand_list)

        prompt = (
            _base_header()
            + "TASK: Verify which IVT triplets occur in the 3rd frame. The list below is a SHORTLIST of candidates.\n"
            + "Selection goal: HIGH RECALL with reasonable precision. It is better to include a plausible true event than to miss it.\n"
            + "However, do NOT select events that are clearly unsupported by the visuals.\n"
            + (self.compat.ivt_constraints_text() + "\n\n" if (self.compat and self.compat.enabled) else "")
            + "Candidates (select only from these exact strings):\n"
            + "\n".join([f"- {x}" for x in cand_list])
            + "\n\nReturn JSON schema:\n"
            + "{\n  \"selected\": [\"<candidate>\", ...],\n  \"maybe\": [\"<candidate>\", ...]\n}\n"
        )

        runs: List[List[str]] = []
        last_raw: Dict[str, Any] = {}
        for _ in range(max(1, int(k))):
            obj = _call_json(
                model=model,
                prompt=prompt,
                image_data_urls=image_data_urls,
                api_key=api_key,
                base_url=base_url,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
            last_raw = obj
            sel = mapper.map_string_list(obj.get("selected", []))
            may = mapper.map_string_list(obj.get("maybe", []))
            # Treat maybe as soft positives to reduce misses.
            runs.append(sel + [x for x in may if normalize(x) not in {normalize(y) for y in sel}])

        selected = _vote_union(runs, min_votes=1) if len(runs) > 1 else (runs[0] if runs else [])
        return MultiLabelPred(selected=selected, raw=last_raw)
