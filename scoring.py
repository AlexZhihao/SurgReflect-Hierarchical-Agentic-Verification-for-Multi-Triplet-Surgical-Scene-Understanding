from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


# =============================================================================
# v6 SCORING IMPLEMENTATION (one-to-one with spec)
# =============================================================================


def _strip_prefix(s: str) -> str:
    """Remove dataset-facing prefixes like 'Token:' / 'Triplet:'"""
    s = (s or "").strip()
    s = re.sub(r"^(Token|Triplet)\s*:\s*", "", s, flags=re.IGNORECASE)
    return s.strip()


def normalize(s: str) -> str:
    """Lower + normalize whitespace/punct for robust set matching."""
    s = _strip_prefix(str(s))
    s = s.strip().lower()
    s = s.replace("’", "'")
    s = re.sub(r"\s+", " ", s)
    # historical spelling variants in some surgical datasets
    s = re.sub(r"\bcalot\b", "carlot", s)
    s = re.sub(r"calot\s*'s", "carlot's", s)
    return s


def _token_variants(token: str) -> List[str]:
    t = normalize(token)
    variants = {t}
    variants.add(t.replace("_", " "))
    variants.add(t.replace("-", " "))
    variants.add(t.replace("_", "-"))
    if "gallbladder" in t:
        variants.add(t.replace("gallbladder", "gall bladder"))
    return sorted(variants, key=len, reverse=True)


def contains_token(text: str, token: str) -> bool:
    """Lightweight, reproducible token-matching used in gating/major-error checks."""
    tx = " " + normalize(text or "") + " "
    for v in _token_variants(token):
        base = re.escape(v)
        pat = r"(?<![a-z0-9])" + base + r"(?![a-z0-9])"
        if re.search(pat, tx):
            return True

        # light morphology for common verbs/nouns: clip/clipping, dissect/dissection
        if v.isalpha() and len(v) >= 3:
            morph = r"(?:s|es|ed|ing)?" if not v.endswith("e") else r"(?:s|d|ing)?"
            pat2 = r"(?<![a-z0-9])" + base + morph + r"(?![a-z0-9])"
            if re.search(pat2, tx):
                return True

        if v in {"dissect", "coagulate", "aspirate", "irrigate"}:
            noun_map = {
                "dissect": "dissection",
                "coagulate": "coagulation",
                "aspirate": "aspiration",
                "irrigate": "irrigation",
            }
            nv = noun_map.get(v)
            if nv:
                pat3 = r"(?<![a-z0-9])" + re.escape(nv) + r"(?![a-z0-9])"
                if re.search(pat3, tx):
                    return True
    return False


def _parse_triplet(s: str) -> Tuple[str, str, str]:
    s = normalize(s)
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        return "", "", ""
    return parts[0], parts[1], parts[2]


# =============================================================================
# 2. MCQ scoring (I/V/T/IVT/Phase)
# =============================================================================


@dataclass
class SetMetrics:
    tp: int
    fp: int
    fn: int
    f1: float
    coverage: float
    fp_rate: float
    fn_rate: float
    exact_set_match: float


def set_micro_f1_sample(pred: Sequence[str], gold: Sequence[str]) -> SetMetrics:
    """Per-sample Set Micro-F1 per v6 spec (multi-label tasks)."""
    P = {normalize(x) for x in (pred or []) if str(x).strip()}
    G = {normalize(x) for x in (gold or []) if str(x).strip()}

    if not P and not G:
        return SetMetrics(tp=0, fp=0, fn=0, f1=1.0, coverage=1.0, fp_rate=0.0, fn_rate=0.0, exact_set_match=1.0)
    if not G and P:
        tp = 0
        fp = len(P)
        fn = 0
        f1 = 0.0
    elif G and not P:
        tp = 0
        fp = 0
        fn = len(G)
        f1 = 0.0
    else:
        tp = len(P & G)
        fp = len(P - G)
        fn = len(G - P)
        denom = (2 * tp + fp + fn)
        f1 = (2 * tp / denom) if denom else 0.0

    coverage = tp / max(1, len(G))
    fp_rate = fp / max(1, len(P))
    fn_rate = fn / max(1, len(G))
    exact = 1.0 if P == G else 0.0
    return SetMetrics(tp=tp, fp=fp, fn=fn, f1=f1, coverage=coverage, fp_rate=fp_rate, fn_rate=fn_rate, exact_set_match=exact)


@dataclass
class SetAgg:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    sum_f1: float = 0.0
    sum_coverage: float = 0.0
    sum_fp_rate: float = 0.0
    sum_fn_rate: float = 0.0
    sum_exact: float = 0.0
    n: int = 0

    def add(self, m: SetMetrics) -> None:
        self.tp += int(m.tp)
        self.fp += int(m.fp)
        self.fn += int(m.fn)
        self.sum_f1 += float(m.f1)
        self.sum_coverage += float(m.coverage)
        self.sum_fp_rate += float(m.fp_rate)
        self.sum_fn_rate += float(m.fn_rate)
        self.sum_exact += float(m.exact_set_match)
        self.n += 1

    def micro_f1(self) -> float:
        denom = (2 * self.tp + self.fp + self.fn)
        if denom == 0:
            return 1.0
        return 2 * self.tp / denom

    def mean_f1(self) -> float:
        return self.sum_f1 / max(1, self.n)

    def exact_match_rate(self) -> float:
        return self.sum_exact / max(1, self.n)

    def profile_means(self) -> Dict[str, float]:
        return {
            "coverage": self.sum_coverage / max(1, self.n),
            "fp_rate": self.sum_fp_rate / max(1, self.n),
            "fn_rate": self.sum_fn_rate / max(1, self.n),
        }


def score_phase_single(pred_obj: Any, gt_list: Any) -> Tuple[float, int, int]:
    """Returns (acc, pred_id, gt_id)."""
    gt_id = None
    if isinstance(gt_list, list) and gt_list:
        if isinstance(gt_list[0], dict):
            gt_id = gt_list[0].get("id")
        else:
            try:
                gt_id = int(gt_list[0])
            except Exception:
                gt_id = None
    if gt_id is None:
        return 0.0, -1, -1

    pred_id = None
    if isinstance(pred_obj, dict):
        pred_id = pred_obj.get("id")
    else:
        try:
            pred_id = int(pred_obj)
        except Exception:
            pred_id = None
    if pred_id is None:
        pred_id = -1

    acc = 1.0 if int(pred_id) == int(gt_id) else 0.0
    return acc, int(pred_id), int(gt_id)


# =============================================================================
# 3. Report scoring (dual-channel + gating + major error)
# =============================================================================


RISK_ANATOMY_TERMS = [
    "cbd",
    "common bile duct",
    "bile duct",
    "cystic duct",
    "cystic artery",
    "hepatic artery",
]

UNCERTAINTY_TERMS = [
    "possible",
    "possibly",
    "may",
    "might",
    "could",
    "uncertain",
    "cannot confirm",
    "not confirmed",
    "suggest",
    "suggests",
    "likely",
]

ASSERTIVE_CUES = [
    "clearly",
    "identified",
    "we see",
    "seen",
    "visible",
    "shows",
    "demonstrates",
    "there is",
]


def _text_has_uncertainty_near(text: str, term: str, window_chars: int = 80) -> bool:
    t = (text or "")
    t_low = t.lower()
    idx = t_low.find(term.lower())
    if idx < 0:
        return False
    lo = max(0, idx - window_chars)
    hi = min(len(t_low), idx + len(term) + window_chars)
    win = t_low[lo:hi]
    return any(u in win for u in UNCERTAINTY_TERMS)


def _text_has_assertive_cue_near(text: str, term: str, window_chars: int = 80) -> bool:
    t = (text or "")
    t_low = t.lower()
    idx = t_low.find(term.lower())
    if idx < 0:
        return False
    lo = max(0, idx - window_chars)
    hi = min(len(t_low), idx + len(term) + window_chars)
    win = t_low[lo:hi]
    return any(c in win for c in ASSERTIVE_CUES)


def _extract_vocab_from_choices(task_choices: Sequence[Any]) -> List[str]:
    out: List[str] = []
    for c in (task_choices or []):
        if isinstance(c, str):
            tok = _strip_prefix(c)
            if tok:
                out.append(tok)
    # dedup but keep stable
    seen = set()
    uniq = []
    for x in out:
        nx = normalize(x)
        if nx in seen:
            continue
        seen.add(nx)
        uniq.append(x)
    return uniq


def extract_vocab_from_choices(task_choices: Sequence[Any]) -> List[str]:
    """Public wrapper for vocabulary extraction from i/v/t choices."""
    return _extract_vocab_from_choices(task_choices)


def _canonical_support_sets(canonical_ivt: Sequence[str]) -> Tuple[Set[str], Set[str], Set[str]]:
    inst, verb, targ = set(), set(), set()
    for x in (canonical_ivt or []):
        i, v, t = _parse_triplet(x)
        if i:
            inst.add(i)
        if v:
            verb.add(v)
        if t:
            targ.add(t)
    return inst, verb, targ


def report_rule_channel(
    gold_ivt: Sequence[str],
    canonical_ivt: Sequence[str],
    allowed_ivt_choices: Sequence[str],
    global_triplet_vocab: Set[str],
    enable_oov_penalty: bool = True,
    lambda_oov: float = 0.1,
) -> Tuple[float, Dict[str, Any]]:
    """Rule channel per v6 spec: ONLY appendix.canonical_ivt (+ optional OOV penalty)."""

    G = {normalize(x) for x in (gold_ivt or []) if str(x).strip()}
    P = {normalize(x) for x in (canonical_ivt or []) if str(x).strip()}

    tp = len(P & G)
    fp = len(P - G)
    fn = len(G - P)
    denom = (2 * tp + fp + fn)
    rule_f1 = (2 * tp / denom) if denom else (1.0 if (not P and not G) else 0.0)
    rule_coverage = tp / max(1, len(G))

    allowed = {normalize(x) for x in (allowed_ivt_choices or []) if str(x).strip()}
    oov_items = [x for x in P if (x not in allowed and x not in global_triplet_vocab)]
    oov_count = len(oov_items)

    rule_score = float(rule_f1)
    if enable_oov_penalty and oov_count > 0:
        rule_score = max(0.0, rule_score - float(lambda_oov) * oov_count)

    dbg = {
        "gold_ivt": sorted(G),
        "pred_ivt": sorted(P),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "rule_f1": rule_f1,
        "rule_coverage": rule_coverage,
        "oov_count": oov_count,
        "oov_items": sorted(oov_items),
        "enable_oov_penalty": enable_oov_penalty,
        "lambda_oov": lambda_oov,
    }
    return rule_score, dbg


def apply_llm_gating(
    report_text: str,
    llm_dims_01: Dict[str, float],
    canonical_ivt: Sequence[str],
    canonical_phase: Any,
    vocab_i: Sequence[str],
    vocab_v: Sequence[str],
    vocab_t: Sequence[str],
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """v6 gating: consistency cap + over-assertion penalty (post-processing)."""

    report_text = report_text or ""
    dims = dict(llm_dims_01)

    supported_i, supported_v, supported_t = _canonical_support_sets(canonical_ivt)
    supported_all = set(supported_i) | set(supported_v) | set(supported_t)

    vocab_all = list(vocab_i) + list(vocab_v) + list(vocab_t)
    mentioned = set()
    for tok in vocab_all:
        if tok and contains_token(report_text, tok):
            mentioned.add(normalize(tok))

    unsupported = {m for m in mentioned if m not in supported_all}

    # missing core tool/target mention
    miss_core = False
    if supported_i or supported_t:
        any_i = any(contains_token(report_text, x) for x in supported_i)
        any_t = any(contains_token(report_text, x) for x in supported_t)
        if (supported_i and not any_i) or (supported_t and not any_t):
            miss_core = True

    inconsistency = False
    if len(mentioned) >= 2 and len(unsupported) >= max(2, int(math.ceil(0.5 * len(mentioned)))):
        inconsistency = True
    if miss_core:
        inconsistency = True

    # (A) consistency cap
    if inconsistency:
        dims["structure"] = min(float(dims.get("structure", 0.0)), 0.6)
        dims["next_step"] = min(float(dims.get("next_step", 0.0)), 0.6)
        dims["professionalism"] = min(float(dims.get("professionalism", 0.0)), 0.6)

    # (B) over-assertion penalty on risky anatomy
    # if risk anatomy strongly asserted AND not supported by canonical targets
    canonical_targets = supported_t
    over_assertion = False
    for term in RISK_ANATOMY_TERMS:
        if contains_token(report_text, term):
            if normalize(term) not in canonical_targets and (not _text_has_uncertainty_near(report_text, term)):
                over_assertion = True
                break
    if over_assertion:
        dims["conservativeness"] = min(float(dims.get("conservativeness", 0.0)), 0.4)

    dbg = {
        "mentioned_tokens": sorted(mentioned),
        "unsupported_tokens": sorted(unsupported),
        "miss_core": miss_core,
        "inconsistency": inconsistency,
        "over_assertion": over_assertion,
    }
    return dims, dbg


def detect_major_errors(
    report_text: str,
    canonical_ivt: Sequence[str],
    vocab_i: Sequence[str],
    vocab_v: Sequence[str],
    vocab_t: Sequence[str],
) -> Dict[str, Any]:
    """Major errors per v6 spec (reproducible heuristics)."""
    report_text = report_text or ""
    supported_i, supported_v, supported_t = _canonical_support_sets(canonical_ivt)

    # 1) major anatomy hallucination: strong assertion of risky structure not supported by canonical targets
    major_anatomy_hallucination = False
    for term in RISK_ANATOMY_TERMS:
        if contains_token(report_text, term) and normalize(term) not in supported_t:
            # treat as major if assertive cue exists and no uncertainty nearby
            if (not _text_has_uncertainty_near(report_text, term)) and _text_has_assertive_cue_near(report_text, term):
                major_anatomy_hallucination = True
                break

    # 2) tool-action mismatch: small compatibility table
    allowed_verbs_by_instrument = {
        "clipper": {"clip"},
        "hook": {"dissect", "coagulate"},
        "bipolar": {"coagulate"},
        "scissors": {"cut"},
        "irrigator": {"irrigate"},
        "aspirator": {"aspirate"},
        "grasper": {"grasp", "retract"},
    }
    major_tool_action_mismatch = False
    mismatch_items: List[str] = []
    for x in (canonical_ivt or []):
        i, v, t = _parse_triplet(x)
        if not i or not v:
            continue
        if i in allowed_verbs_by_instrument:
            allowed = allowed_verbs_by_instrument[i]
            if v not in allowed:
                major_tool_action_mismatch = True
                mismatch_items.append(f"{i},{v},{t}")

    # 3) self-contradiction: assertive claim of unsupported token
    supported_all = set(supported_i) | set(supported_v) | set(supported_t)
    vocab_all = list(vocab_i) + list(vocab_v) + list(vocab_t)
    major_self_contradiction = False
    contrad_tokens: List[str] = []
    # sentence-level scan
    sentences = re.split(r"[\n\.\!\?;]+", report_text)
    for sent in sentences:
        s_low = sent.lower()
        if not s_low.strip():
            continue
        has_assert = any(c in s_low for c in ASSERTIVE_CUES)
        has_uncert = any(u in s_low for u in UNCERTAINTY_TERMS)
        if (not has_assert) or has_uncert:
            continue
        for tok in vocab_all:
            nt = normalize(tok)
            if not tok:
                continue
            if nt in supported_all:
                continue
            if contains_token(sent, tok):
                major_self_contradiction = True
                contrad_tokens.append(nt)
                break
        if major_self_contradiction:
            break

    return {
        "major_anatomy_hallucination": major_anatomy_hallucination,
        "major_tool_action_mismatch": major_tool_action_mismatch,
        "tool_action_mismatch_items": mismatch_items,
        "major_self_contradiction": major_self_contradiction,
        "self_contradiction_tokens": sorted(set(contrad_tokens)),
        "any_major_error": bool(major_anatomy_hallucination or major_tool_action_mismatch or major_self_contradiction),
    }

