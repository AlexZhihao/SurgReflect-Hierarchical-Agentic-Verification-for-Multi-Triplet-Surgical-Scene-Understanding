from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import math
from collections import Counter

from ..scoring import normalize, _parse_triplet
from ..scoring import RISK_ANATOMY_TERMS  # reuse list for report safety guidance
from .rag_store import StatsRAGStore, Doc


@dataclass
class ToolContext:
    name: str
    text: str


class AgentTool:
    name: str = "tool"

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def prepare(self, global_ctx: Dict[str, Any]) -> None:
        """Optional one-time preparation (e.g., build indices)."""
        return

    def get_context(self, sample_ctx: Dict[str, Any]) -> Optional[ToolContext]:
        return None


class RAGTool(AgentTool):
    name = "rag"

    def __init__(self, store: StatsRAGStore, enabled: bool = True, topk: int = 6) -> None:
        super().__init__(enabled=enabled)
        self.store = store
        self.topk = topk

    def retrieve_text(self, query: str, topk: Optional[int] = None, tag_filter: Optional[str] = None) -> str:
        if not self.enabled:
            return ""
        docs = self.store.retrieve(query=query, topk=topk or self.topk, tag_filter=tag_filter)
        if not docs:
            return ""
        return "\n\n".join([d.text for d in docs])


class CompatConstraintsTool(AgentTool):
    name = "compat_constraints"

    def __init__(self, enabled: bool = True) -> None:
        super().__init__(enabled=enabled)
        # mirror scoring.py major mismatch table (keep in sync)
        self.allowed_verbs_by_instrument = {
            "clipper": {"clip"},
            "hook": {"dissect", "coagulate"},
            "bipolar": {"coagulate"},
            "scissors": {"cut"},
            "irrigator": {"irrigate"},
            "aspirator": {"aspirate"},
            "grasper": {"grasp", "retract"},
        }

    def ivt_constraints_text(self) -> str:
        if not self.enabled:
            return ""
        lines = ["[CONSTRAINT] Tool–action compatibility (avoid mismatches):"]
        for inst, verbs in self.allowed_verbs_by_instrument.items():
            lines.append(f"- {inst}: {', '.join(sorted(verbs))}")
        return "\n".join(lines)

    def is_triplet_compatible(self, triplet: str) -> bool:
        i, v, _t = _parse_triplet(triplet)
        if not i or not v:
            return True
        if i not in self.allowed_verbs_by_instrument:
            return True
        return v in self.allowed_verbs_by_instrument[i]


class PhasePriorTool(AgentTool):
    name = "phase_prior"

    def __init__(self, rag: RAGTool, enabled: bool = True) -> None:
        super().__init__(enabled=enabled)
        self.rag = rag

    def phase_prior_text(self, predicted_ivt: Sequence[str]) -> str:
        if not self.enabled:
            return ""
        q = " ".join([normalize(x) for x in (predicted_ivt or [])])[:2000]
        txt = self.rag.retrieve_text(query=q, topk=4, tag_filter="phase_prior")
        if not txt:
            return ""
        return "[PHASE PRIOR HINTS]\n" + txt


def risk_terms_text() -> str:
    return "[SAFETY] Risk anatomy terms to avoid asserting unless supported: " + ", ".join(RISK_ANATOMY_TERMS)


class PhaseStatsTool(AgentTool):
    """Deterministic phase prior from dataset co-occurrence stats.

    This uses ONLY the benchmark's own gold labels aggregated across samples
    (via :class:`StatsRAGStore`). It does NOT use per-sample answers.

    Rationale: in CholecT50, phases correlate strongly with IVT patterns.
    """

    name = "phase_stats"

    def __init__(self, store: StatsRAGStore, enabled: bool = True) -> None:
        super().__init__(enabled=enabled)
        self.store = store

    def predict_phase(
        self,
        predicted_ivt: Sequence[str],
        phase_choices: Sequence[Dict[str, Any]],
        *,
        min_score: float = 0.5,
        min_margin: float = 0.15,
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        """Return (phase_choice_obj, debug).

        Scoring: sum(log(count+1)) over predicted triplets.
        """
        dbg: Dict[str, Any] = {"scores": {}, "best": None, "second": None, "min_score": min_score, "min_margin": min_margin}
        if not self.enabled:
            return None, dbg

        if not predicted_ivt:
            return None, dbg

        # Map available phase ids from choices
        allowed_ids = set()
        by_id: Dict[int, Dict[str, Any]] = {}
        for c in (phase_choices or []):
            if isinstance(c, dict) and "id" in c:
                try:
                    pid = int(c.get("id"))
                    allowed_ids.add(pid)
                    by_id[pid] = c
                except Exception:
                    continue

        if not allowed_ids:
            return None, dbg

        # Score each phase id in store
        scores: Dict[int, float] = {}
        for pid, ctr in (self.store.phase_triplet or {}).items():
            if pid not in allowed_ids:
                continue
            s = 0.0
            for trip in predicted_ivt:
                tnorm = normalize(trip)
                c = ctr.get(tnorm, 0)
                if c > 0:
                    s += math.log(float(c) + 1.0)
            scores[int(pid)] = s

        if not scores:
            return None, dbg

        # Rank
        ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        best_pid, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = best_score - second_score
        dbg["scores"] = {str(pid): float(sc) for pid, sc in ranked[:8]}
        dbg["best"] = {"id": best_pid, "score": float(best_score)}
        dbg["second"] = {"id": ranked[1][0], "score": float(second_score)} if len(ranked) > 1 else None
        dbg["margin"] = float(margin)

        # Gate
        if best_score < float(min_score):
            return None, dbg
        if margin < float(min_margin):
            # low confidence: still return, but mark as low
            dbg["low_confidence"] = True
        return by_id.get(best_pid), dbg


class IVTCandidateExpansionTool(AgentTool):
    """Build a short, high-recall IVT candidate list for LLM verification.

    The tool is deterministic and dataset-grounded via :class:`StatsRAGStore`.
    """

    name = "ivt_candidate_expand"

    def __init__(self, store: StatsRAGStore, enabled: bool = True) -> None:
        super().__init__(enabled=enabled)
        self.store = store

    def propose(
        self,
        *,
        ivt_choices: Sequence[str],
        inst_sel: Sequence[str],
        verb_sel: Sequence[str],
        targ_sel: Sequence[str],
        phase_hint: Optional[Dict[str, Any]] = None,
        include: Optional[Sequence[str]] = None,
        topk: int = 24,
    ) -> Tuple[List[str], Dict[str, Any]]:
        dbg: Dict[str, Any] = {"topk": int(topk)}
        if not self.enabled:
            return list(include or []), dbg

        inst_n = {normalize(x) for x in (inst_sel or []) if str(x).strip()}
        verb_n = {normalize(x) for x in (verb_sel or []) if str(x).strip()}
        targ_n = {normalize(x) for x in (targ_sel or []) if str(x).strip()}

        phase_id = None
        if isinstance(phase_hint, dict) and "id" in phase_hint:
            try:
                phase_id = int(phase_hint.get("id"))
            except Exception:
                phase_id = None

        scored: List[Tuple[float, str]] = []

        for ch in (ivt_choices or []):
            if not str(ch).strip():
                continue
            n = normalize(ch)
            i, v, t = _parse_triplet(n)
            if not (i and v and t):
                continue

            # Base score from dataset co-occurrence.
            s = 0.0
            s += float(self.store.instr_verb.get(i, Counter()).get(v, 0))
            s += float(self.store.verb_target.get(v, Counter()).get(t, 0))
            s += 0.5 * float(self.store.instr_target.get(i, Counter()).get(t, 0))

            # Phase-specific boost (if available)
            if phase_id is not None:
                s += 0.6 * float(self.store.phase_triplet.get(phase_id, Counter()).get(n, 0))

            # Alignment boosts with current I/V/T
            if i in inst_n:
                s += 3.0
            if v in verb_n:
                s += 2.0
            if t in targ_n:
                s += 1.5

            # Keep some relaxed candidates even if one dimension missing.
            overlap = int(i in inst_n) + int(v in verb_n) + int(t in targ_n)
            if overlap == 0:
                continue
            # Prefer high overlap
            s += 1.2 * overlap

            scored.append((s, str(ch)))

        scored.sort(key=lambda x: (-x[0], x[1]))

        # Assemble with include-first and de-dup.
        out: List[str] = []
        seen = set()
        for x in (include or []):
            nx = normalize(x)
            if nx and nx not in seen:
                out.append(str(x))
                seen.add(nx)
        for s, ch in scored:
            if len(out) >= int(topk):
                break
            nx = normalize(ch)
            if nx in seen:
                continue
            out.append(ch)
            seen.add(nx)

        dbg["n_scored"] = len(scored)
        dbg["selected"] = out
        dbg["top10"] = [{"triplet": t, "score": float(s)} for s, t in scored[:10]]
        return out, dbg
