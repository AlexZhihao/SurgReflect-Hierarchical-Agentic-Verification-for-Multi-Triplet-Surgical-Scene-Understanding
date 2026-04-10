from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from collections import Counter, defaultdict
import re

from ..scoring import normalize, _parse_triplet


def _tok(s: str) -> List[str]:
    s = normalize(s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    parts = [p for p in s.split() if p]
    # remove ultra-common stopwords
    stop = {"the", "a", "an", "and", "or", "to", "of", "in", "on", "with", "for"}
    return [p for p in parts if p not in stop]


@dataclass
class Doc:
    key: str
    text: str
    tags: Tuple[str, ...] = ()


class StatsRAGStore:
    """
    Dataset-grounded RAG store that ONLY uses information from the benchmark itself:
      - phase <-> triplet co-occurrence
      - instrument/verb/target co-occurrence derived from gold triplets
    This avoids adding any speculative medical knowledge while still being very useful.
    """

    def __init__(self) -> None:
        self.docs: List[Doc] = []
        self._built = False

        # Structured stats (normalized keys) for deterministic tools.
        # These are dataset-grounded priors computed from the benchmark itself.
        self.phase_triplet: Dict[int, Counter[str]] = defaultdict(Counter)  # phase_id -> Counter(triplet_str)
        self.instr_verb: Dict[str, Counter[str]] = defaultdict(Counter)     # instrument -> Counter(verb)
        self.instr_target: Dict[str, Counter[str]] = defaultdict(Counter)   # instrument -> Counter(target)
        self.verb_target: Dict[str, Counter[str]] = defaultdict(Counter)    # verb -> Counter(target)
        self.phase_name_by_id: Dict[int, str] = {}

    def build_from_samples(self, samples: Sequence[Any]) -> None:
        if self._built:
            return

        # reset (idempotent build)
        self.phase_triplet = defaultdict(Counter)
        self.instr_verb = defaultdict(Counter)
        self.instr_target = defaultdict(Counter)
        self.verb_target = defaultdict(Counter)
        self.phase_name_by_id = {}

        for s in samples:
            tasks = getattr(s, "tasks", {}) or {}
            # phase choices mapping (from first occurrence)
            for c in (tasks.get("phase_mcq", {}) or {}).get("choices", []) or []:
                if isinstance(c, dict) and "id" in c and "name" in c:
                    try:
                        self.phase_name_by_id[int(c["id"])] = str(c["name"])
                    except Exception:
                        pass

            gt_phase = (tasks.get("phase_mcq", {}) or {}).get("answer", []) or []
            gt_phase_id = None
            if gt_phase and isinstance(gt_phase[0], dict) and "id" in gt_phase[0]:
                try:
                    gt_phase_id = int(gt_phase[0]["id"])
                except Exception:
                    gt_phase_id = None

            # Prefer report gold_context.ivt if exists, else ivt_mcq answer.
            gold_context = (tasks.get("report_task", {}) or {}).get("gold_context", {}) or {}
            gold_ivt = gold_context.get("ivt", None)
            if not gold_ivt:
                gold_ivt = (tasks.get("ivt_mcq", {}) or {}).get("answer", []) or []
            if not isinstance(gold_ivt, list):
                gold_ivt = [gold_ivt]

            for trip in gold_ivt:
                if not str(trip).strip():
                    continue
                tnorm = normalize(trip)
                i, v, t = _parse_triplet(tnorm)
                if not i or not v or not t:
                    continue
                if gt_phase_id is not None:
                    self.phase_triplet[gt_phase_id][tnorm] += 1
                self.instr_verb[i][v] += 1
                self.instr_target[i][t] += 1
                self.verb_target[v][t] += 1

        # Build docs: phase priors
        for pid, ctr in self.phase_triplet.items():
            pname = self.phase_name_by_id.get(pid, str(pid))
            top = [x for x, _ in ctr.most_common(10)]
            txt = (
                f"[DATASET STATS] Phase '{pname}' (id={pid}) commonly co-occurs with IVT triplets:\n"
                + "\n".join([f"- {x}" for x in top])
            )
            self.docs.append(Doc(key=f"phase:{pid}", text=txt, tags=("phase_prior", pname)))

        # Build docs: instrument behavior in dataset
        for inst, ctr in self.instr_verb.items():
            topv = [x for x, _ in ctr.most_common(8)]
            topt = [x for x, _ in self.instr_target.get(inst, Counter()).most_common(8)]
            txt = (
                f"[DATASET STATS] Instrument '{inst}' frequently pairs with verbs: {', '.join(topv)}.\n"
                f"Common targets: {', '.join(topt)}."
            )
            self.docs.append(Doc(key=f"inst:{inst}", text=txt, tags=("instrument_prior", inst)))

        # Build docs: verb->target
        for v, ctr in self.verb_target.items():
            topt = [x for x, _ in ctr.most_common(10)]
            txt = f"[DATASET STATS] Verb '{v}' commonly applies to targets: {', '.join(topt)}."
            self.docs.append(Doc(key=f"verb:{v}", text=txt, tags=("verb_prior", v)))

        self._built = True

    def retrieve(self, query: str, topk: int = 6, tag_filter: Optional[str] = None) -> List[Doc]:
        if not self.docs:
            return []
        qtok = _tok(query)
        if not qtok:
            return []

        scored: List[Tuple[float, Doc]] = []
        for d in self.docs:
            if tag_filter and (tag_filter not in d.tags):
                continue
            dtok = _tok(d.text)
            if not dtok:
                continue
            # simple overlap score
            overlap = sum(1 for t in qtok if t in dtok)
            if overlap <= 0:
                continue
            # bonus: tag hit
            tag_bonus = sum(1 for t in qtok if t in [normalize(x) for x in d.tags])
            score = float(overlap) + 0.5 * float(tag_bonus)
            scored.append((score, d))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored[: max(1, topk)]]
