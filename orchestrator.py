from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import json

from ..scoring import extract_vocab_from_choices
from ..io_utils import write_json
from .config import AgentConfig
from .rag_store import StatsRAGStore
from .tools import (
    RAGTool,
    CompatConstraintsTool,
    PhasePriorTool,
    PhaseStatsTool,
    IVTCandidateExpansionTool,
)
from .experts import (
    InstrumentExpert,
    VerbExpert,
    TargetExpert,
    IVTExpert,
    PhaseExpert,
    ReportExpert,
    ChoiceMapper,
    JointPredictExpert,
    IVTVerifyExpert,
)
from .reflection import (
    enforce_cross_task_consistency,
    repair_ivt_with_constraints,
    sanitize_report,
    model_repair,
)


class SurgicalAgent:
    """
    A pluggable multi-step agent for 3-frame surgical video understanding.

    Design goals:
    - outputs EXACTLY the same JSON schema as baseline (for drop-in evaluation)
    - tools are fully togglable for ablations (RAG, priors, constraints, repair)
    - deterministic repairs maximize "exact-string" compatibility with choice spaces
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        rag_store: Optional[StatsRAGStore] = None,
    ) -> None:
        self.config = config or AgentConfig()
        self.rag_store = rag_store or StatsRAGStore()

        # Tools
        self.rag_tool = RAGTool(self.rag_store, enabled=self.config.tools.rag, topk=self.config.rag_topk)
        self.compat_tool = CompatConstraintsTool(enabled=self.config.tools.compat_constraints)
        self.phase_prior_tool = PhasePriorTool(self.rag_tool, enabled=self.config.tools.phase_prior)
        self.phase_stats_tool = PhaseStatsTool(self.rag_store, enabled=self.config.tools.phase_stats)
        self.ivt_expand_tool = IVTCandidateExpansionTool(self.rag_store, enabled=self.config.tools.ivt_candidate_expand)

        # Experts
        self.joint_expert = JointPredictExpert(
            rag=self.rag_tool if self.config.tools.rag else None,
            compat=self.compat_tool if self.config.tools.compat_constraints else None,
            phase_prior=self.phase_prior_tool if self.config.tools.phase_prior else None,
        )
        self.ivt_verify_expert = IVTVerifyExpert(compat=self.compat_tool if self.config.tools.compat_constraints else None)

        self.instrument_expert = InstrumentExpert(rag=self.rag_tool if self.config.tools.rag else None)
        self.verb_expert = VerbExpert(rag=self.rag_tool if self.config.tools.rag else None)
        self.target_expert = TargetExpert(rag=self.rag_tool if self.config.tools.rag else None)
        self.ivt_expert = IVTExpert(
            rag=self.rag_tool if self.config.tools.rag else None,
            compat=self.compat_tool if self.config.tools.compat_constraints else None,
        )
        self.phase_expert = PhaseExpert(
            rag=self.rag_tool if self.config.tools.rag else None,
            phase_prior=self.phase_prior_tool if self.config.tools.phase_prior else None,
        )
        self.report_expert = ReportExpert(rag=self.rag_tool if self.config.tools.rag else None)

    def build_global(self, samples: Sequence[Any]) -> None:
        """Build dataset-grounded RAG indices once per run."""
        if (
            self.config.tools.rag
            or self.config.tools.phase_prior
            or self.config.tools.phase_stats
            or self.config.tools.ivt_candidate_expand
        ):
            self.rag_store.build_from_samples(samples)

    def _maybe_save_debug(self, sample_id: str, dbg: Dict[str, Any]) -> None:
        if not self.config.save_debug:
            return
        if not self.config.debug_dir:
            return
        d = Path(self.config.debug_dir)
        d.mkdir(parents=True, exist_ok=True)
        write_json(d / f"{sample_id}.json", dbg)

    def solve(
        self,
        *,
        sample_id: str,
        tasks: Dict[str, Any],
        image_data_urls: List[str],
        base_model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Returns a dict matching the evaluator JSON schema.
        """
        cfg = self.config
        temp = float(cfg.temperature if temperature is None else temperature)
        max_tok = int(cfg.max_output_tokens_per_call if max_output_tokens is None else max_output_tokens)

        debug: Dict[str, Any] = {
            "agent_config": {
                "tools": asdict(cfg.tools),
                "self_consistency": asdict(cfg.self_consistency),
                "max_rounds": cfg.max_rounds,
                "temperature": temp,
                "max_output_tokens_per_call": max_tok,
                "enabled_tools": cfg.enabled_tools(),
            },
            "raw_rounds": [],
        }

        # =========================
        # Round 0: get base predictions
        # =========================
        # Default path (stronger recall): a single multitask call.
        # Fallback path: the legacy per-task experts.
        i_pred = v_pred = t_pred = ivt_pred = phase_pred = report_pred = None  # type: ignore
        use_cot = cfg.tools.chain_of_thought
        if cfg.tools.joint_predict:
            joint = self.joint_expert.run(
                model=base_model,
                tasks=tasks,
                image_data_urls=image_data_urls,
                api_key=api_key,
                base_url=base_url,
                temperature=temp,
                max_output_tokens=max_tok,
                k=cfg.effective_k("k_joint"),
                use_cot=use_cot,
            )
            debug["raw_rounds"].append({"joint_raw": joint.raw})
            i0, v0, t0, ivt0, phase0 = (
                joint.i_selected,
                joint.v_selected,
                joint.t_selected,
                joint.ivt_selected,
                joint.phase_selected,
            )
        else:
            i_pred = self.instrument_expert.run(
                model=base_model,
                tasks=tasks,
                image_data_urls=image_data_urls,
                api_key=api_key,
                base_url=base_url,
                temperature=temp,
                max_output_tokens=max_tok,
                k=cfg.effective_k("k_instrument"),
                use_cot=use_cot,
            )
            v_pred = self.verb_expert.run(
                model=base_model,
                tasks=tasks,
                image_data_urls=image_data_urls,
                api_key=api_key,
                base_url=base_url,
                temperature=temp,
                max_output_tokens=max_tok,
                k=cfg.effective_k("k_verb"),
                instrument_selected=i_pred.selected,
                use_cot=use_cot,
            )
            t_pred = self.target_expert.run(
                model=base_model,
                tasks=tasks,
                image_data_urls=image_data_urls,
                api_key=api_key,
                base_url=base_url,
                temperature=temp,
                max_output_tokens=max_tok,
                k=cfg.effective_k("k_target"),
                instrument_selected=i_pred.selected,
                verb_selected=v_pred.selected,
                use_cot=use_cot,
            )
            ivt_pred = self.ivt_expert.run(
                model=base_model,
                tasks=tasks,
                image_data_urls=image_data_urls,
                api_key=api_key,
                base_url=base_url,
                temperature=temp,
                max_output_tokens=max_tok,
                k=cfg.effective_k("k_ivt"),
                instrument_selected=i_pred.selected,
                verb_selected=v_pred.selected,
                target_selected=t_pred.selected,
                use_cot=use_cot,
            )
            phase_pred = self.phase_expert.run(
                model=base_model,
                tasks=tasks,
                image_data_urls=image_data_urls,
                api_key=api_key,
                base_url=base_url,
                temperature=temp,
                max_output_tokens=max_tok,
                k=cfg.effective_k("k_phase"),
                predicted_ivt=ivt_pred.selected,
                instrument_selected=i_pred.selected,
                verb_selected=v_pred.selected,
                target_selected=t_pred.selected,
                use_cot=use_cot,
            )

            debug["raw_rounds"].append(
                {
                    "i_raw": i_pred.raw,
                    "v_raw": v_pred.raw,
                    "t_raw": t_pred.raw,
                    "ivt_raw": ivt_pred.raw,
                    "phase_raw": phase_pred.raw,
                }
            )
            i0, v0, t0, ivt0, phase0 = (
                i_pred.selected,
                v_pred.selected,
                t_pred.selected,
                ivt_pred.selected,
                phase_pred.selected,
            )

        # =========================
        # Deterministic repair pass
        # =========================
        i_choices = (tasks.get("i_mcq", {}) or {}).get("choices", []) or []
        v_choices = (tasks.get("v_mcq", {}) or {}).get("choices", []) or []
        t_choices = (tasks.get("t_mcq", {}) or {}).get("choices", []) or []
        ivt_choices = (tasks.get("ivt_mcq", {}) or {}).get("choices", []) or []
        p_choices = (tasks.get("phase_mcq", {}) or {}).get("choices", []) or []

        # Ensure exact mapping again (defensive)
        i_mapper = ChoiceMapper(i_choices)
        v_mapper = ChoiceMapper(v_choices)
        t_mapper = ChoiceMapper(t_choices)
        ivt_mapper = ChoiceMapper(ivt_choices)
        p_mapper = ChoiceMapper(p_choices)

        # Map base predictions to exact choice strings
        i_sel = i_mapper.map_string_list(i0)
        v_sel = v_mapper.map_string_list(v0)
        t_sel = t_mapper.map_string_list(t0)
        ivt_sel = ivt_mapper.map_string_list(ivt0)
        phase_sel = p_mapper.map_phase_obj(phase0)

        issues: List[str] = []

        # =========================
        # Reflection gate: when reflection is OFF, skip all repair stages
        # =========================
        use_reflection = cfg.tools.reflection

        # -------------------------
        # Phase stats prior (pre)
        # -------------------------
        phase_stats_choice = None
        phase_stats_dbg: Dict[str, Any] = {}
        if use_reflection and cfg.tools.phase_stats and p_choices:
            phase_stats_choice, phase_stats_dbg = self.phase_stats_tool.predict_phase(
                predicted_ivt=ivt_sel,
                phase_choices=p_choices,
            )
            debug["phase_stats_pre"] = phase_stats_dbg

        # -------------------------
        # IVT candidate expansion + verification (recall-oriented)
        # -------------------------
        ivt_cand_dbg: Dict[str, Any] = {}
        if use_reflection and cfg.tools.ivt_verify and cfg.tools.ivt_candidate_expand and ivt_choices:
            phase_hint = phase_stats_choice or phase_sel
            candidates, ivt_cand_dbg = self.ivt_expand_tool.propose(
                ivt_choices=ivt_choices,
                inst_sel=i_sel,
                verb_sel=v_sel,
                targ_sel=t_sel,
                phase_hint=phase_hint,
                include=ivt_sel,
                topk=24,
            )
            if candidates:
                ivt_ver = self.ivt_verify_expert.run(
                    model=base_model,
                    candidates=candidates,
                    image_data_urls=image_data_urls,
                    api_key=api_key,
                    base_url=base_url,
                    temperature=temp,
                    max_output_tokens=max_tok,
                    k=cfg.effective_k("k_ivt_verify"),
                )
                debug["raw_rounds"].append({"ivt_verify_raw": ivt_ver.raw, "ivt_candidates": candidates})
                ivt_sel = ivt_mapper.map_string_list(ivt_ver.selected)
                issues.append("IVT verified on a candidate shortlist.")

        debug["ivt_candidate_expand"] = ivt_cand_dbg

        # -------------------------
        # Deterministic enforcement
        # -------------------------
        ivt_dbg: Dict[str, Any] = {}
        if use_reflection and cfg.tools.deterministic_repair:
            # IVT constraint repair
            ivt_sel2, ivt_dbg = repair_ivt_with_constraints(
                ivt_sel,
                ivt_choices=ivt_choices,
                compat=self.compat_tool if cfg.tools.compat_constraints else None,
            )
            if ivt_sel2 != ivt_sel:
                issues.append("IVT repaired for compatibility / in-choice exactness.")
            ivt_sel = ivt_sel2

            # Cross-task consistency (union in ivt tokens)
            i_sel2, v_sel2, t_sel2 = enforce_cross_task_consistency(
                i_sel=i_sel,
                v_sel=v_sel,
                t_sel=t_sel,
                ivt_sel=ivt_sel,
                i_choices=i_choices,
                v_choices=v_choices,
                t_choices=t_choices,
            )
            if (i_sel2 != i_sel) or (v_sel2 != v_sel) or (t_sel2 != t_sel):
                issues.append("Added missing I/V/T tokens implied by IVT.")
            i_sel, v_sel, t_sel = i_sel2, v_sel2, t_sel2

        # -------------------------
        # Phase stats override (post)
        # -------------------------
        if use_reflection and cfg.tools.phase_stats and p_choices:
            phase_stats_choice2, phase_stats_dbg2 = self.phase_stats_tool.predict_phase(
                predicted_ivt=ivt_sel,
                phase_choices=p_choices,
            )
            debug["phase_stats_post"] = phase_stats_dbg2
            if phase_stats_choice2 is not None:
                # Override if the model phase is missing, or not among top-2 stats phases,
                # or the stats signal is confident.
                if phase_sel is None:
                    phase_sel = phase_stats_choice2
                    issues.append("Phase set by stats prior derived from IVT.")
                else:
                    top_ids: List[int] = []
                    try:
                        items = [(int(k), float(v)) for k, v in (phase_stats_dbg2.get("scores", {}) or {}).items()]
                        items.sort(key=lambda x: (-x[1], x[0]))
                        top_ids = [pid for pid, _ in items[:2]]
                    except Exception:
                        top_ids = []
                    confident = not bool(phase_stats_dbg2.get("low_confidence", False))
                    try:
                        cur_id = int(phase_sel.get("id"))
                    except Exception:
                        cur_id = -999
                    if confident or (top_ids and (cur_id not in top_ids)):
                        if int(phase_stats_choice2.get("id")) != cur_id:
                            phase_sel = phase_stats_choice2
                            issues.append("Phase overridden by stats prior derived from IVT.")

        if phase_sel is None:
            issues.append("Phase selection invalid; needs repair/mapping.")

        # -------------------------
        # Generate report from canonical IVT + phase (safer than free-form baseline report)
        # -------------------------
        report_pred = self.report_expert.run(
            model=base_model,
            tasks=tasks,
            image_data_urls=image_data_urls,
            api_key=api_key,
            base_url=base_url,
            temperature=temp,
            max_output_tokens=int(cfg.max_output_tokens_report),
            k=cfg.effective_k("k_report"),
            canonical_ivt=ivt_sel,
            canonical_phase=phase_sel,
            use_cot=use_cot,
        )
        debug["raw_rounds"].append({"report_raw": report_pred.raw})

        vocab_i = extract_vocab_from_choices(i_choices)
        vocab_v = extract_vocab_from_choices(v_choices)
        vocab_t = extract_vocab_from_choices(t_choices)
        report_text = (
            sanitize_report(
                report_pred.report_text,
                canonical_ivt=ivt_sel,
                vocab_i=vocab_i,
                vocab_v=vocab_v,
                vocab_t=vocab_t,
            )
            if use_reflection and cfg.tools.deterministic_repair
            else (report_pred.report_text or "").strip()
        )
        if use_reflection and cfg.tools.deterministic_repair and report_text != (report_pred.report_text or "").strip():
            issues.append("Report sanitized to reduce major errors / enforce structure.")

        debug["deterministic_repair"] = {
            "issues": issues,
            "ivt_debug": ivt_dbg,
            "post": {
                "i_selected": i_sel,
                "v_selected": v_sel,
                "t_selected": t_sel,
                "ivt_selected": ivt_sel,
                "phase_selected": phase_sel,
                "report_text": report_text,
            },
        }

        # =========================
        # Optional model-based repair loop
        # =========================
        current = {
            "i_mcq": {"selected": i_sel},
            "v_mcq": {"selected": v_sel},
            "t_mcq": {"selected": t_sel},
            "ivt_mcq": {"selected": ivt_sel},
            "phase_mcq": {"selected": phase_sel if phase_sel is not None else {"id": -1, "name": ""}},
            "report_task": {"report_text": report_text},
        }

        if use_reflection and cfg.tools.model_repair and cfg.max_rounds > 0 and issues:
            # Provide extra hints: compat constraints + phase priors from RAG
            extra = ""
            if cfg.tools.compat_constraints:
                extra += self.compat_tool.ivt_constraints_text() + "\n\n"
            if cfg.tools.phase_prior:
                extra += self.phase_prior_tool.phase_prior_text(ivt_sel) + "\n\n"

            repaired = model_repair(
                model=base_model,
                image_data_urls=image_data_urls,
                api_key=api_key,
                base_url=base_url,
                temperature=temp,
                max_output_tokens=max_tok,
                tasks=tasks,
                current=current,
                issues=issues[:8],
                extra_hints=extra,
            )

            debug["model_repair_raw"] = repaired

            # Map repaired outputs back to exact choice strings
            i_sel_r = i_mapper.map_string_list((repaired.get("i_mcq", {}) or {}).get("selected", []))
            v_sel_r = v_mapper.map_string_list((repaired.get("v_mcq", {}) or {}).get("selected", []))
            t_sel_r = t_mapper.map_string_list((repaired.get("t_mcq", {}) or {}).get("selected", []))
            ivt_sel_r = ivt_mapper.map_string_list((repaired.get("ivt_mcq", {}) or {}).get("selected", []))
            phase_sel_r = p_mapper.map_phase_obj(((repaired.get("phase_mcq", {}) or {}).get("selected", None)))
            report_text_r = str(((repaired.get("report_task", {}) or {}).get("report_text", "")) or "").strip()

            # Re-apply deterministic enforcement after repair
            ivt_sel_r, _ = repair_ivt_with_constraints(ivt_sel_r, ivt_choices=ivt_choices, compat=self.compat_tool if cfg.tools.compat_constraints else None)
            i_sel_r, v_sel_r, t_sel_r = enforce_cross_task_consistency(
                i_sel=i_sel_r, v_sel=v_sel_r, t_sel=t_sel_r, ivt_sel=ivt_sel_r,
                i_choices=i_choices, v_choices=v_choices, t_choices=t_choices
            )
            report_text_r = sanitize_report(report_text_r, canonical_ivt=ivt_sel_r, vocab_i=vocab_i, vocab_v=vocab_v, vocab_t=vocab_t)

            # accept repaired only if it doesn't get empty unexpectedly
            if ivt_sel_r or i_sel_r or v_sel_r or t_sel_r:
                i_sel, v_sel, t_sel, ivt_sel = i_sel_r, v_sel_r, t_sel_r, ivt_sel_r
            if phase_sel_r is not None:
                phase_sel = phase_sel_r

            # Always regenerate the report from the (possibly repaired) canonical IVT/phase.
            # This keeps the report aligned and reduces major-error triggers.
            report_pred2 = self.report_expert.run(
                model=base_model,
                tasks=tasks,
                image_data_urls=image_data_urls,
                api_key=api_key,
                base_url=base_url,
                temperature=temp,
                max_output_tokens=int(cfg.max_output_tokens_report),
                k=1,
                canonical_ivt=ivt_sel,
                canonical_phase=phase_sel,
                use_cot=use_cot,
            )
            debug["raw_rounds"].append({"report_raw_after_repair": report_pred2.raw})
            report_text = (
                sanitize_report(
                    report_pred2.report_text,
                    canonical_ivt=ivt_sel,
                    vocab_i=vocab_i,
                    vocab_v=vocab_v,
                    vocab_t=vocab_t,
                )
                if cfg.tools.deterministic_repair
                else (report_pred2.report_text or "").strip()
            )

        # Skip reflection stages but still log
        if not use_reflection:
            debug["reflection_skipped"] = True

        # =========================
        # Final assemble (evaluator schema)
        # =========================
        final = {
            "i_mcq": {"selected": i_sel},
            "v_mcq": {"selected": v_sel},
            "t_mcq": {"selected": t_sel},
            "ivt_mcq": {"selected": ivt_sel},
            "phase_mcq": {"selected": phase_sel if phase_sel is not None else (p_choices[0] if p_choices else {"id": 0, "name": "preparation"})},
            "report_task": {
                "report_text": report_text,
                "appendix": {
                    "canonical_ivt": ivt_sel,
                    "canonical_phase": phase_sel if phase_sel is not None else (p_choices[0] if p_choices else {"id": 0, "name": "preparation"}),
                },
            },
        }

        debug["final"] = final
        self._maybe_save_debug(sample_id=sample_id, dbg=debug)
        return final
