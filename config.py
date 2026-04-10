from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class AgentToolsConfig:
    """Enable/disable pluggable tools for ablations."""
    # Core pipeline variants
    # - joint_predict: a single multitask call (baseline-style) to get strong recall on I/V/T/IVT.
    # - if disabled, fall back to the legacy per-task experts.
    joint_predict: bool = True

    # Dataset-grounded priors / reasoning tools
    rag: bool = True
    phase_prior: bool = True
    phase_stats: bool = True
    compat_constraints: bool = True

    # IVT recall/precision helpers
    ivt_candidate_expand: bool = True
    ivt_verify: bool = True

    # Repairs
    deterministic_repair: bool = True
    model_repair: bool = True

    # --- High-level ablation switches ---
    # Reflection: master switch for the entire reflect→repair pipeline.
    # When False, skip ALL repair stages (deterministic_repair + model_repair +
    # cross-task consistency + report sanitization + phase stats override).
    # Individual sub-switches above are still respected when reflection=True.
    reflection: bool = True

    # Chain-of-Thought: master switch for task-specific structured reasoning.
    # When True, prompts include explicit step-by-step reasoning chains
    # (visual-semantic for I/V/T, cognitive-reasoning for IVT/Phase/Report).
    # When False, prompts directly request JSON answers (baseline-style).
    chain_of_thought: bool = True

    # Self-Consistency: master switch for multi-call voting/union.
    # When False, force all k_* = 1 regardless of SelfConsistencyConfig values.
    self_consistency: bool = True


@dataclass
class SelfConsistencyConfig:
    """Optional self-consistency for each subtask (k calls, vote/union)."""
    k_joint: int = 1
    k_instrument: int = 1
    k_verb: int = 1
    k_target: int = 1
    k_ivt: int = 1
    k_ivt_verify: int = 1
    k_phase: int = 1
    k_report: int = 1


@dataclass
class AgentConfig:
    # Tooling / ablations
    tools: AgentToolsConfig = field(default_factory=AgentToolsConfig)
    self_consistency: SelfConsistencyConfig = field(default_factory=SelfConsistencyConfig)

    # Limits
    max_rounds: int = 1  # how many reflect->repair rounds
    max_output_tokens_per_call: int = 900
    max_output_tokens_report: int = 900

    # Decoding
    temperature: float = 0.0

    # Prompting behavior
    be_conservative: bool = True
    include_evidence: bool = True

    # Debugging
    save_debug: bool = False
    debug_dir: Optional[str] = None  # if set, write per-sample debug json

    # RAG parameters
    rag_topk: int = 6

    def effective_k(self, field: str) -> int:
        """Return effective k for a self-consistency field, respecting master switch."""
        raw = getattr(self.self_consistency, field, 1)
        if not self.tools.self_consistency:
            return 1
        return max(1, int(raw))

    def enabled_tools(self) -> List[str]:
        out: List[str] = []
        if self.tools.joint_predict:
            out.append("joint_predict")
        if self.tools.rag:
            out.append("rag")
        if self.tools.phase_prior:
            out.append("phase_prior")
        if self.tools.phase_stats:
            out.append("phase_stats")
        if self.tools.ivt_candidate_expand:
            out.append("ivt_candidate_expand")
        if self.tools.ivt_verify:
            out.append("ivt_verify")
        if self.tools.compat_constraints:
            out.append("compat_constraints")
        if self.tools.reflection:
            out.append("reflection")
            if self.tools.deterministic_repair:
                out.append("deterministic_repair")
            if self.tools.model_repair:
                out.append("model_repair")
        if self.tools.chain_of_thought:
            out.append("chain_of_thought")
        if self.tools.self_consistency:
            out.append("self_consistency")
        return out
