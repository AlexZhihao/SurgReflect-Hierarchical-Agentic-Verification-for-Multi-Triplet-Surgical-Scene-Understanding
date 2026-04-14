# SurgicalAgent — Agentic Framework for Surgical Video Understanding & Benchmark

A multi-expert, tool-augmented agentic pipeline for **laparoscopic cholecystectomy** video understanding, evaluated on the **CholecT50-Bench** benchmark. The framework orchestrates multimodal LLM calls through specialized experts, dataset-grounded reasoning tools, and multi-stage deterministic + model-based reflection to answer six surgical scene understanding tasks per sample.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Pipeline Walkthrough](#pipeline-walkthrough)
- [Compatible LLM APIs](#compatible-llm-apis)
- [Benchmark: CholecT50-Bench](#benchmark-cholect50-bench)
- [Configuration & Ablation](#configuration--ablation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                        Orchestrator                              │
│                    (SurgicalAgent.solve)                          │
│                                                                  │
│  ┌────────────┐   ┌──────────────┐   ┌────────────────────────┐  │
│  │  Stage 1   │──▶│   Stage 2    │──▶│      Stage 3-4         │  │
│  │   Base     │   │  Reflection  │   │  Report Generation     │  │
│  │ Prediction │   │   & Repair   │   │     & Sanitization     │  │
│  └────────────┘   └──────────────┘   └────────────────────────┘  │
│        │                 │                      │                 │
│        ▼                 ▼                      ▼                 │
│  ┌──────────┐   ┌───────────────┐   ┌────────────────────────┐   │
│  │ Experts  │   │    Tools      │   │  Model-Based Repair    │   │
│  │ (LLM)    │   │(Deterministic)│   │  (optional Stage 5)    │   │
│  └──────────┘   └───────────────┘   └────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

| Layer | Module | Role |
|---|---|---|
| **Orchestrator** | `orchestrator.py` | Master pipeline — wires experts, tools, and reflection stages |
| **Experts** | `experts.py` | LLM-calling modules, each specialized for one task |
| **Tools** | `tools.py` | Deterministic, dataset-grounded reasoning aids (RAG, constraints, stats) |
| **RAG Store** | `rag_store.py` | Builds co-occurrence statistics from benchmark gold labels; provides token-overlap retrieval |
| **Reflection** | `reflection.py` | Constraint enforcement, cross-task consistency, report sanitization, model-based repair |
| **Config** | `config.py` | Dataclass-based toggles for every component (ablation-friendly) |

---

## Pipeline Walkthrough

### Stage 0 — Global Pre-computation

Before processing any sample, the pipeline calls `StatsRAGStore.build_from_samples()` to compute **dataset-grounded co-occurrence statistics** from gold labels across the entire benchmark:

- Phase ↔ Triplet co-occurrence
- Instrument ↔ Verb co-occurrence
- Instrument ↔ Target co-occurrence
- Verb ↔ Target co-occurrence

These statistics power the downstream RAG retrieval, phase prediction priors, and IVT candidate expansion.

### Stage 1 — Base Predictions

Two prediction strategies are available (controlled by `joint_predict` toggle):

| Mode | Description |
|---|---|
| **Joint (default)** | A single `JointPredictExpert` call predicts **I, V, T, IVT, Phase** simultaneously for higher recall |
| **Sequential** | Per-task experts are called in dependency order: `Instrument → Verb → Target → IVT → Phase`, each receiving hints from upstream predictions |

Each expert supports:

- **Chain-of-Thought (CoT)** prompting for step-by-step reasoning
- **Self-Consistency** (k > 1 calls with union/vote aggregation)
- **Fuzzy choice mapping** via `ChoiceMapper` (difflib, cutoff = 0.86) to ground LLM outputs to exact answer choices

### Stage 2 — Deterministic Reflection & Repair

When `reflection=True`, the following repair passes execute sequentially:

1. **Phase Stats Prior (pre):** Scores phase choices against predicted IVT using `log(count + 1)` co-occurrence.
2. **IVT Candidate Expansion:** `IVTCandidateExpansionTool` proposes a ranked shortlist (up to 24 candidates) from co-occurrence stats + I/V/T alignment.
3. **IVT Verification:** `IVTVerifyExpert` asks the LLM to verify which candidates actually appear in the frames (high-recall mode: "maybe" counts as a soft positive).
4. **Constraint Repair:** `repair_ivt_with_constraints()` drops invalid triplets and fixes tool-action compatibility mismatches (e.g., *clipper* can only *clip*).
5. **Cross-Task Consistency:** `enforce_cross_task_consistency()` ensures I/V/T selections include all tokens found in selected IVT triplets (union enforcement).
6. **Phase Stats Override (post):** Recomputes phase score with repaired IVT; overrides model choice if statistical confidence is high.

### Stage 3 — Report Generation

`ReportExpert` generates a structured **3-section report**:

1. **IVT Events** — what instruments are doing to which targets
2. **Anatomy & Risk Assessment** — spatial/safety observations
3. **Next-Step Recommendation** — anticipated surgical actions

The report is grounded on the **canonical IVT + phase** produced by the reflection stages and includes risk-term safety guidance in the prompt.

### Stage 4 — Report Sanitization

`sanitize_report()` applies sentence-level safety filtering:

- Drops sentences with assertive mentions of risky anatomy terms not in canonical IVT
- Softens assertive language for unsupported vocabulary
- Ensures the 3-numbered-section format exists (with fallback template)

### Stage 5 — Model-Based Repair (optional)

When `model_repair=True`, the pipeline makes a one-shot LLM call presenting the current predictions, detected issues, and constraints/priors, asking the model to self-repair. All deterministic constraints are re-enforced afterward, and the report is regenerated to stay aligned.

### Stage 6 — Final Assembly

Outputs an evaluator-compatible JSON per sample:

```json
{
  "i_mcq":      ["grasper", "hook"],
  "v_mcq":      ["dissect", "retract"],
  "t_mcq":      ["gallbladder"],
  "ivt_mcq":    ["grasper,retract,gallbladder", "hook,dissect,gallbladder"],
  "phase_mcq":  ["calot-triangle-dissection"],
  "report_task": "1. The grasper is retracting the gallbladder..."
}
```

---

## Compatible LLM APIs

The framework supports **three multimodal LLM backends**, automatically dispatched by model name:

| Backend | Detection Logic | Supported Models (examples) |
|---|---|---|
| **OpenAI-compatible** (default) | Any model string not matching below | `gpt-4o`, `gpt-4.1`, `gpt-5`, `o3`, or any OpenAI-compatible endpoint |
| **Anthropic Claude** | `"claude"` in model name | `claude-sonnet-4-20250514`, `claude-4-opus-...` |
| **Google Gemini** | `"gemini"` in model name | `gemini-2.5-pro`, `gemini-2.5-flash` |

### OpenAI-compatible Endpoints

The OpenAI backend passes a `base_url` parameter, meaning it works with:

- **OpenAI API** directly
- **Azure OpenAI Service**
- **Local proxies** (LiteLLM, vLLM, Ollama with OpenAI-compatible mode, etc.)
- **Third-party providers** that expose an OpenAI-compatible `/v1/chat/completions` endpoint

### API Parameters

| Parameter | Default | Description |
|---|---|---|
| `model` | — | Model identifier string |
| `api_key` | — | API key for the provider |
| `base_url` | — | Custom endpoint URL (OpenAI-compatible only) |
| `anthropic_api_key` | — | Separate key for Anthropic backend |
| `temperature` | `0.0` | Sampling temperature (deterministic by default) |
| `max_output_tokens` | `900` | Max tokens per LLM call |

---

## Benchmark: CholecT50-Bench

### Overview

**CholecT50-Bench** (`cholect50_bench_tiers_1500x6.json`) is a multi-task benchmark comprising **1,500 samples × 6 tasks = 9,000 questions** for evaluating surgical scene understanding on laparoscopic cholecystectomy videos.

### Data Source

| Component | Source |
|---|---|
| **Video frames** | Extracted from the [Cholec80](http://camma.u-strasbg.fr/datasets) dataset surgical videos. Each `video_id` (e.g., `VID22`, `VID35`) corresponds to the original Cholec80 surgery recording. Frame images are extracted at the original resolution and organized into per-video folders (e.g., `videos/VID22/001301.png`). |
| **Surgical annotations** | Ground-truth labels for instruments, verbs, targets, IVT triplets, and surgical phases are derived from the [CholecT50](https://github.com/CAMMA-public/cholect50) annotation set, which provides fine-grained action triplet labels for 50 of the 80 Cholec80 videos. |
| **Question generation** | All multiple-choice questions, answer choices, and distractors were **generated by GPT-5.2** (multimodal) through structured prompting over the frame images and ground-truth annotations. The model was used to produce natural-language questions with clinically plausible distractors grounded in the visual content. |
| **Report gold standard** | Open-ended report references were constructed from the canonical IVT + phase annotations, providing a grounded factual basis for report evaluation. |

### Sample Structure

Each benchmark sample contains **3 consecutive frames** from the same surgical video and 6 tasks:

```json
{
  "video_id": "VID22",
  "frame_ids": [1301, 1302, 1303],
  "image_paths": ["videos/VID22/001301.png", "videos/VID22/001302.png", "videos/VID22/001303.png"],
  "meta": {
    "ivt_count": 3,
    "difficulty_bucket": "hard",
    "phase_rarity_bucket": "mid",
    "triplet_rarity_bucket": "mid",
    "tier": "hardplus",
    "diversity": 9
  },
  "tasks": { ... }
}
```

### Task Types

| Task | Key | Type | Description |
|---|---|---|---|
| **Instrument Recognition** | `i_mcq` | Multi-choice | Identify which surgical instruments are present |
| **Verb Recognition** | `v_mcq` | Multi-choice | Identify what actions/verbs are being performed |
| **Target Recognition** | `t_mcq` | Multi-choice | Identify anatomical targets being acted upon |
| **IVT Triplet Recognition** | `ivt_mcq` | Multi-choice | Identify complete ⟨instrument, verb, target⟩ triplets |
| **Phase Recognition** | `phase_mcq` | Single-choice | Identify the current surgical phase |
| **Report Generation** | `report_task` | Open-ended | Generate a structured surgical scene report |

### Difficulty Tiers

Samples are stratified across multiple difficulty dimensions:

| Dimension | Buckets | Based on |
|---|---|---|
| `difficulty_bucket` | easy / medium / hard | Overall composite difficulty |
| `phase_rarity_bucket` | common / mid / rare | Frequency of the surgical phase |
| `triplet_rarity_bucket` | common / mid / rare | Frequency of the rarest triplet in the sample |
| `tier` | composite label (e.g., `hardplus`) | Combined stratification |
| `diversity` | integer score | Number of distinct triplet components |

---

## Configuration & Ablation

All pipeline components can be individually toggled for systematic ablation studies:

```python
from agent import AgentConfig, AgentToolsConfig, SelfConsistencyConfig

tools_cfg = AgentToolsConfig(
    joint_predict=True,          # Joint vs. sequential prediction
    rag=True,                    # RAG retrieval from co-occurrence stats
    phase_prior=True,            # Phase-specific RAG hints
    phase_stats=True,            # Statistical phase scoring
    compat_constraints=True,     # Instrument-verb compatibility rules
    ivt_candidate_expand=True,   # IVT candidate expansion
    ivt_verify=True,             # LLM-based IVT verification
    reflection=True,             # Master switch for all repair stages
    deterministic_repair=True,   # Constraint-based deterministic repair
    model_repair=True,           # LLM-based self-repair
    chain_of_thought=True,       # CoT prompting
    self_consistency=True,       # Self-consistency (k > 1)
)

sc_cfg = SelfConsistencyConfig(
    k_joint=1,                   # Repetitions for joint prediction
    k_instrument=1,              # Repetitions for instrument expert
    k_verb=1,                    # ...
    k_ivt_verify=1,
    k_phase=1,
    k_report=1,
)

cfg = AgentConfig(
    tools=tools_cfg,
    self_consistency=sc_cfg,
    max_rounds=1,                # Reflect-repair iterations
    temperature=0.0,
    max_output_tokens_per_call=900,
    save_debug=True,             # Save per-sample debug JSON
)
```

---

## Quick Start

```python
import json
from agent import SurgicalAgent, AgentConfig

# Load benchmark
with open("cholect50_bench_tiers_1500x6.json") as f:
    samples = json.load(f)

# Initialize agent
cfg = AgentConfig()
agent = SurgicalAgent(cfg)

# Build global RAG store (once)
agent.build_global(samples)

# Solve a single sample
result = agent.solve(
    sample=samples[0],
    model="gpt-4o",
    api_key="sk-...",
    base_url="https://api.openai.com/v1",   # optional, for OpenAI-compatible endpoints
)

print(json.dumps(result, indent=2))
```

### Using Different LLM Backends

```python
# Anthropic Claude
result = agent.solve(
    sample=samples[0],
    model="claude-sonnet-4-20250514",
    anthropic_api_key="sk-ant-...",
)

# Google Gemini
result = agent.solve(
    sample=samples[0],
    model="gemini-2.5-pro",
    api_key="AIza...",
)

# Local / third-party OpenAI-compatible endpoint
result = agent.solve(
    sample=samples[0],
    model="my-local-model",
    api_key="dummy",
    base_url="http://localhost:8000/v1",
)
```

---

## Project Structure

```
agent/
├── __init__.py                         # Public API exports
├── config.py                           # AgentConfig, AgentToolsConfig, SelfConsistencyConfig
├── orchestrator.py                     # SurgicalAgent — master pipeline
├── experts.py                          # LLM expert modules (I/V/T/IVT/Phase/Report/Joint/Verify)
├── tools.py                            # Deterministic tools (RAG, constraints, stats, expansion)
├── rag_store.py                        # StatsRAGStore — co-occurrence statistics & retrieval
├── reflection.py                       # Deterministic repair + model-based repair + sanitization
└── cholect50_bench_tiers_1500x6.json   # Benchmark (1500 samples × 6 tasks)
```


## License

Please refer to the [Cholec80](http://camma.u-strasbg.fr/datasets) and [CholecT50](https://github.com/CAMMA-public/cholect50) dataset licenses for data usage terms.


## References

- [SurAgent (Related Project)](https://github.com/GeorgeHuLite/SurAgent)
