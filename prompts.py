from __future__ import annotations

from typing import Any, Dict, List


def _format_choices(choices: List[Any]) -> str:
    if not choices:
        return "[]"
    # phase choices are dicts: {"id":..,"name":..}
    if isinstance(choices[0], dict):
        return "\n".join([f"- {c['id']}: {c['name']}" for c in choices])
    return "\n".join([f"- {c}" for c in choices])


# =========================================================================
# Task-specific Chain-of-Thought templates (inspired by SurgRAW)
#
# Two reasoning families:
#   1. Visual-Semantic CoT  — for I (instrument), V (verb), T (target)
#      Chain: Scene Overview → Visual Feature Extraction → Cross-Verification
#            → Option Elimination → Final Selection & Self-Check
#
#   2. Cognitive-Reasoning CoT — for IVT (triplet), Phase, Report
#      Chain: Problem Decomposition → Visual Feature Extraction
#            → Sub-problem Solving → Knowledge Cross-Check
#            → Option Elimination → Final Selection & Self-Check
#
# Each template is injected BEFORE the JSON output instruction so the model
# reasons first, then emits structured JSON.  When CoT is disabled the
# prompt falls back to the current direct-answer style.
# =========================================================================

# --------------- Visual-Semantic CoT (I / V / T) ---------------

_COT_INSTRUMENT = """\
Before answering, follow this structured reasoning chain step by step.
Write your reasoning inside the "reasoning" field of the JSON output.

STEP 1 — Scene Overview:
  Briefly describe the overall surgical scene in the 3rd (target) frame.
  What anatomical region is visible? What is the overall surgical context?

STEP 2 — Visual Feature Extraction:
  For EACH candidate instrument in the choices, describe visual cues you
  look for (shape, color, tip geometry, shaft visibility, position in frame).
  Note which cues are PRESENT vs ABSENT in the 3rd frame.

STEP 3 — Cross-Verification with Temporal Context:
  Compare frames 1→2→3. Are the instruments consistent across frames?
  Does any instrument appear/disappear? Use motion continuity to confirm.

STEP 4 — Option Elimination:
  Explicitly list choices you can RULE OUT and why (e.g., "scissors: no
  visible scissor-shaped tip or cutting motion").

STEP 5 — Final Selection & Self-Check:
  State your final selections. For each, cite at least one visual cue.
  If confidence is low for any, mark it uncertain rather than guessing.

"""

_COT_VERB = """\
Before answering, follow this structured reasoning chain step by step.
Write your reasoning inside the "reasoning" field of the JSON output.

STEP 1 — Scene Overview:
  Describe the surgical scene and what activity appears to be happening.

STEP 2 — Motion & Interaction Analysis:
  Compare instrument positions across frames 1→2→3.
  What motion patterns do you observe? (opening/closing, pulling, pushing,
  rotating, static contact, fluid flow, etc.)

STEP 3 — Instrument-Action Association:
  For each instrument you identified (or given as hints), what action is it
  performing? Use the instrument type to constrain plausible verbs
  (e.g., hook → dissect/coagulate, clipper → clip).

STEP 4 — Option Elimination:
  Which verb choices can you rule out? Cite visual evidence for exclusions.

STEP 5 — Final Selection & Self-Check:
  State final verb selections with supporting visual/motion evidence.
  Flag any uncertain selections.

"""

_COT_TARGET = """\
Before answering, follow this structured reasoning chain step by step.
Write your reasoning inside the "reasoning" field of the JSON output.

STEP 1 — Scene Overview:
  Describe the visible anatomy and surgical field in the 3rd frame.

STEP 2 — Anatomical Feature Extraction:
  What tissue structures are visible? Describe color, texture, location,
  and spatial relationship to instruments. Be conservative — do NOT name
  specific ducts/arteries unless unambiguously visible.

STEP 3 — Instrument-Target Interaction:
  Where is each instrument's tip contacting or pointing? What tissue is
  at the point of interaction?

STEP 4 — Option Elimination:
  Which target choices can you rule out based on visual evidence?
  (e.g., "liver bed: instruments are not near the liver surface")

STEP 5 — Final Selection & Self-Check:
  State final target selections. Cite anatomical visual cues.
  Prefer conservative choices if ambiguous.

"""

# --------------- Cognitive-Reasoning CoT (IVT / Phase / Report) ---------------

_COT_IVT = """\
Before answering, follow this structured reasoning chain step by step.
Write your reasoning inside the "reasoning" field of the JSON output.

STEP 1 — Problem Decomposition:
  An IVT triplet = (instrument, verb, target). You need to find ALL valid
  combinations that occur in the 3rd frame.

STEP 2 — Visual Feature Extraction:
  List each instrument visible, its action, and the tissue it contacts.

STEP 3 — Sub-problem: Build Candidate Triplets:
  For each visible instrument, pair it with its most likely verb and target
  to form candidate IVT triplets. Then check each against the choice list.

STEP 4 — Knowledge Cross-Check (Compatibility):
  Verify tool-action compatibility:
  - clipper → clip only; hook → dissect/coagulate; scissors → cut;
  - bipolar → coagulate; irrigator → irrigate; grasper → grasp/retract.
  Reject any triplet that violates these constraints.

STEP 5 — Option Elimination:
  From the provided IVT choices, eliminate those where:
  (a) the instrument is not visible, OR
  (b) the action contradicts the observed motion, OR
  (c) the target is not at the point of interaction.

STEP 6 — Final Selection & Self-Check:
  List final IVT selections. Each must have visual support for ALL three
  components. If uncertain, prefer including a plausible triplet over
  missing a true event (favor recall).

"""

_COT_PHASE = """\
Before answering, follow this structured reasoning chain step by step.
Write your reasoning inside the "reasoning" field of the JSON output.

STEP 1 — Problem Decomposition:
  Surgical phase = the high-level stage of the operation. It depends on
  which instruments are active, what actions are being performed, and on
  which anatomical structures.

STEP 2 — Visual Feature Extraction:
  Summarize the scene: instruments present, actions occurring, anatomy
  visible, tissue state (intact/dissected/clipped).

STEP 3 — Sub-problem: Phase–IVT Association:
  Match the observed IVT pattern to typical phase characteristics:
  - Preparation: minimal instruments, abdominal wall, trocar placement.
  - Calot triangle dissection: hook dissecting near gallbladder pedicle.
  - Clipping & cutting: clipper applying clips, scissors cutting.
  - Gallbladder dissection: hook/spatula dissecting gallbladder from liver bed.
  - Gallbladder packaging: specimen retrieval.
  - Cleaning & coagulation: irrigator washing, bipolar coagulating.
  - Gallbladder retraction: grasper retracting gallbladder.

STEP 4 — Option Elimination:
  Which phase choices are clearly inconsistent with the visual evidence?

STEP 5 — Final Selection & Self-Check:
  Select exactly ONE phase. Cite the primary visual/IVT cues that support it.

"""

_COT_REPORT = """\
Before writing the report, follow this structured reasoning chain internally.

STEP 1 — Summarize Observations:
  What instruments, actions, and targets are confirmed (from canonical IVT)?

STEP 2 — Anatomy & Risk Assessment:
  What anatomy is visible? Are there any risk-relevant structures?
  Be conservative: do NOT assert cystic duct/artery/CBD unless clearly visible.

STEP 3 — Phase-Consistent Next Step:
  Given the current phase and IVT events, what is the logical next
  surgical step? It must be consistent and safe.

STEP 4 — Draft Report:
  Write the 3 numbered sections based on Steps 1-3.

"""

# Mapping from task key to CoT template
_COT_TEMPLATES: Dict[str, str] = {
    "instrument": _COT_INSTRUMENT,
    "verb": _COT_VERB,
    "target": _COT_TARGET,
    "ivt": _COT_IVT,
    "phase": _COT_PHASE,
    "report": _COT_REPORT,
}


def get_cot_prompt(task_key: str) -> str:
    """Return the CoT template for a given task, or empty string if unknown."""
    return _COT_TEMPLATES.get(task_key, "")


def build_multitask_cot_block() -> str:
    """Build a combined CoT instruction block for the joint multitask prompt.

    This merges the key reasoning steps from all sub-tasks into one coherent
    chain that the model follows before emitting the JSON answer.
    """
    return """\
Before producing the JSON answer, you MUST follow this structured reasoning
chain. Write your full reasoning inside the "reasoning" field at the top
level of the JSON output.

=== STRUCTURED REASONING CHAIN ===

STEP 1 — Scene Overview:
  Describe the overall surgical scene in the 3rd (target) frame.
  What anatomical region is visible? What is the general surgical activity?

STEP 2 — Instrument Identification (Visual-Semantic):
  For each candidate instrument, describe visual cues (shape, color, tip
  geometry). Note which are PRESENT vs ABSENT. Use frames 1-2 for temporal
  confirmation.

STEP 3 — Action/Verb Analysis (Visual-Semantic):
  Compare instrument positions across frames 1→2→3. What motion patterns
  (opening/closing, pulling, static contact, fluid flow) do you see?
  Associate each instrument with its action.

STEP 4 — Target Identification (Visual-Semantic):
  What anatomy is at each instrument's point of interaction?
  Be conservative — do NOT name specific ducts/arteries unless unambiguous.

STEP 5 — IVT Triplet Assembly (Cognitive-Reasoning):
  Combine STEP 2-4 into (instrument, verb, target) triplets.
  Cross-check each against tool-action compatibility constraints:
  clipper→clip, hook→dissect/coagulate, scissors→cut,
  bipolar→coagulate, irrigator→irrigate, grasper→grasp/retract.
  Eliminate invalid combinations.

STEP 6 — Phase Recognition (Cognitive-Reasoning):
  Match the IVT pattern and visual scene to the most likely surgical phase.
  Reason about phase-specific characteristics (e.g., clipping phase shows
  clipper + clip action, dissection phase shows hook + dissect).

STEP 7 — Option Elimination & Self-Check:
  For EACH task, explicitly state which choices you eliminated and why.
  Verify cross-task consistency: instruments in IVT must appear in TASK 1,
  verbs in IVT must appear in TASK 2, targets must appear in TASK 3.

=== END REASONING — NOW OUTPUT JSON ===

"""


def build_multitask_prompt(tasks: Dict[str, Any], *, use_cot: bool = False) -> str:
    """Build ONE prompt that requests answers for all 6 tasks.

    Args:
        tasks: The benchmark sample tasks dict.
        use_cot: If True, inject the structured Chain-of-Thought reasoning
                 block before the JSON output instruction.
    """
    i_t = tasks["i_mcq"]
    v_t = tasks["v_mcq"]
    t_t = tasks["t_mcq"]
    ivt_t = tasks["ivt_mcq"]
    p_t = tasks["phase_mcq"]
    r_t = tasks["report_task"]

    prompt = f'''You are an expert surgical-video assistant for laparoscopic cholecystectomy.
You will be given THREE consecutive frames. The 3rd frame is the target (primary) frame; the first two frames provide temporal context.
Use ONLY visual evidence from the frames.
If uncertain, be conservative and avoid guessing.

Answer SIX tasks and output ONE JSON object ONLY.

IMPORTANT OUTPUT RULES
- For i/v/t tasks, you MUST select from the provided choices (strings). Output a LIST of chosen choice-strings.
- For ivt task, select from provided Triplet choices (strings). Output a LIST of chosen choice-strings.
- For phase, select EXACTLY ONE from the provided phase options. Output as an object: {{ "id": <int>, "name": <string> }}.
- For report, you MUST output TWO things:
  (a) a CANONICAL event list that uses EXACT choice strings (for robust rule-based scoring), and
  (b) a short structured intraoperative event report text.

  Canonical fields (for scoring; keep them machine-readable) MUST be nested under appendix:
  - appendix.canonical_ivt: LIST of items from TASK 4 choices.
    IMPORTANT: set appendix.canonical_ivt to be EXACTLY the same list as ivt_mcq.selected (copy it),
    unless you have a strong reason to omit an item.
  - appendix.canonical_phase: one phase option object from TASK 5 choices.
    IMPORTANT: set appendix.canonical_phase to be EXACTLY the same object as phase_mcq.selected (copy it).

  Report text requirements:
  - 3 numbered sections: (1) IVT events (2) anatomy/risk (3) next-step intent.
  - Be conservative. Do NOT assert specific anatomy (e.g., cystic duct/artery/CBD) unless clearly visible.
  - Avoid listing extra tools/actions/targets not supported by the frames.

TASK 1 - Instrument token(s)
Question: {i_t.get('question','')}
Choices:
{_format_choices(i_t.get('choices', []))}

TASK 2 - Verb/Action token(s)
Question: {v_t.get('question','')}
Choices:
{_format_choices(v_t.get('choices', []))}

TASK 3 - Target token(s)
Question: {t_t.get('question','')}
Choices:
{_format_choices(t_t.get('choices', []))}

TASK 4 - IVT triplet(s) (instrument, verb, target)
Question: {ivt_t.get('question','')}
Choices:
{_format_choices(ivt_t.get('choices', []))}

TASK 5 - Phase recognition
Question: {p_t.get('question','')}
Choices (id:name):
{_format_choices(p_t.get('choices', []))}

TASK 6 - Open-ended report
Question: {r_t.get('question','')}

'''

    # Inject CoT reasoning block if enabled
    if use_cot:
        cot_block = build_multitask_cot_block()
        prompt += "\n" + cot_block + "\n"

    # JSON schema — add "reasoning" field when CoT is active
    if use_cot:
        prompt += """Return EXACTLY this JSON schema (no extra keys, no markdown):

{
  "reasoning": "<your full step-by-step reasoning from the chain above>",
  "i_mcq": {
    "selected": ["<one or more items from TASK 1 choices>"]
  },
  "v_mcq": {
    "selected": ["<one or more items from TASK 2 choices>"]
  },
  "t_mcq": {
    "selected": ["<one or more items from TASK 3 choices>"]
  },
  "ivt_mcq": {
    "selected": ["<one or more items from TASK 4 choices>"]
  },
  "phase_mcq": {
    "selected": { "id": 0, "name": "preparation" }
  },
  "report_task": {
    "report_text": "<a short structured report with 3 numbered sections>",
    "appendix": {
      "canonical_ivt": ["<one or more items from TASK 4 choices>"],
      "canonical_phase": { "id": 0, "name": "preparation" }
    }
  }
}"""
    else:
        prompt += f"""Return EXACTLY this JSON schema (no extra keys, no markdown):

{{{{
  "i_mcq": {{{{
    "selected": ["<one or more items from TASK 1 choices>"]
  }}}},
  "v_mcq": {{{{
    "selected": ["<one or more items from TASK 2 choices>"]
  }}}},
  "t_mcq": {{{{
    "selected": ["<one or more items from TASK 3 choices>"]
  }}}},
  "ivt_mcq": {{{{
    "selected": ["<one or more items from TASK 4 choices>"]
  }}}},
  "phase_mcq": {{{{
    "selected": {{{{ "id": 0, "name": "preparation" }}}}
  }}}},
  "report_task": {{{{
    "report_text": "<a short structured report with 3 numbered sections>",
    "appendix": {{{{
      "canonical_ivt": ["<one or more items from TASK 4 choices>"],
      "canonical_phase": {{{{ "id": 0, "name": "preparation" }}}}
    }}}}
  }}}}
}}}}"""

    return prompt
