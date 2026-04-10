from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple
import os

from .prompts import build_multitask_prompt
from .openai_mm import call_openai_multimodal
from .claude_mm import call_claude_multimodal
from .gemini_mm import call_gemini_multimodal
from .agent import SurgicalAgent, AgentConfig, AgentToolsConfig, SelfConsistencyConfig


def parse_agent_model_string(model: str) -> Tuple[str, str]:
    """
    Model string convention:
      - "agent:<base_model>"  => agent enabled, base model is after colon
      - otherwise => baseline single-call model
    Returns (kind, base_model).
    """
    if model.startswith("agent:"):
        base = model.split("agent:", 1)[1].strip()
        if not base:
            base = os.getenv("AGENT_BASE_MODEL", "gpt-5.2")
        return ("agent", base)
    return ("baseline", model)


class ModelRunner:
    def __init__(
        self,
        *,
        samples: Sequence[Any],
        agent_config: Optional[AgentConfig] = None,
    ) -> None:
        self.samples = list(samples)
        self.agent_config = agent_config or AgentConfig()
        self._agent: Optional[SurgicalAgent] = None

    def _get_agent(self) -> SurgicalAgent:
        if self._agent is None:
            self._agent = SurgicalAgent(config=self.agent_config)
            self._agent.build_global(self.samples)
        return self._agent

    def call(
        self,
        *,
        model: str,
        tasks: Dict[str, Any],
        image_data_urls: List[str],
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_output_tokens: int = 1500,
        sample_id: str = "",
    ) -> str:
        kind, base = parse_agent_model_string(model)
        if kind == "baseline":
            prompt = build_multitask_prompt(tasks)
            if "claude" in base.lower():
                return call_claude_multimodal(
                    model=base,
                    prompt=prompt,
                    image_data_urls=image_data_urls,
                    api_key=anthropic_api_key,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                )
            if "gemini" in base.lower():
                return call_gemini_multimodal(
                    model=base,
                    prompt=prompt,
                    image_data_urls=image_data_urls,
                    api_key=None,  # reads GEMINI_API_KEY from env
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                )
            return call_openai_multimodal(
                model=base,
                prompt=prompt,
                image_data_urls=image_data_urls,
                api_key=api_key,
                base_url=base_url,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )

        # agent mode
        agent = self._get_agent()
        # For Claude models, pass anthropic key; for Gemini, use gemini key from env
        if "claude" in base.lower():
            effective_key = anthropic_api_key
        elif "gemini" in base.lower():
            effective_key = None  # gemini_mm reads GEMINI_API_KEY from env
        else:
            effective_key = api_key
        out = agent.solve(
            sample_id=sample_id,
            tasks=tasks,
            image_data_urls=image_data_urls,
            base_model=base,
            api_key=effective_key,
            base_url=base_url,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        import json
        return json.dumps(out, ensure_ascii=False)
