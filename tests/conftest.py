import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sage_mas.answer.extractor import RegexExtractor
from sage_mas.config import SageConfig
from sage_mas.context import QuestionContext
from sage_mas.llm.replay import ScriptedLLM


def make_context(respond=lambda request: "\\boxed{1}", config=None, n=4, **kwargs) -> QuestionContext:
    """A question context backed by a scripted LLM and the regex extractor."""
    config = config or SageConfig(num_agents=n)
    return QuestionContext(
        query=kwargs.get("query", "What is 1?"),
        choices=kwargs.get("choices", []),
        seed=kwargs.get("seed", 0),
        config=config,
        llm=ScriptedLLM(respond),
        extractor=RegexExtractor(),
        backbones=["m"] * config.num_agents,
        roles=list(config.roles[: config.num_agents]),
        system_prompts=[f"prompt {i}\nbody" for i in range(config.num_agents)],
        review_rng=random.Random(0),
    )


@pytest.fixture
def context():
    return make_context
