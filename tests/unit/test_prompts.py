"""The prompts must stay byte-identical to the ones used for the paper runs."""

import random

from sage_mas.prompts.roles import ROLE_POOL, SPECIALTIES, role_prompt, sample_roles
from sage_mas.prompts.templates import (
    CRITIC_SYSTEM,
    REWRITER_SYSTEM,
    first_nonempty_line,
    initial_prompt,
    rewrite_retry_prompt,
    update_prompt,
)
from sage_mas.seeding import sha256_text

# SHA-256 of each role prompt, as recorded in the paper runs (``selected_prompt_hashes``).
PAPER_ROLE_HASHES = {
    "WordProblemParser": "226461ada26047b16e4c69aa6c18a941dd75deb43e21665b0f91b8d88da2387c",
    "AlgebraicSolver": "1b353a2672b4c4fca8daacbdcbcc20545b4287a08cff7db817e8635fd09d310a",
    "ArithmeticCalculation": "e7366ee2f43bbf014cdd540de717a30c5d29596086947867e2ce533eded13ad6",
    "DiscreteMath": "088103dd867e570dd5661922c048b9b46af97146cfadfca597b7e008beef1bf9",
    "GeometryPrecalculus": "90eca66189690fe7bdca35bc5afe9c517f0b6ac295eb6e9473289ff0ed735627",
    "CompetitionMath": "e331020608b15da7d02365f8245b6f77b85e0463d3b4ed9a534a0e5360f33ece",
    "MultipleChoiceStrategist": "27dca29cef4cf94a4f2727f19cd3ec6bb17fe21ccd41e4d67388c5cc0606afb0",
    "BroadAcademicKnowledge": "03201c949c063e62e06b10ba798493fcca3d6b73dd8eb7a378d239f542fc739e",
    "SkepticalVerificationSolver": "7ea2f46ffbd251e7e2a2df019a5cb8a6a7312680882cad68385936475df0d781",
}


def test_role_prompts_match_paper():
    assert list(ROLE_POOL) == list(PAPER_ROLE_HASHES)
    for role, digest in PAPER_ROLE_HASHES.items():
        assert sha256_text(role_prompt(role)) == digest, role


def test_fixed_system_prompts_match_paper():
    assert sha256_text(CRITIC_SYSTEM) == "b8e22bbb931517b0cc50daab0f892068b9976b56f12d8f9fb402793140472853"
    assert sha256_text(REWRITER_SYSTEM) == "6b55a5adf97c5bfe3837db824ff1c87c0785368ddcde11e8debab7b3abc90d6f"


def test_role_prompts_share_contract_and_have_distinct_identities():
    identities = [first_nonempty_line(role_prompt(role)) for role in ROLE_POOL]
    assert len(set(identities)) == len(identities)
    assert all("\\boxed{}" in role_prompt(role) for role in SPECIALTIES)


def test_sample_roles_is_deterministic_and_without_replacement():
    sample = {"sample_id": "q1", "query": "What is 2+2?"}
    roles = sample_roles(sample, 2025, ROLE_POOL, 4)
    assert roles == sample_roles(dict(sample), 2025, ROLE_POOL, 4)
    assert len(set(roles)) == 4 and set(roles) <= set(ROLE_POOL)
    others = {tuple(sample_roles({"sample_id": f"q{i}", "query": "x"}, 2025, ROLE_POOL, 4)) for i in range(20)}
    assert len(others) > 1  # different questions get different teams


def test_sample_roles_does_not_touch_global_rng():
    random.seed(0)
    before = random.random()
    random.seed(0)
    sample_roles({"sample_id": "q"}, 1, ROLE_POOL, 4)
    assert random.random() == before


def test_templates():
    assert initial_prompt("Q").endswith("# Task\n```text\nQ\n```\n")
    text = update_prompt("Q", "mine", [(0, "a"), (2, "b")])
    assert text.index("## Peer 0 Answer") < text.index("## Peer 2 Answer")
    retry = rewrite_retry_prompt("BASE", ["rewrite_not_material", "rewrite_not_material"], "L1", "D1")
    assert retry.startswith("BASE\n\nRETRY_FEEDBACK_BEGIN\nFAILED_VALIDATION_CODES: rewrite_not_material\n")
    assert retry.endswith("RETRY_FEEDBACK_END")
