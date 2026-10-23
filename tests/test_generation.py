import pytest

from hyde_rag.generation.abstention import calibrate_tau
from hyde_rag.generation.answer import ABSTAIN_TEXT, build_request, gate, gated_answer, parse_answer
from hyde_rag.generation.citations import split_sentences, validate
from hyde_rag.generation.context import Chunk, format_context

CHUNKS = [
    Chunk(
        "fda_food_code_2026",
        87,
        3,
        "Cooked food shall be cooled from 57C to 21C within 2 hours.",
        title="Food Code",
        section="3-501.14 Cooling",
        score=6.1,
    ),
    Chunk("fsis_appendix_b", 4, 0, "Cooling must then reach 5C within a further 4 hours.", score=2.0),
]
IDS = {c.chunk_id for c in CHUNKS}


def test_chunk_ids_and_context_block():
    assert CHUNKS[0].chunk_id == "fda_food_code_2026:87:3"
    ctx = format_context(CHUNKS)
    assert ctx.startswith("[fda_food_code_2026:87:3] Food Code | 3-501.14 Cooling\nCooked food")
    assert "[fsis_appendix_b:4:0]\nCooling must" in ctx


def test_valid_answer_with_multiple_citations():
    answer = (
        "Cool cooked food from 57C to 21C within 2 hours [fda_food_code_2026:87:3]. "
        "Then reach 5C within 4 more hours [fsis_appendix_b:4:0, fda_food_code_2026:87:3]."
    )
    report = validate(answer, IDS)
    assert report.valid
    assert [s.cited_ids for s in report.sentences] == [
        ["fda_food_code_2026:87:3"],
        ["fsis_appendix_b:4:0", "fda_food_code_2026:87:3"],
    ]
    assert report.sentences[0].text == "Cool cooked food from 57C to 21C within 2 hours."


def test_citation_after_full_stop_or_on_its_own_line_attaches_to_sentence():
    assert validate("Cooling takes 2 hours. [fda_food_code_2026:87:3]", IDS).valid
    report = validate(
        "Cooling takes 2 hours.\n[fda_food_code_2026:87:3]\nNext step is 5C [fsis_appendix_b:4:0].", IDS
    )
    assert report.valid and len(report.sentences) == 2


def test_unknown_ids_and_uncited_sentences_are_flagged():
    report = validate("Cooling takes 2 hours [made_up:1:1]. This sentence has no source.", IDS)
    assert report.unknown_ids == ["made_up:1:1"]
    assert report.uncited == [1]
    assert not report.valid


def test_abbreviations_do_not_split_sentences():
    assert len(split_sentences("Pathogens, e.g. listeria, grow slowly [a:1:0]. Cool fast [a:1:0].")) == 2


def test_claim_mode_parsing_and_abstention():
    ok = parse_answer(
        "Verdict: SUPPORTED\nCooling within 2 hours is required [fda_food_code_2026:87:3].", CHUNKS, "claim"
    )
    assert ok.verdict == "SUPPORTED" and not ok.abstained and ok.citations.valid
    nf = parse_answer(f"Verdict: NOT FOUND\n{ABSTAIN_TEXT}", CHUNKS, "claim")
    assert nf.verdict == "NOT FOUND" and nf.abstained
    qa = parse_answer(ABSTAIN_TEXT, CHUNKS)
    assert qa.abstained and qa.citations.valid


def test_retrieval_gate():
    assert not gate(CHUNKS, tau=5.0)  # best score 6.1 >= 5
    assert gate(CHUNKS, tau=7.0)
    assert not gate(CHUNKS, tau=None)
    assert gate([], tau=None)
    g = gated_answer(CHUNKS, "claim")
    assert g.gated and g.abstained and g.verdict == "NOT FOUND"


def test_request_is_deterministic_and_grounded():
    req = build_request("How fast must cooked food cool?", CHUNKS, "m", reasoning_effort="low")
    assert req.temperature == 0.0
    system, user = req.messages[0][1], req.messages[1][1]
    assert ABSTAIN_TEXT in system and "[fda_food_code_2026:87:3]" in user


def test_calibrate_tau_respects_false_refusal_cap():
    answerable = [5.0, 6.0, 7.0, 8.0, 0.5] * 4  # one low-scoring answerable per five
    unanswerable = [0.1, 0.2, 0.3, 4.0]
    cal = calibrate_tau(answerable, unanswerable, max_false_refusal=0.05)
    assert cal.tau is not None and cal.false_refusal <= 0.05
    assert cal.recall == pytest.approx(0.75)  # catches 0.1/0.2/0.3 but not 4.0
    assert calibrate_tau(answerable, []).tau is None


def test_judge_request_and_parsing():
    import json

    from hyde_rag.eval.judge import FAITHFULNESS_SCHEMA, faithfulness_request, parse_faithfulness

    ans = parse_answer(
        "Cool from 57C to 21C within 2 hours [fda_food_code_2026:87:3]. "
        "Then reach 5C within 4 hours [fsis_appendix_b:4:0, fda_food_code_2026:87:3].",
        CHUNKS,
    )
    req = faithfulness_request(ans, "judge-model")
    user = req.messages[1][1]
    assert req.json_schema == FAITHFULNESS_SCHEMA and req.temperature == 0.0
    assert "Sentence 1: Then reach 5C within 4 hours." in user
    assert "Passage fsis_appendix_b:4:0: Cooling must then reach 5C" in user

    raw = json.dumps(
        {
            "claims": [
                {"sentence": 0, "claim": "cool to 21C within 2 h", "verdict": "supported"},
                {"sentence": 1, "claim": "reach 5C within 4 h", "verdict": "supported"},
                {"sentence": 1, "claim": "extra unsupported detail", "verdict": "not_supported"},
            ],
            "citations": [
                {"sentence": 0, "passage_id": "fda_food_code_2026:87:3", "supports": True},
                {"sentence": 1, "passage_id": "fsis_appendix_b:4:0", "supports": True},
                {"sentence": 1, "passage_id": "fda_food_code_2026:87:3", "supports": False},
                {"sentence": 5, "passage_id": "invented:1:1", "supports": True},  # ignored: not made
            ],
        }
    )
    res = parse_faithfulness(raw, ans)
    assert res.faithfulness == pytest.approx(2 / 3)
    assert res.citation_precision == pytest.approx(2 / 3)


@pytest.mark.parametrize(
    "answer",
    [
        "The rule was issued by the U.S. Department of Agriculture (USDA). [fc:424:0]",
        "No. A foreign facility is not required to submit a report. [rfr:30:0]",
        "Procedures should:\n1. Be scientifically valid. [lm:41:0]\n2. Name the sampling sites. [lm:41:0]",
        "Keep it cold, e.g. Below 41°F. [fc:1:0]",
    ],
)
def test_fragments_join_the_next_sentence(answer):
    report = validate(answer, {"fc:424:0", "rfr:30:0", "lm:41:0", "fc:1:0"})
    assert report.valid, (report.uncited, [s.text for s in report.sentences])


def test_real_uncited_sentences_are_still_flagged():
    report = validate("Cool food within 2 hours. [fc:1:0] Then store it below 5C for later use.", {"fc:1:0"})
    assert report.uncited == [1]
    report = validate(
        "Repeat the check at step 2. [fc:1:0] Record the result in the cooling log.", {"fc:1:0"}
    )
    assert report.uncited == [1]


def test_fullwidth_bracket_citations_count():
    report = validate("Three agencies regulate pesticides: EPA, FDA and USDA FSIS【fc:22:1】.", {"fc:22:1"})
    assert report.valid and report.sentences[0].cited_ids == ["fc:22:1"]
