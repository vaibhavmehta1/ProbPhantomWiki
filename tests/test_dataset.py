import json

import pytest

pytest.importorskip("problog")

from prob_phantom_wiki import DatasetConfig, generate_dataset


def test_dataset_is_complete_and_reproducible(tmp_path):
    config = DatasetConfig(size=8, questions=6, max_depth=2, seed=11)
    first = tmp_path / "first"
    second = tmp_path / "second"
    manifest = generate_dataset(first, config)
    generate_dataset(second, config)

    assert manifest["generated"]["articles"] == 8
    assert manifest["generated"]["questions"] == 6
    for filename in ("program.pl", "corpus.jsonl", "questions.jsonl", "manifest.json"):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()

    questions = [json.loads(line) for line in (first / "questions.jsonl").read_text().splitlines()]
    assert all(
        item.get("answer_probability") is not None or item.get("answer") in {"A", "B", "tie"}
        for item in questions
    )
    assert {item["question_type"] for item in questions} <= {
        "prior", "posterior", "conditional", "joint", "disjunction", "comparison"
    }
    assert all("cfg_derivation" in item for item in questions)


@pytest.mark.parametrize(
    "question_type",
    ["prior", "posterior", "conditional", "joint", "disjunction", "comparison"],
)
def test_each_question_family_can_be_generated(tmp_path, question_type):
    output = tmp_path / question_type
    generate_dataset(
        output,
        DatasetConfig(
            size=6,
            questions=1,
            max_depth=1,
            seed=19,
            question_types=(question_type,),
        ),
    )
    item = json.loads((output / "questions.jsonl").read_text().strip())
    assert item["question_type"] == question_type
    assert "query(" in item["problog_query"]

