import random

from prob_phantom_wiki.grammar import (
    Anchor,
    QuestionPlan,
    RelationExpression,
    compile_answer_clause,
    derivation,
    ground_candidate,
    render_question,
    sample_plan,
)
from prob_phantom_wiki.world import generate_world


def test_nested_question_and_query_are_compiled_from_same_plan():
    open_plan = QuestionPlan(
        kind="entity",
        expression=RelationExpression(
            anchor=Anchor("name", "p0000", display="Ada North"),
            relations=("parent", "friend", "sister"),
        ),
    )
    plan = ground_candidate(open_plan, "p0009", "Bea Hart")
    assert render_question(plan) == (
        "What is the probability that Bea Hart is the sister of the friend of the parent of Ada North?"
    )
    assert compile_answer_clause(plan) == (
        "answer(true) :- parent(V1, 'p0000'), friend(V2, V1), sister(V3, V2), V3 = 'p0009'."
    )


def test_attribute_anchor_compiles_to_a_constraint():
    open_plan = QuestionPlan(
        kind="attribute",
        expression=RelationExpression(
            anchor=Anchor("attribute", "chess", attribute="hobby", display="chess"),
            relations=("brother",),
        ),
        target="job",
    )
    plan = ground_candidate(open_plan, "librarian")
    assert render_question(plan) == (
        "What is the probability that the job of the brother of the person whose hobby is chess is librarian?"
    )
    assert compile_answer_clause(plan) == (
        "answer(true) :- hobby(V0, 'chess'), brother(V1, V0), job(V1, 'librarian')."
    )


def test_sampling_respects_relation_depth():
    world = generate_world(8, random.Random(4))
    rng = random.Random(9)
    for _ in range(100):
        plan = sample_plan(world, rng, max_depth=2)
        assert len(plan.expression.relations) <= 2


def test_expected_count_allows_a_direct_name_subject():
    plan = QuestionPlan(
        kind="expected_count",
        expression=RelationExpression(Anchor("name", "p0000", display="Ada North"), ()),
        target="friend",
    )
    assert render_question(plan) == "What is the expected number of friends that Ada North has?"
    assert compile_answer_clause(plan) == "answer(Answer) :- friend(Answer, 'p0000')."
    assert derivation(plan) == [
        "S -> What is the expected number of RN_p that R_c has?",
        "RN_p -> friends",
        "R_c -> N",
        "N -> Ada North",
    ]

