import random

import pytest

problog = pytest.importorskip("problog")

from prob_phantom_wiki.grammar import Anchor, QuestionPlan, RelationExpression, compile_answer_clause
from prob_phantom_wiki.model import Person, ProbabilisticFact, World
from prob_phantom_wiki.problog import evaluate_answers, evaluate_event, world_program
from prob_phantom_wiki.world import generate_world


def tiny_world():
    return World(
        people=[
            Person("ada", "Ada North", "female"),
            Person("bea", "Bea Hart", "female"),
            Person("cal", "Cal Rowan", "male"),
        ],
        facts=[
            ProbabilisticFact(0.8, "parent", ("ada", "cal")),
            ProbabilisticFact(0.5, "friendship", ("bea", "cal")),
            ProbabilisticFact(0.6, "hobby", ("bea", "chess")),
        ],
    )


def test_problog_evaluates_a_direct_relation_probability():
    plan = QuestionPlan(
        "entity", RelationExpression(Anchor("name", "cal", display="Cal Rowan"), ("friend",))
    )
    answers = evaluate_answers(world_program(tiny_world()), compile_answer_clause(plan))
    assert answers["bea"] == pytest.approx(0.5)


def test_problog_evaluates_a_two_fact_chain():
    plan = QuestionPlan(
        "attribute",
        RelationExpression(Anchor("name", "cal", display="Cal Rowan"), ("friend",)),
        "hobby",
    )
    answers = evaluate_answers(world_program(tiny_world()), compile_answer_clause(plan))
    assert answers["chess"] == pytest.approx(0.3)


def test_posterior_from_a_noisy_positive_report():
    program = "0.2::rain."
    clauses = [
        "target_event :- rain.",
        "0.9::report_positive :- target_event.",
        "0.05::report_positive :- \\+ target_event.",
    ]
    posterior = evaluate_event(
        program,
        clauses,
        evidence=[("report_positive", True)],
    )
    assert posterior == pytest.approx((0.9 * 0.2) / (0.9 * 0.2 + 0.05 * 0.8))


def test_joint_and_disjunction_probabilities():
    program = "0.5::a.\n0.4::b."
    base = ["event_a :- a.", "event_b :- b."]
    joint = evaluate_event(program, [*base, "target_event :- event_a, event_b."])
    either = evaluate_event(
        program,
        [*base, "target_event :- event_a.", "target_event :- event_b."],
    )
    assert joint == pytest.approx(0.2)
    assert either == pytest.approx(0.7)


def test_generated_attributes_are_annotated_distributions():
    world = generate_world(5, random.Random(3))
    program = world_program(world)
    person = world.people[0]
    answers = evaluate_answers(
        program,
        f"answer(Value) :- job('{person.key}', Value).",
    )
    assert len(answers) == 3
    assert sum(answers.values()) == pytest.approx(1.0)

