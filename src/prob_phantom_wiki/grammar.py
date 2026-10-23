from __future__ import annotations

import random
from dataclasses import dataclass, replace

from .model import World
from .problog import atom


RELATIONS: dict[str, tuple[str, int]] = {
    "friend": ("friends", 1),
    "parent": ("parents", 1),
    "child": ("children", 1),
    "mother": ("mothers", 1),
    "father": ("fathers", 1),
    "sibling": ("siblings", 2),
    "sister": ("sisters", 2),
    "brother": ("brothers", 2),
    "grandparent": ("grandparents", 2),
    "grandmother": ("grandmothers", 2),
    "grandfather": ("grandfathers", 2),
    "grandchild": ("grandchildren", 2),
    "aunt": ("aunts", 3),
    "uncle": ("uncles", 3),
    "cousin": ("cousins", 4),
}
ATTRIBUTES = ("job", "hobby")


@dataclass(frozen=True)
class Anchor:
    kind: str  # name | attribute
    value: str
    attribute: str | None = None
    display: str | None = None


@dataclass(frozen=True)
class RelationExpression:
    anchor: Anchor
    # Relations are stored from the anchor outwards. (friend, brother) means
    # "the brother of the friend of <anchor>".
    relations: tuple[str, ...]


@dataclass(frozen=True)
class QuestionPlan:
    kind: str  # entity | attribute | expected_count
    expression: RelationExpression
    target: str | None = None
    candidate: str | None = None
    candidate_display: str | None = None


def render_expression(expression: RelationExpression) -> str:
    anchor = expression.anchor
    if anchor.kind == "name":
        text = anchor.display or anchor.value
    else:
        text = f"the person whose {anchor.attribute} is {anchor.display or anchor.value}"
    for relation in expression.relations:
        text = f"the {relation} of {text}"
    return text


def render_question(plan: QuestionPlan) -> str:
    if plan.kind == "entity":
        return f"What is the probability that {render_event(plan)}?"
    if plan.kind == "attribute":
        return f"What is the probability that {render_event(plan)}?"
    expression = render_expression(plan.expression)
    plural = RELATIONS[plan.target or "friend"][0]
    return f"What is the expected number of {plural} that {expression} has?"


def render_event(plan: QuestionPlan) -> str:
    """Render a grounded entity/attribute plan as an English proposition."""
    if plan.candidate is None:
        raise ValueError("probability events require a grounded candidate")
    expression = render_expression(plan.expression)
    candidate = plan.candidate_display or plan.candidate
    if plan.kind == "entity":
        return f"{candidate} is {expression}"
    if plan.kind == "attribute":
        return f"the {plan.target} of {expression} is {candidate}"
    raise ValueError("expected-count plans are not Boolean events")


def _expression_goals(expression: RelationExpression) -> tuple[list[str], str]:
    goals: list[str] = []
    anchor = expression.anchor
    if anchor.kind == "name":
        current = atom(anchor.value)
    else:
        current = "V0"
        goals.append(f"{anchor.attribute}({current}, {atom(anchor.value)})")
    for index, relation in enumerate(expression.relations, start=1):
        following = f"V{index}"
        goals.append(f"{relation}({following}, {current})")
        current = following
    return goals, current


def compile_answer_clause(plan: QuestionPlan) -> str:
    goals, current = _expression_goals(plan.expression)
    if plan.kind == "entity":
        if plan.candidate is None:
            answer = current
        else:
            answer = "true"
            goals.append(f"{current} = {atom(plan.candidate)}")
    elif plan.kind == "attribute":
        if plan.candidate is None:
            answer = "Answer"
            goals.append(f"{plan.target}({current}, {answer})")
        else:
            answer = "true"
            goals.append(f"{plan.target}({current}, {atom(plan.candidate)})")
    else:
        answer = "Answer"
        goals.append(f"{plan.target}({answer}, {current})")
    body = ", ".join(goals) if goals else f"person({answer})"
    return f"answer({answer}) :- {body}."


def compile_event_clause(plan: QuestionPlan, predicate: str = "target_event") -> str:
    """Compile a grounded plan into a nullary Boolean ProbLog predicate."""
    if plan.candidate is None or plan.kind == "expected_count":
        raise ValueError("only grounded entity/attribute plans compile to events")
    goals, current = _expression_goals(plan.expression)
    if plan.kind == "entity":
        goals.append(f"{current} = {atom(plan.candidate)}")
    else:
        goals.append(f"{plan.target}({current}, {atom(plan.candidate)})")
    return f"{predicate} :- {', '.join(goals)}."


def derivation(plan: QuestionPlan) -> list[str]:
    """Return the concrete CFG production trace used by this plan."""
    if plan.kind == "entity":
        trace = [
            "S -> What is the probability that N is R?",
            f"N -> {plan.candidate_display or plan.candidate}",
        ]
    elif plan.kind == "attribute":
        trace = [
            "S -> What is the probability that A is AV?",
            "A -> the AN of R",
            f"AN -> {plan.target}",
            f"AV -> {plan.candidate_display or plan.candidate}",
        ]
    else:
        trace = [
            "S -> What is the expected number of RN_p that R_c has?",
            f"RN_p -> {RELATIONS[plan.target or 'friend'][0]}",
        ]
        if plan.expression.relations or plan.expression.anchor.kind == "attribute":
            trace.append("R_c -> R")
        else:
            trace.append("R_c -> N")
    for relation in reversed(plan.expression.relations):
        trace.extend(("R -> the RN of R_c", f"RN -> {relation}", "R_c -> R"))
    anchor = plan.expression.anchor
    if anchor.kind == "name":
        if trace and trace[-1] == "R_c -> R":
            trace[-1] = "R_c -> N"
        trace.append(f"N -> {anchor.display}")
    else:
        trace.extend(
            (
                "R -> the person whose AN is AV",
                f"AN -> {anchor.attribute}",
                f"AV -> {anchor.display or anchor.value}",
            )
        )
    return trace


def ground_candidate(
    plan: QuestionPlan,
    candidate: str,
    candidate_display: str | None = None,
) -> QuestionPlan:
    """Ground the answer slot, turning an open query into a probability event."""
    if plan.kind == "expected_count":
        raise ValueError("expected-count plans do not take a candidate")
    return replace(plan, candidate=candidate, candidate_display=candidate_display or candidate)


def sample_plan(world: World, rng: random.Random, max_depth: int) -> QuestionPlan:
    if max_depth < 1:
        raise ValueError("max_depth must be at least 1")
    # Dataset questions are grounded probability events. The expected_count
    # representation remains available for callers constructing plans
    # directly, but it is not sampled because it asks for an expectation rather
    # than the probability of a proposition.
    kind = rng.choice(("entity", "attribute"))
    anchor_kind = rng.choice(("name", "name", "attribute"))
    if anchor_kind == "name":
        person = rng.choice(world.people)
        anchor = Anchor("name", person.key, display=person.name)
    else:
        person = rng.choice(world.people)
        attribute = rng.choice(ATTRIBUTES)
        value = rng.choice(world.attributes[person.key][attribute])
        anchor = Anchor("attribute", value, attribute=attribute, display=value)

    # Recursive R -> relation(R_c) expansions. Attribute anchors can form a
    # complete entity question without adding a relation.
    minimum = 0 if anchor.kind == "attribute" or kind == "expected_count" else 1
    relation_count = rng.randint(minimum, max_depth)
    relations = tuple(rng.choice(tuple(RELATIONS)) for _ in range(relation_count))
    expression = RelationExpression(anchor=anchor, relations=relations)
    target = None
    if kind == "attribute":
        target = rng.choice(ATTRIBUTES)
    elif kind == "expected_count":
        target = rng.choice(tuple(RELATIONS))
    return QuestionPlan(kind=kind, expression=expression, target=target)


def reasoning_steps(plan: QuestionPlan) -> int:
    steps = sum(RELATIONS[relation][1] for relation in plan.expression.relations)
    if plan.expression.anchor.kind == "attribute":
        steps += 1
    if plan.kind == "attribute":
        steps += 1
    elif plan.kind == "expected_count":
        steps += RELATIONS[plan.target or "friend"][1]
    return steps

