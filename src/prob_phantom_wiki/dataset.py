from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from .articles import generate_articles
from .grammar import (
    compile_answer_clause,
    compile_event_clause,
    derivation,
    ground_candidate,
    reasoning_steps,
    render_event,
    render_question,
    sample_plan,
)
from .problog import atom, evaluate_answers, evaluate_event, world_program
from .world import generate_world


@dataclass(frozen=True)
class DatasetConfig:
    size: int = 10
    questions: int = 20
    max_depth: int = 2
    seed: int = 1
    average_friends: float = 2.0
    min_answer_probability: float = 1e-9
    question_types: tuple[str, ...] = (
        "prior",
        "posterior",
        "conditional",
        "joint",
        "disjunction",
        "comparison",
    )
    report_sensitivity: float = 0.90
    report_false_positive_rate: float = 0.05


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _sample_grounded_plan(world, program, labels, config, rng):
    """Sample an event plan and ground its answer slot to a supported value."""
    for _ in range(100):
        plan = sample_plan(world, rng, config.max_depth)
        probabilities = {
            key: probability
            for key, probability in evaluate_answers(program, compile_answer_clause(plan)).items()
            if probability >= config.min_answer_probability
        }
        if probabilities:
            candidate = rng.choice(sorted(probabilities))
            if (
                plan.kind == "attribute"
                and plan.expression.anchor.kind == "attribute"
                and not plan.expression.relations
                and plan.target == plan.expression.anchor.attribute
                and candidate == plan.expression.anchor.value
            ):
                # Avoid questions equivalent to "what is the probability that
                # the job of a person whose job is X is X?"
                continue
            grounded = ground_candidate(plan, candidate, labels.get(candidate, candidate))
            return grounded, probabilities[candidate]
    raise RuntimeError("could not sample a supported probability event")


def _query_source(clauses: list[str], query: str = "target_event") -> str:
    return "\n".join((*clauses, f"query({query})."))


def _base_record(plan, question: str, clause: str) -> dict[str, object]:
    return {
        "question": question,
        "event": render_event(plan),
        "problog_query": clause,
        "cfg_derivation": derivation(plan),
        "relation_depth": len(plan.expression.relations),
        "reasoning_steps": reasoning_steps(plan),
    }


def _generate_question(world, program, labels, config, rng, question_type: str) -> dict[str, object]:

    if question_type == "conditional":
        person = rng.choice(world.people)
        attribute = rng.choice(("job", "hobby"))
        target_value, ruled_out = rng.sample(world.attributes[person.key][attribute], 2)
        target_clause = f"target_event :- {attribute}({atom(person.key)}, {atom(target_value)})."
        evidence_clause = f"evidence_event :- {attribute}({atom(person.key)}, {atom(ruled_out)})."
        prior = evaluate_event(program, [target_clause])
        posterior = evaluate_event(
            program,
            [target_clause, evidence_clause],
            evidence=[("evidence_event", False)],
        )
        question = (
            f"Given that {person.name}'s {attribute} is not {ruled_out}, what is the probability "
            f"that {person.name}'s {attribute} is {target_value}?"
        )
        clauses = [target_clause, evidence_clause, "evidence(evidence_event, false)."]
        return {
            "question": question,
            "question_type": "conditional",
            "answer_type": "probability",
            "answer_probability": round(posterior, 8),
            "prior_probability": round(prior, 8),
            "event": f"{person.name}'s {attribute} is {target_value}",
            "evidence": {
                "event": f"{person.name}'s {attribute} is {ruled_out}",
                "value": False,
            },
            "problog_query": _query_source(clauses),
            "cfg_derivation": [
                "S -> Given that E is false, what is the probability that E?",
                "E -> N's AN is AV",
                f"N -> {person.name}",
                f"AN -> {attribute}",
                f"AV -> {ruled_out}",
                "E -> N's AN is AV",
                f"N -> {person.name}",
                f"AN -> {attribute}",
                f"AV -> {target_value}",
            ],
            "relation_depth": 0,
            "reasoning_steps": 1,
        }

    first, first_probability = _sample_grounded_plan(world, program, labels, config, rng)
    first_event = render_event(first)
    first_clause = compile_event_clause(first, "event_a")

    if question_type == "prior":
        target_clause = compile_event_clause(first, "target_event")
        record = _base_record(first, render_question(first), _query_source([target_clause]))
        record.update(
            answer_type="probability",
            question_type="prior",
            answer_probability=round(first_probability, 8),
        )
        return record

    if question_type == "posterior":
        positive = rng.choice((True, False))
        sensitivity = config.report_sensitivity
        false_positive = config.report_false_positive_rate
        target_clause = compile_event_clause(first, "target_event")
        report_clauses = [
            f"{sensitivity:.6g}::report_positive :- target_event.",
            f"{false_positive:.6g}::report_positive :- \\+ target_event.",
        ]
        posterior = evaluate_event(
            program,
            [target_clause, *report_clauses],
            evidence=[("report_positive", positive)],
        )
        result_word = "positive" if positive else "negative"
        question = (
            f"A report with a {sensitivity * 100:.0f}% true-positive rate and a "
            f"{false_positive * 100:.0f}% false-positive rate returned {result_word} for the claim "
            f"that {first_event}. Given this report, what is the probability that {first_event}?"
        )
        evidence_line = f"evidence(report_positive, {'true' if positive else 'false'})."
        clauses = [target_clause, *report_clauses, evidence_line]
        record = _base_record(first, question, _query_source(clauses))
        record.update(
            answer_type="probability",
            question_type="posterior",
            prior_probability=round(first_probability, 8),
            answer_probability=round(posterior, 8),
            evidence={
                "event": "report_positive",
                "value": positive,
                "sensitivity": sensitivity,
                "false_positive_rate": false_positive,
            },
        )
        record["cfg_derivation"] = [
            "S -> Given O, what is the probability that E?",
            "O -> noisy report about E",
            *derivation(first),
        ]
        return record

    second, second_probability = _sample_grounded_plan(world, program, labels, config, rng)
    second_event = render_event(second)
    if second_event == first_event:
        raise RuntimeError("sampled duplicate events")
    second_clause = compile_event_clause(second, "event_b")

    if question_type in {"joint", "disjunction"}:
        if question_type == "joint":
            target_clauses = ["target_event :- event_a, event_b."]
            connective = "and"
            start_rule = "S -> What is the probability that E and E?"
        else:
            target_clauses = ["target_event :- event_a.", "target_event :- event_b."]
            connective = "or"
            start_rule = "S -> What is the probability that E or E?"
        clauses = [first_clause, second_clause, *target_clauses]
        probability = evaluate_event(program, clauses)
        question = f"What is the probability that {first_event} {connective} {second_event}?"
        record = _base_record(first, question, _query_source(clauses))
        record.update(
            answer_type="probability",
            question_type=question_type,
            answer_probability=round(probability, 8),
            events=[first_event, second_event],
            cfg_derivation=[start_rule, *derivation(first), *derivation(second)],
            relation_depth=len(first.expression.relations) + len(second.expression.relations),
            reasoning_steps=reasoning_steps(first) + reasoning_steps(second),
        )
        return record

    if question_type == "comparison":
        if abs(first_probability - second_probability) < 1e-10:
            answer = "tie"
        elif first_probability > second_probability:
            answer = "A"
        else:
            answer = "B"
        question = f"Which claim is more likely: (A) {first_event}; or (B) {second_event}?"
        record = _base_record(first, question, _query_source([first_clause, second_clause], "event_a"))
        record.update(
            answer_type="choice",
            question_type="comparison",
            answer=answer,
            probabilities={"A": round(first_probability, 8), "B": round(second_probability, 8)},
            events={"A": first_event, "B": second_event},
            problog_query="\n".join((first_clause, second_clause, "query(event_a).", "query(event_b).")),
            cfg_derivation=[
                "S -> Which claim is more likely: E or E?",
                *derivation(first),
                *derivation(second),
            ],
            relation_depth=len(first.expression.relations) + len(second.expression.relations),
            reasoning_steps=reasoning_steps(first) + reasoning_steps(second),
        )
        return record

    raise ValueError(f"unsupported question type: {question_type}")


def generate_dataset(output_dir: str | Path, config: DatasetConfig | None = None) -> dict[str, object]:
    config = config or DatasetConfig()
    if config.questions < 1:
        raise ValueError("questions must be at least 1")
    allowed_types = {"prior", "posterior", "conditional", "joint", "disjunction", "comparison"}
    if not config.question_types or not set(config.question_types) <= allowed_types:
        raise ValueError(f"question_types must be a non-empty subset of {sorted(allowed_types)}")
    if not 0 < config.report_false_positive_rate < config.report_sensitivity < 1:
        raise ValueError("report rates must satisfy 0 < false-positive rate < sensitivity < 1")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(config.seed)
    world = generate_world(config.size, rng, average_friends=config.average_friends)
    program = world_program(world)
    labels = {person.key: person.name for person in world.people}

    records: list[dict[str, object]] = []
    seen: set[str] = set()
    question_schedule = list(config.question_types)
    rng.shuffle(question_schedule)
    max_attempts = max(200, config.questions * 100)
    for _ in range(max_attempts):
        if len(records) >= config.questions:
            break
        try:
            question_type = question_schedule[len(records) % len(question_schedule)]
            record = _generate_question(world, program, labels, config, rng, question_type)
        except RuntimeError:
            continue
        question = str(record["question"])
        if question in seen:
            continue
        record["id"] = f"q{len(records):05d}"
        records.append(record)
        seen.add(question)
    if len(records) < config.questions:
        raise RuntimeError(
            f"could generate only {len(records)} non-empty unique questions after {max_attempts} attempts"
        )

    (output / "program.pl").write_text(program, encoding="utf-8")
    _write_jsonl(output / "corpus.jsonl", generate_articles(world))
    _write_jsonl(output / "questions.jsonl", records)
    manifest: dict[str, object] = {
        "format_version": 1,
        "semantics": (
            "Relationship facts are independent Bernoulli variables; each job and hobby is a mutually exclusive "
            "annotated disjunction under ProbLog distribution semantics. "
            "Questions ask for prior, conditional, noisy-observation posterior, joint, disjunctive, or comparative "
            "event probabilities."
        ),
        "config": asdict(config),
        "files": {"program": "program.pl", "corpus": "corpus.jsonl", "questions": "questions.jsonl"},
        "generated": {"articles": len(world.people), "questions": len(records), "base_facts": len(world.facts)},
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest

