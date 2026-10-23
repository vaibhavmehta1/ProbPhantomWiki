from __future__ import annotations

import random

from .model import Person, ProbabilisticFact, World


FIRST_NAMES = (
    "Ada", "Bea", "Cleo", "Dara", "Elio", "Faye", "Galen", "Hana",
    "Ivo", "Juno", "Kira", "Luca", "Mara", "Nico", "Orla", "Pavel",
    "Quinn", "Rhea", "Soren", "Talia", "Uma", "Vera", "Wren", "Xavi",
    "Yara", "Zane",
)
LAST_NAMES = (
    "Ashford", "Briar", "Calder", "Dunley", "Evers", "Fenwick",
    "Grove", "Hart", "Ives", "Jasper", "Keene", "Linden", "Morrow",
    "North", "Oakes", "Pryce", "Quill", "Rowan", "Sable", "Thorne",
)
JOBS = (
    "beekeeper", "cartographer", "clockmaker", "glassblower", "librarian",
    "marine biologist", "meteorologist", "potter", "radio engineer",
    "restorer", "surveyor", "translator",
)
HOBBIES = (
    "astronomy", "birdwatching", "bookbinding", "chess", "foraging",
    "kite flying", "origami", "rowing", "stargazing", "woodcarving",
)


def _probability(rng: random.Random, low: float, high: float) -> float:
    return round(rng.uniform(low, high), 2)


def _distribution(rng: random.Random, size: int) -> list[float]:
    """Create a strictly positive discrete distribution with stable rounding."""
    weights = [rng.uniform(0.25, 1.0) for _ in range(size)]
    total = sum(weights)
    values = [round(weight / total, 6) for weight in weights[:-1]]
    values.append(round(1.0 - sum(values), 6))
    return values


def generate_world(
    size: int,
    rng: random.Random,
    *,
    average_friends: float = 2.0,
    evidence_probability: tuple[float, float] = (0.65, 0.98),
) -> World:
    """Create an acyclic family graph plus uncertain base facts.

    Probabilities mean that the corresponding statement is true in a sampled
    possible world. Relationship facts are independent Bernoulli variables;
    each person's job and hobby alternatives form annotated disjunctions.
    """
    if size < 4:
        raise ValueError("size must be at least 4")
    low, high = evidence_probability
    if not 0.0 < low <= high <= 1.0:
        raise ValueError("evidence probabilities must satisfy 0 < low <= high <= 1")

    all_names = [f"{first} {last}" for last in LAST_NAMES for first in FIRST_NAMES]
    if size > len(all_names):
        raise ValueError(f"size cannot exceed {len(all_names)} with the built-in names")
    rng.shuffle(all_names)
    people = [
        Person(key=f"p{i:04d}", name=all_names[i], gender=rng.choice(("female", "male")))
        for i in range(size)
    ]

    facts: list[ProbabilisticFact] = []
    attributes: dict[str, dict[str, tuple[str, ...]]] = {}
    for person in people:
        jobs = tuple(rng.sample(JOBS, 3))
        hobbies = tuple(rng.sample(HOBBIES, 3))
        attributes[person.key] = {"job": jobs, "hobby": hobbies}
        for job, probability in zip(jobs, _distribution(rng, len(jobs))):
            facts.append(ProbabilisticFact(probability, "job", (person.key, job)))
        for hobby, probability in zip(hobbies, _distribution(rng, len(hobbies))):
            facts.append(ProbabilisticFact(probability, "hobby", (person.key, hobby)))

    # Earlier identifiers are older generations. Each later character has one
    # or two candidate parents, so the graph cannot contain ancestry cycles.
    for child_index in range(2, size):
        eligible = people[:child_index]
        first_parent = rng.choice(eligible)
        facts.append(
            ProbabilisticFact(
                _probability(rng, low, high), "parent", (first_parent.key, people[child_index].key)
            )
        )
        if child_index >= 4 and rng.random() < 0.72:
            alternatives = [p for p in eligible if p.key != first_parent.key]
            second_parent = rng.choice(alternatives)
            facts.append(
                ProbabilisticFact(
                    _probability(rng, low, high),
                    "parent",
                    (second_parent.key, people[child_index].key),
                )
            )

    edge_probability = min(1.0, average_friends / max(1, size - 1))
    for left_index, left in enumerate(people):
        for right in people[left_index + 1 :]:
            if rng.random() < edge_probability:
                facts.append(
                    ProbabilisticFact(
                        _probability(rng, low, high), "friendship", (left.key, right.key)
                    )
                )

    return World(people=people, facts=facts, attributes=attributes)

