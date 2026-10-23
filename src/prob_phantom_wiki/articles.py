from __future__ import annotations

from collections import defaultdict

from .model import ProbabilisticFact, World


def _percent(probability: float) -> str:
    return f"{probability * 100:.0f}%"


def generate_articles(world: World) -> list[dict[str, str]]:
    people = world.people_by_key
    owned: dict[str, list[ProbabilisticFact]] = defaultdict(list)
    for fact in world.facts:
        if fact.predicate in {"job", "hobby"}:
            owned[fact.arguments[0]].append(fact)
        elif fact.predicate == "parent":
            owned[fact.arguments[0]].append(fact)
            owned[fact.arguments[1]].append(fact)
        elif fact.predicate == "friendship":
            owned[fact.arguments[0]].append(fact)
            owned[fact.arguments[1]].append(fact)

    articles: list[dict[str, str]] = []
    for person in world.people:
        sentences = [
            f"{person.name} is a fictional {'woman' if person.gender == 'female' else 'man'}."
        ]
        for fact in owned[person.key]:
            probability = _percent(fact.probability)
            if fact.predicate == "job" and fact.arguments[0] == person.key:
                sentences.append(
                    f"There is a {probability} probability that {person.name} works as a {fact.arguments[1]}."
                )
            elif fact.predicate == "hobby" and fact.arguments[0] == person.key:
                sentences.append(
                    f"There is a {probability} probability that {person.name}'s hobby is {fact.arguments[1]}."
                )
            elif fact.predicate == "parent":
                parent, child = (people[key].name for key in fact.arguments)
                sentences.append(
                    f"Records assign a {probability} probability that {parent} is a parent of {child}."
                )
            elif fact.predicate == "friendship":
                left, right = (people[key].name for key in fact.arguments)
                sentences.append(
                    f"Records assign a {probability} probability that {left} and {right} are friends."
                )
        articles.append(
            {
                "id": person.key,
                "title": person.name,
                "text": f"# {person.name}\n\n" + " ".join(sentences),
            }
        )
    return articles

