from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Person:
    key: str
    name: str
    gender: str


@dataclass(frozen=True)
class ProbabilisticFact:
    probability: float
    predicate: str
    arguments: tuple[str, ...]


@dataclass
class World:
    people: list[Person]
    facts: list[ProbabilisticFact]
    attributes: dict[str, dict[str, tuple[str, ...]]] = field(default_factory=dict)

    @property
    def people_by_key(self) -> dict[str, Person]:
        return {person.key: person for person in self.people}

