# ProbPhantomWiki

ProbPhantomWiki is a small, reproducible implementation of the central
PhantomWiki idea using **ProbLog instead of ordinary Prolog**. It generates:

1. a fictional universe represented by probabilistic base predicates;
2. one natural-language article per fictional person;
3. questions produced by a bounded context-free grammar (CFG); and
4. exact marginal answer probabilities computed by ProbLog.

It is an independent prototype inspired by the
[PhantomWiki paper](https://arxiv.org/abs/2502.20377), not a fork of the
authors' package.

## Install and run

Python 3.10 or newer is required. ProbLog's Python package is the only runtime
dependency; SWI-Prolog is not required.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"

prob-phantom-wiki out/demo --size 10 --questions 20 --max-depth 2 --seed 7
```

Equivalently, without using the installed command:

```bash
python -m prob_phantom_wiki out/demo --size 10 --questions 20 --max-depth 2
```

The output directory contains:

- `program.pl`: the complete ProbLog knowledge base;
- `corpus.jsonl`: the public-facing fictional articles;
- `questions.jsonl`: questions, query clauses, CFG derivations, and gold
  probabilistic answers; and
- `manifest.json`: configuration and semantics.

For evaluation, give a model only `corpus.jsonl` and each `question` field.
Keep `program.pl`, `problog_query`, and the answer fields hidden.

By default, generation mixes six question families. Select a subset with:

```bash
prob-phantom-wiki out/posterior \
  --question-types posterior,conditional \
  --questions 40 --seed 7
```

The available types are `prior`, `posterior`, `conditional`, `joint`,
`disjunction`, and `comparison`. For noisy-report posterior questions, the
observation model is configurable:

```bash
prob-phantom-wiki out/reports \
  --question-types posterior \
  --report-sensitivity 0.90 \
  --report-false-positive-rate 0.05
```

## ProbLog model

The generated program separates base predicates from derived predicates.
Relationship facts are independent Bernoulli variables, while jobs and
hobbies are mutually exclusive annotated disjunctions:

```prolog
0.91::parent('p0001', 'p0007').
0.73::friendship('p0002', 'p0007').
0.50::hobby('p0007', 'birdwatching');
0.30::hobby('p0007', 'chess');
0.20::hobby('p0007', 'rowing').
```

Rules define the relational vocabulary:

```prolog
friend(X,Y) :- friendship(X,Y).
friend(X,Y) :- friendship(Y,X).
sibling(X,Y) :- parent(P,X), parent(P,Y), X \= Y.
grandparent(X,Y) :- parent(X,P), parent(P,Y).
```

Thus a multi-hop conclusion has a probability induced by all of its proofs,
not a hand-multiplied score. ProbLog handles shared facts and alternative
proofs under its distribution semantics. The articles verbalize exactly the
same base evidence (for example, "Records assign a 73% probability ..."), so
the corpus and executable world cannot disagree.

## How question generation uses a CFG

The paper uses entity, attribute, and count questions. This project grounds
entity and attribute templates as events, then composes those events into
several probabilistic question forms:

```text
S   -> What is the probability that N is R?
     | What is the probability that A is AV?
     | Given O, what is the probability that E?
     | What is the probability that E and E?
     | What is the probability that E or E?
     | Which claim is more likely: E or E?

E   -> N is R | A is AV
O   -> noisy-report(E) | not(E)

R   -> the RN of R_c
     | the person whose AN is AV
R_c -> R | N
A   -> the AN of R
RN  -> friend | parent | sibling | grandparent | aunt | ...
AN  -> job | hobby
AV  -> <generated attribute value>
N   -> <generated person name>
```

`R_c -> R` is the recursive production. Applying it repeatedly creates nested
questions. At depth three, for example:

```text
What is the probability that Bea Hart is the sister of the friend of the parent of Ada North?
```

The generator does not construct English and logic independently. It first
samples a typed syntax tree (`QuestionPlan`) from the bounded grammar, then
uses two interpreters over that same tree:

```text
QuestionPlan syntax tree
        |                         |
        v                         v
 English renderer          ProbLog compiler
        |                         |
        v                         v
 "What is the         target_event :- parent(V1,p0),
 probability ...?"                    friend(V2,V1),
                                     sister(X,V2), X=p9.
                     query(target_event).
```

This shared intermediate representation is the key consistency mechanism. A
question cannot silently drift away from its query. Every question record also
contains its concrete `cfg_derivation`, `relation_depth`, and estimated
`reasoning_steps`, making difficulty inspectable.

The six generated families are:

- **Prior:** the marginal probability of one grounded event.
- **Posterior:** the probability after a positive or negative noisy report;
  report sensitivity and false-positive rate are stated in the question.
- **Conditional:** the posterior for one job/hobby value after another value
  is ruled out. This exercises the annotated-disjunction semantics.
- **Joint:** the probability that two events both hold.
- **Disjunction:** the probability that at least one of two events holds.
- **Comparison:** which of two events has the greater marginal probability.

The ordinary PhantomWiki wording "How many ...?" assumes a deterministic
world. Here the count is a random variable and needs an explicit statistic,
so ambiguous integer count questions are not sampled.

## Output example

```json
{
  "question": "What is the probability that Cleo Hart is the friend of Ada North?",
  "answer_type": "probability",
  "answer_probability": 0.73,
  "question_type": "prior",
  "event": "Cleo Hart is the friend of Ada North",
  "problog_query": "target_event :- friend(V1, 'p0000'), V1 = 'p0004'.\nquery(target_event).",
  "cfg_derivation": [
    "S -> What is the probability that N is R?",
    "N -> Cleo Hart",
    "R -> the RN of R_c",
    "RN -> friend",
    "R_c -> N",
    "N -> Ada North"
  ]
}
```

A conditional item looks like:

```json
{
  "question": "Given that Mara's job is not librarian, what is the probability that Mara's job is teacher?",
  "question_type": "conditional",
  "prior_probability": 0.30,
  "answer_probability": 0.60,
  "evidence": {"event": "Mara's job is librarian", "value": false}
}
```

## Development

```bash
pytest -q
```

The generator is deterministic for a fixed configuration and seed.

