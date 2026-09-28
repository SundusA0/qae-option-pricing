# AURA

**User-Aligned Metacognitive Routing for AI Systems**

> AURA asks a broader question than *which model should answer?* It asks **what the system should do next, how much capability it should allocate, and which trade-offs should govern that decision.**

AURA is an early-stage research project on **adaptive inference intervention** for AI systems. The core idea is that a difficult-looking request does not always need a larger model. It may instead need current information, a tool, a clarification from the user, or an independent verification step.

## Status

**Research design draft — v0.2. No experimental results yet.**

The immediate goal is to validate the benchmark and evaluation protocol before training any routing model.

## Why AURA?

Model routing is usually framed as:

> Given a request, which model should answer it?

AURA treats that as only one part of the decision. A system can fail because of:

- **capability** — more reasoning or a stronger model is needed;
- **missing or stale information** — retrieval is needed;
- **underspecification** — the user needs to clarify something;
- **tool dependence** — code, files, calculators, APIs, or other tools are required;
- **verification need** — an answer should be checked before it is trusted.

The research question is therefore:

> **Can an AI controller diagnose the likely source of failure and choose the appropriate intervention while respecting different user trade-offs?**

## Candidate interventions

AURA chooses among a fixed set of base interventions:

1. small-model direct answer;
2. medium-model direct answer;
3. frontier-model direct answer;
4. retrieval-assisted answer;
5. tool-assisted answer;
6. clarify-first answer;
7. verify-then-commit answer.

AURA itself is **not** one of these actions. It is the policy that chooses among them and, in later experiments, may revise its choice after observing an intermediate result.

## Evaluation objective

For each task $x$ and feasible action $a$, we record an outcome vector

$$
o(x,a) = (q, c, \ell, i, r),
$$

where the terms represent normalized quality, cost, latency, interaction burden, and residual-risk penalty. Each component is mapped to a predefined $[0,1]$ scale before the final evaluation.

For a utility profile $\theta$ with non-negative weights $w_\theta$ that sum to one, the provisional utility is

$$
U_\theta(x,a)
=
w_Q q
-
w_C c
-
w_L \ell
-
w_I i
-
w_R r.
$$

The exact normalization functions and profile weights are **not final yet**. They will be frozen before the locked evaluation set is used.

For each task, we can execute the feasible base interventions offline and define an **empirical oracle**:

$$
a_\theta^*(x)
=
\operatorname*{arg\,max}_{a \in \mathcal{A}(x)} U_\theta(x,a).
$$

A routing policy $\pi$ is then evaluated by utility regret:

$$
\operatorname{Regret}_\theta(x;\pi)
=
U_\theta\!\left(x,a_\theta^*(x)\right)
-
U_\theta\!\left(x,\pi(x,\theta)\right).
$$

Lower regret is better.

## Phase 0

Before training a router, AURA will build a **30-task benchmark sanity set** across five failure families:

- 6 capability-limited tasks;
- 6 knowledge/freshness-limited tasks;
- 6 specification-limited tasks;
- 6 tool/execution-limited tasks;
- 6 verification-sensitive tasks.

Each feasible intervention is executed once per task. The recorded outcomes are then re-scored under multiple utility profiles such as reliability-first, speed-first, cost-first, and balanced. This avoids unnecessarily rerunning the same model output simply because the utility weights changed.

The project proceeds only if the benchmark shows meaningful variation in which intervention is empirically best.

## Research questions

1. Does intervention-aware routing outperform model-only routing?
2. Can a controller distinguish capability limitations from missing information, ambiguity, tool dependence, and verification need?
3. Do different utility profiles change which route is optimal for the same task?
4. Can uncertainty in the **routing decision itself** be calibrated?
5. Does limited closed-loop re-routing outperform a one-shot routing decision?

## Repository structure

```text
AURA/
├── README.md
├── RESEARCH_SPEC.md
├── LITERATURE_MAP.md
├── EXPERIMENT_PLAN.md
├── configs/
│   ├── interventions.yaml
│   └── utility_profiles.yaml
├── docs/
│   └── DECISIONS.md
├── src/
│   └── aura/
├── evals/
└── results/
```

## Scientific posture

AURA does **not** claim that model routing, clarification, uncertainty estimation, preference-aware routing, or agent routing are individually novel.

The current research hypothesis is narrower: **joint intervention selection under heterogeneous uncertainty and user-dependent utility may outperform model-only routing, and this can be tested using an empirical-oracle regret framework.**

That claim is provisional and will be re-checked against the literature before any publication-level novelty statement is made.
