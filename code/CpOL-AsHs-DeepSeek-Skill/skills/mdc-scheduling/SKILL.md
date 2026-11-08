---
name: mdc-scheduling
description: Select a mobile decision center region and evolve six coarse scoring formulas and their weights from MDC simulation feedback through the DeepSeek API.
---

# MDC scheduling and formula evolution

You are invoked by the MDC simulator in either `select` or `evolve` mode.
Return one JSON object, without Markdown. Maximize tasks completed by the end
of the experiment. Region positions are zero-based window top-left coordinates,
not geometric centers. Coverage selects commandable agents; targets can be outside.
There is no MDC speed constraint.
Keep the reason brief (at most 200 characters). Return only the required fields;
do not repeat input candidates, simulation histories, or prior proposals in the output.

## Select

Input contains the current policy, current-step candidate positions, computed
components and scores, and feedback from earlier steps in this episode.
Output `{"position": [row, lin], "reason": "brief explanation"}`.
Choose the FIRST highest-ranked supplied candidate. Arithmetic ranking is computed
by the host to avoid numerical mistakes; use the supplied order for ties.
Do not change formulas during an episode. Explain which components drive the choice.
If scores are poor, include that observation in the reason for end-of-episode learning.

## Evolve

Input contains the current policy, episode results, prior proposals and replay outcomes.
Propose a complete replacement policy by changing component formulas and/or weights.
Return `{"policy": {"formulas": {...}, "weights": {...}}, "reason": "hypothesis"}`.
Both maps must contain exactly task, pair, charge, imbalance, move, energy_risk.
Weights are nonnegative, at most 1, and sum to 1. Task, pair and charge contribute
positively; imbalance, move and energy_risk are penalties.
The host evaluates old and new policies on identical full-episode data and adopts
only a strict task-count improvement. Failed proposals remain feedback for next rounds.
Do not claim an improvement before receiving replay evidence. No claim of test-set
generalization follows from improving on the adaptation dataset.

## Formula language and raw measurements

Each formula is an expression (at most 400 characters, 100 AST nodes) over the
following raw variables. Variables always refer to raw measurements, not to another
new component, so cross-component references never form recursive definitions.

- task: number of DISTINCT remaining tasks with a time/battery-feasible WUT triple.
- pair: greedy number of triples with disjoint worker, UAV and task IDs.
- charge: sum of low-battery urgency over individually feasible charge rescues;
  urgency=max(0,(0.3-power/full_power)/0.3). This is a potential, not a joint car schedule.
- imbalance: abs(useful_workers-useful_uavs)/(useful_workers+useful_uavs+1e-8).
- move: distance from previous region divided by maximum window displacement; 0 initially.
- energy_risk: mean unrescued UAV risk using 20% reserve after reaching nearest charger.
- workers, uavs, cars: currently commandable counts; remaining: number of uncompleted tasks.
- time_left: global deadline minus the baseline decision-time origin, in simulation units.
- triples: count of feasible WUT triples (contains repeated resources).
- mean_duration, mean_energy, mean_slack: mean completion duration, required-energy/full-power,
  and time-window slack of feasible triples; 0 if no triples.

Allowed: finite numeric literals, variables, + - * /, unary +/- and
min/max (1–4 args), abs, sqrt, log1p, tanh. No Python statements, imports,
attributes, indexing, comprehensions or exponentiation. Protect division with
max(x,1e-8). Results must be nonnegative and finite, intermediates within 1e9.
Host normalizes first three components by their current candidate maximum +1e-8;
penalties are clipped at 1. Initial formulas are identity (task="task", etc.).
For example task="task / (1 + mean_duration / max(time_left, 1))" changes task
potential to prefer tasks that can finish quickly; pair="pair + 0.1*log1p(triples)"
adds choice diversity. These are hypotheses to evaluate, not proven improvements.

Time/battery feasibility, conflict rules, cached graph construction, MWIS and
environment transitions are fixed. Formula evolution changes the six scores, not
the physical feasibility constraints. Do not request future unseen trajectories.
