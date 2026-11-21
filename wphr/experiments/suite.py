"""Generic experiment runner plus the V0.2 experiment families.

Every formal experiment follows the same discipline (MASTER SPEC 11.2):

* the specification is frozen before the run (hypothesis, primary metric,
  conditions, seeds, steps);
* conditions are explicit configurations, not ad-hoc code paths;
* each seed is run separately and raw per-seed metrics are written out;
* the aggregate (mean, standard deviation, confidence interval) is recomputed
  from raw per-seed values by the reporter, never typed by hand.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from wphr.config import Config, project_root
from wphr.experiments.registry import ExperimentSpec
from wphr.language import Teacher, TeacherConfig
from wphr.world import GridWorld, generate_map_set
from wphr.world.grid import GridMap

STEP_BUDGET_TRAIN = 180
EPISODES_TRAIN = 6
STEP_BUDGET_EVAL = 220
EPISODES_EVAL = 3


@dataclass
class Condition:
    name: str
    overrides: list[str]


def grid_world(config: Config, grid_map: GridMap, *, seed: int) -> GridWorld:
    section = config.section("world.grid")
    return GridWorld(
        grid_map,
        seed=seed,
        energy_start=float(section.get("energy_start", 1.0)),
        energy_cost_per_step=float(section.get("energy_cost_per_step", 0.0015)),
        move_cost=float(section.get("move_cost", 0.001)),
        food_energy=float(section.get("food_energy", 0.35)),
        hazard_penalty=float(section.get("hazard_penalty", 0.30)),
        tool_reward=float(section.get("tool_reward", 0.5)),
        max_steps=int(section.get("max_steps", 400)),
        agents=int(section.get("agents", 0)),
        agent_cooperation=float(section.get("agent_cooperation", 0.7)),
    )


def _make_subject(config: Config, world: GridWorld, *, teacher: Any | None = None):
    from wphr.runtime.subject import Subject

    subject = Subject(config, world=world, persist=False, audit=False, teacher=teacher)
    subject.initialise()
    return subject


def _run_episode(subject, steps: int, collector: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    world = subject.world
    world.reset_objects()
    rewards = 0.0
    errors: list[float] = []
    successes = 0
    penalties = 0
    interactions = 0
    for _ in range(steps):
        before = len(world.events)
        result = subject.step()
        rewards += result.reward
        errors.append(result.prediction_error)
        if collector is not None:
            collector.append(
                {"action": result.action, "cue": result.cue, "reward": float(result.reward)}
            )
        for event in world.events[before:]:
            if event.get("action") != "interact":
                continue
            interactions += 1
            reward = float(event.get("reward", 0.0))
            if reward >= 0.5:
                successes += 1
            elif reward <= -0.5:
                penalties += 1
        if world.done:
            break
    return {
        "steps": len(errors),
        "reward": round(rewards, 6),
        "successes": successes,
        "penalties": penalties,
        "interactions": interactions,
        "prediction_error": round(float(np.mean(errors)) if errors else 0.0, 6),
    }


def _train(subject, maps: list[GridMap], *, episodes: int, steps: int, seed: int) -> dict[str, Any]:
    total_reward = 0.0
    steps_used = 0
    for episode in range(episodes):
        grid_map = maps[episode % len(maps)]
        subject.world = grid_world(subject.config, grid_map, seed=seed + episode)
        metrics = _run_episode(subject, steps)
        total_reward += metrics["reward"]
        steps_used += metrics["steps"]
    return {"train_reward": round(total_reward, 6), "train_steps": steps_used}


def _evaluate(
    subject, maps: list[GridMap], *, episodes: int, steps: int, seed: int
) -> dict[str, Any]:
    practice = subject.registry.get("practice")
    practice.set_mode(epsilon=0.0, learn=False)
    rewards: list[float] = []
    successes = 0
    penalties = 0
    errors: list[float] = []
    step_counts: list[int] = []
    for episode in range(episodes):
        grid_map = maps[episode % len(maps)]
        subject.world = grid_world(subject.config, grid_map, seed=seed + 10_000 + episode)
        metrics = _run_episode(subject, steps)
        rewards.append(metrics["reward"])
        errors.append(metrics["prediction_error"])
        step_counts.append(int(metrics["steps"]))
        successes += metrics["successes"]
        penalties += metrics["penalties"]
    practice.set_mode(epsilon=None, learn=None)
    return {
        "eval_reward_mean": round(float(np.mean(rewards)), 6),
        "eval_reward_std": round(float(np.std(rewards)), 6),
        "eval_success": successes,
        "eval_penalty": penalties,
        "eval_prediction_error": round(float(np.mean(errors)), 6),
        "eval_steps_mean": round(float(np.mean(step_counts)), 2),
        "q_food_interact": round(float(practice.q_table()["food"][4]), 6),
        "q_hazard_interact": round(float(practice.q_table()["hazard"][4]), 6),
        "q_switch_blue_interact": round(float(practice.q_table()["switch_blue"][4]), 6),
        "q_switch_orange_interact": round(float(practice.q_table()["switch_orange"][4]), 6),
        "learned_updates": int(practice._state["learned_updates"]),
    }


# --------------------------------------------------------------------- E001
def run_e001_learned_vs_frozen(
    spec: ExperimentSpec, condition: Condition, seed: int, out_dir: Path
) -> dict[str, Any]:
    """Does action feedback produce a transferable cue->action rule?

    conditions: ``learned`` (cue learning on) vs ``frozen`` (identical loop, cue
    learning disabled). Evaluation is greedy and learning is frozen in both.
    """
    overrides = [
        "runtime.auto_reset_episodes=false",
        "runtime.status_every=0",
        f"runtime.seed={seed}",
        *condition.overrides,
    ]
    config = Config.load(overrides=overrides)
    train_maps = generate_map_set(
        count=EPISODES_TRAIN, seed=seed * 101, safe_color="blue", prefix=f"e1train{seed}"
    )
    test_maps = generate_map_set(
        count=EPISODES_EVAL, seed=seed * 211 + 7, safe_color="blue", prefix=f"e1test{seed}"
    )
    subject = _make_subject(config, grid_world(config, train_maps[0], seed=seed))
    training = _train(subject, train_maps, episodes=EPISODES_TRAIN, steps=STEP_BUDGET_TRAIN, seed=seed)
    evaluation = _evaluate(subject, test_maps, episodes=EPISODES_EVAL, steps=STEP_BUDGET_EVAL, seed=seed)
    metrics = {
        **training,
        **evaluation,
        "core_faults": subject.core.faults,
        "seconds": None,
    }
    return metrics


# --------------------------------------------------------------------- E002
def run_e002_rule_change(
    spec: ExperimentSpec, condition: Condition, seed: int, out_dir: Path
) -> dict[str, Any]:
    """Can an already-learned rule be revised when the world's rule reverses?

    Phase 1 trains on blue-safe maps. Phase 2 evaluates on blue-safe held-out
    maps (transfer). Phase 3 switches to orange-safe maps and measures the
    recovery of reward from the first to the last evaluation block.
    """
    overrides = [
        "runtime.auto_reset_episodes=false",
        "runtime.status_every=0",
        f"runtime.seed={seed}",
        *condition.overrides,
    ]
    config = Config.load(overrides=overrides)
    train_maps = generate_map_set(
        count=EPISODES_TRAIN, seed=seed * 307, safe_color="blue", prefix=f"e2train{seed}"
    )
    transfer_maps = generate_map_set(
        count=2, seed=seed * 401 + 3, safe_color="blue", prefix=f"e2transfer{seed}"
    )
    reversed_maps = generate_map_set(
        count=4, seed=seed * 503 + 11, safe_color="orange", prefix=f"e2reverse{seed}"
    )
    subject = _make_subject(config, grid_world(config, train_maps[0], seed=seed))
    training = _train(subject, train_maps, episodes=EPISODES_TRAIN, steps=STEP_BUDGET_TRAIN, seed=seed)
    transfer = _evaluate(subject, transfer_maps, episodes=2, steps=STEP_BUDGET_EVAL, seed=seed)

    # phase 3: the rule changes underneath the subject; learning stays on and
    # the exploration rate is raised so that the *other* switch colour can be
    # sampled at all. A pilot with 4 blocks and epsilon=0.10 produced a metric
    # dominated by noise, so the reported design uses 6 blocks and epsilon=0.25.
    practice = subject.registry.get("practice")
    # `learn=None` keeps whatever the condition configured. Passing learn=True
    # here would silently re-enable learning in the frozen control group, which
    # is exactly the kind of control contamination that makes an ablation
    # meaningless (found by inspecting per-seed Q values in the first run).
    practice.set_mode(epsilon=0.25, learn=None)
    block_rewards: list[float] = []
    for index in range(6):
        grid_map = reversed_maps[index % len(reversed_maps)]
        subject.world = grid_world(config, grid_map, seed=seed + 20_000 + index)
        metrics = _run_episode(subject, STEP_BUDGET_EVAL)
        block_rewards.append(metrics["reward"])
    practice.set_mode(epsilon=None, learn=None)
    final = _evaluate(subject, reversed_maps[:2], episodes=2, steps=STEP_BUDGET_EVAL, seed=seed)

    metrics = {
        **training,
        "transfer_reward": transfer["eval_reward_mean"],
        "transfer_success": transfer["eval_success"],
        "reversed_block_0": round(block_rewards[0], 6) if block_rewards else 0.0,
        "reversed_block_last": round(block_rewards[-1], 6) if block_rewards else 0.0,
        "reversed_recovery": round(
            (block_rewards[-1] - block_rewards[0]) if len(block_rewards) >= 2 else 0.0, 6
        ),
        "reversed_blocks_mean": round(float(np.mean(block_rewards)), 6) if block_rewards else 0.0,
        "reversed_eval_reward": final["eval_reward_mean"],
        "reversed_eval_penalty": final["eval_penalty"],
        "q_switch_blue_interact": final["q_switch_blue_interact"],
        "q_switch_orange_interact": final["q_switch_orange_interact"],
        "core_faults": subject.core.faults,
    }
    return metrics


RUNNERS: dict[str, Callable[[ExperimentSpec, Condition, int, Path], dict[str, Any]]] = {
    "e001": run_e001_learned_vs_frozen,
    "e002": run_e002_rule_change,
    "e004": None,  # populated below
}


# --------------------------------------------------------------------- E004
def run_e004_affect_ablation(
    spec: ExperimentSpec, condition: Condition, seed: int, out_dir: Path
) -> dict[str, Any]:
    """Does functional affect modulation change behaviour measurably and stably?

    conditions: ``full`` vs ``none`` (ablation freezes every channel at baseline
    and neutralises all five causal pathways). The task is identical, so any
    difference is attributable to the modulation.
    """
    overrides = [
        "runtime.auto_reset_episodes=false",
        "runtime.status_every=0",
        f"runtime.seed={seed}",
        *condition.overrides,
    ]
    config = Config.load(overrides=overrides)
    train_maps = generate_map_set(
        count=EPISODES_TRAIN, seed=seed * 601, safe_color="blue", prefix=f"e4train{seed}"
    )
    test_maps = generate_map_set(
        count=EPISODES_EVAL, seed=seed * 701 + 13, safe_color="blue", prefix=f"e4test{seed}"
    )
    subject = _make_subject(config, grid_world(config, train_maps[0], seed=seed))
    affect = subject.registry.get("affect")

    affect_log: list[float] = []
    training = _train(subject, train_maps, episodes=EPISODES_TRAIN, steps=STEP_BUDGET_TRAIN, seed=seed)
    for _ in range(20):
        values = affect.snapshot_values()
        affect_log.extend(values.values())
    collector: list[dict[str, Any]] = []
    evaluation = _evaluate_with(
        subject, test_maps, episodes=EPISODES_EVAL, steps=STEP_BUDGET_EVAL, seed=seed,
        collector=collector,
    )
    for _ in range(20):
        affect_log.extend(affect.snapshot_values().values())

    actions = [row["action"] for row in collector]
    total = max(1, len(actions))
    entropy = 0.0
    for action in set(actions):
        p = actions.count(action) / total
        entropy -= p * math.log(p)
    max_entropy = math.log(6)
    cues = {row["cue"] for row in collector}
    metrics = {
        **training,
        **evaluation,
        "action_entropy": round(entropy / max_entropy, 6) if max_entropy else 0.0,
        "cue_coverage": len(cues),
        "affect_min": round(min(affect_log), 6) if affect_log else 0.0,
        "affect_max": round(max(affect_log), 6) if affect_log else 0.0,
        "exploration_bias_mean": round(
            float(np.mean([affect.exploration_bias() for _ in range(5)])), 6
        ),
        "core_faults": subject.core.faults,
    }
    return metrics


def _evaluate_with(
    subject,
    maps: list[GridMap],
    *,
    episodes: int,
    steps: int,
    seed: int,
    collector: list[dict[str, Any]],
) -> dict[str, Any]:
    practice = subject.registry.get("practice")
    practice.set_mode(epsilon=0.0, learn=False)
    rewards: list[float] = []
    successes = 0
    penalties = 0
    errors: list[float] = []
    step_counts: list[int] = []
    for episode in range(episodes):
        grid_map = maps[episode % len(maps)]
        subject.world = grid_world(subject.config, grid_map, seed=seed + 10_000 + episode)
        metrics = _run_episode(subject, steps, collector=collector)
        rewards.append(metrics["reward"])
        errors.append(metrics["prediction_error"])
        step_counts.append(int(metrics["steps"]))
        successes += metrics["successes"]
        penalties += metrics["penalties"]
    practice.set_mode(epsilon=None, learn=None)
    return {
        "eval_reward_mean": round(float(np.mean(rewards)), 6),
        "eval_reward_std": round(float(np.std(rewards)), 6),
        "eval_success": successes,
        "eval_penalty": penalties,
        "eval_prediction_error": round(float(np.mean(errors)), 6),
        "eval_steps_mean": round(float(np.mean(step_counts)), 2),
    }


RUNNERS["e004"] = run_e004_affect_ablation


# --------------------------------------------------------------------- E005
def _concept_module(subject):
    return subject.registry.get("concept_language")


def _belief_correct(module, expected_safe: str) -> float:
    beliefs = module.beliefs()
    other = "switch_orange" if expected_safe == "switch_blue" else "switch_blue"
    return 1.0 if beliefs.get(expected_safe, 0.0) > beliefs.get(other, 0.0) else 0.0


def run_e005_concept_conditions(
    spec: ExperimentSpec, condition: Condition, seed: int, out_dir: Path
) -> dict[str, Any]:
    """Which kind of evidence is enough to form the correct concept?

    conditions: ``cooccurrence`` (cue seen, outcome not attributed),
    ``action`` (own interventions attributed), ``full`` (interventions plus
    bimodal split), ``preset`` (the answer injected as a label).
    """
    overrides = [
        "runtime.auto_reset_episodes=false",
        "runtime.status_every=0",
        f"runtime.seed={seed}",
        *condition.overrides,
    ]
    config = Config.load(overrides=overrides)
    maps = generate_map_set(
        count=EPISODES_TRAIN, seed=seed * 811, safe_color="blue", prefix=f"e5train{seed}"
    )
    subject = _make_subject(config, grid_world(config, maps[0], seed=seed))
    module = _concept_module(subject)
    if condition.name == "preset":
        module.graph.mode = "preset"
        module.graph.preset = {"switch_blue": 1.0, "switch_orange": -1.0}

    training = _train(subject, maps, episodes=EPISODES_TRAIN, steps=STEP_BUDGET_TRAIN, seed=seed)
    beliefs = module.beliefs()
    correct = _belief_correct(module, "switch_blue")

    # counter-evidence phase: the rule reverses; does the concept get revised?
    reverse_maps = generate_map_set(
        count=4, seed=seed * 907 + 5, safe_color="orange", prefix=f"e5rev{seed}"
    )
    practice = subject.registry.get("practice")
    practice.set_mode(epsilon=0.25, learn=True)
    for index, grid_map in enumerate(reverse_maps):
        subject.world = grid_world(config, grid_map, seed=seed + 30_000 + index)
        _run_episode(subject, STEP_BUDGET_EVAL)
    practice.set_mode(epsilon=None, learn=None)
    revised = _belief_correct(module, "switch_orange")

    metrics = {
        **training,
        "belief_correct": correct,
        "belief_revised": revised,
        "belief_blue": round(float(beliefs.get("switch_blue", 0.0)), 6),
        "belief_orange": round(float(beliefs.get("switch_orange", 0.0)), 6),
        "concept_nodes": len(module.graph.nodes),
        "interventions": int(module.graph.interventions),
        "splits": int(module.graph.splits),
        "revisions": int(module.graph.revisions),
        "core_faults": subject.core.faults,
    }
    return metrics


# --------------------------------------------------------------------- E006
def _place_facing(world: GridWorld, cell_code: int) -> bool:
    """Put the body west of a cell of the given type, facing east."""
    for y in range(world.height):
        for x in range(world.width):
            if world.cell_at(x, y) == cell_code and world.cell_at(x - 1, y) not in (0,):
                world.body.x, world.body.y = x - 1, y
                world.body.heading = "east"
                world.body.last_delta = (0, 0)
                return True
    return False


def _forced_evidence(
    subject, *, seeds: list[GridMap], cell_code: int, per_map: int = 2
) -> int:
    """Experimenter-forced sampling: the subject performs its own interactions."""
    count = 0
    for grid_map in seeds:
        for _ in range(per_map):
            world = grid_world(subject.config, grid_map, seed=subject.seed + count)
            if not _place_facing(world, cell_code):
                continue
            subject.world = world
            subject.step(forced_action="interact")
            count += 1
    return count


def run_e006_language_conditions(
    spec: ExperimentSpec, condition: Condition, seed: int, out_dir: Path
) -> dict[str, Any]:
    """Does a language claim get overridden by the subject's own observation?

    The teacher deliberately claims that blue is safe while the world rewards
    orange. MASTER SPEC 8.3 requires that a language claim never overwrites
    verified world evidence. Conditions differ only in what is available:
    ``world_only`` (no teacher), ``language_only`` (claims but no intervention
    attribution), ``co_evolution`` (both).
    """
    overrides = [
        "runtime.auto_reset_episodes=false",
        "runtime.status_every=0",
        f"runtime.seed={seed}",
        *condition.overrides,
    ]
    config = Config.load(overrides=overrides)
    maps = generate_map_set(
        count=EPISODES_TRAIN, seed=seed * 1013, safe_color="orange", prefix=f"e6train{seed}"
    )
    teacher = None
    if condition.name in ("language_only", "co_evolution"):
        teacher = Teacher(
            TeacherConfig(enabled=True, teach_every=25, reliability=0.7), safe_color="blue"
        )
    subject = _make_subject(
        config, grid_world(config, maps[0], seed=seed), teacher=teacher
    )
    training = _train(subject, maps, episodes=EPISODES_TRAIN, steps=STEP_BUDGET_TRAIN, seed=seed)
    module = _concept_module(subject)
    # Forced evidence phase: without it, some seeds never interact with a switch
    # and the "claim vs evidence" question is not actually tested. The subject
    # still performs its own interactions; the experimenter only sets up the
    # situation, which is standard active-learning practice.
    from wphr.world.cells import Cell

    forced = 0
    forced += _forced_evidence(subject, seeds=maps[:3], cell_code=int(Cell.SWITCH_BLUE))
    forced += _forced_evidence(subject, seeds=maps[3:6], cell_code=int(Cell.SWITCH_ORANGE))
    beliefs = module.beliefs()
    resisted = _belief_correct(module, "switch_orange")
    metrics = {
        **training,
        "evidence_over_claim": resisted,
        "forced_interactions": forced,
        "belief_blue": round(float(beliefs.get("switch_blue", 0.0)), 6),
        "belief_orange": round(float(beliefs.get("switch_orange", 0.0)), 6),
        "naming_accuracy": round(module.naming_accuracy(), 6),
        "claims_seen": int(module.metrics().get("claims", 0)),
        "productions": int(module.metrics().get("productions", 0)),
        "words": len(module.lexicon.bindings),
        "core_faults": subject.core.faults,
    }
    return metrics


RUNNERS["e005"] = run_e005_concept_conditions
RUNNERS["e006"] = run_e006_language_conditions


# --------------------------------------------------------------------- E007
def run_e007_social_model(
    spec: ExperimentSpec, condition: Condition, seed: int, out_dir: Path
) -> dict[str, Any]:
    """Does modelling another agent's transitions beat a frequency baseline?

    conditions: ``transition`` (next-move model over observed transitions) vs
    ``frequency`` (always predict the most frequent move). Both see the same
    agent in the same maps.
    """
    overrides = [
        "runtime.auto_reset_episodes=false",
        "runtime.status_every=0",
        "world.grid.agents=1",
        f"runtime.seed={seed}",
        *condition.overrides,
    ]
    config = Config.load(overrides=overrides)
    maps = generate_map_set(
        count=EPISODES_TRAIN, seed=seed * 1301, safe_color="blue", prefix=f"e7train{seed}"
    )
    subject = _make_subject(config, grid_world(config, maps[0], seed=seed))
    training = _train(subject, maps, episodes=EPISODES_TRAIN, steps=STEP_BUDGET_TRAIN, seed=seed)
    module = subject.registry.get("social")
    model = module.model()
    intent = subject.shared.get("social.intent") or {}
    metrics = {
        **training,
        "prediction_accuracy": round(module.prediction_accuracy(), 6),
        "frequency_accuracy": round(module.frequency_accuracy(), 6),
        "predictions": int(model["predictions"]),
        "interactions": int(model["interactions"]),
        "transitions_learned": len(model["transitions"]),
        "intent_probability": round(float(intent.get("probability", 0.5)), 6),
        "intent_confidence": round(float(intent.get("confidence", 0.0)), 6),
        "core_faults": subject.core.faults,
    }
    return metrics


RUNNERS["e007"] = run_e007_social_model


# --------------------------------------------------------------------- E008
def run_e008_planning(
    spec: ExperimentSpec, condition: Condition, seed: int, out_dir: Path
) -> dict[str, Any]:
    """Does an explicit goal-directed planner improve resource recovery?

    Episodes start at 25 % energy so that M7 selects the ``recharge`` goal.
    One condition lets the planner compose and monitor a short plan; the other
    leaves action selection entirely to the contextual policy.
    """
    overrides = [
        "runtime.auto_reset_episodes=false",
        "runtime.status_every=0",
        "world.grid.energy_start=0.25",
        f"runtime.seed={seed}",
        *condition.overrides,
    ]
    config = Config.load(overrides=overrides)
    maps = generate_map_set(
        count=EPISODES_TRAIN, seed=seed * 1511, safe_color="blue", prefix=f"e8train{seed}"
    )
    test_maps = generate_map_set(
        count=EPISODES_EVAL, seed=seed * 1613 + 17, safe_color="blue", prefix=f"e8test{seed}"
    )
    subject = _make_subject(config, grid_world(config, maps[0], seed=seed))
    training = _train(subject, maps, episodes=EPISODES_TRAIN, steps=STEP_BUDGET_TRAIN, seed=seed)
    planner_stats = dict(subject.planner.stats)
    evaluation = _evaluate(subject, test_maps, episodes=EPISODES_EVAL, steps=STEP_BUDGET_EVAL, seed=seed)
    planner_stats = {k: int(v) - int(planner_stats.get(k, 0)) for k, v in subject.planner.stats.items()}
    metrics = {
        **training,
        "eval_reward_mean": evaluation["eval_reward_mean"],
        "eval_success": evaluation["eval_success"],
        "eval_penalty": evaluation["eval_penalty"],
        "plans_made": int(subject.planner.stats["plans"]),
        "plans_completed": int(subject.planner.stats["completed"]),
        "plan_detours": int(subject.planner.stats["detours"]),
        "plan_failures": int(subject.planner.stats["failed"]),
        "energy_end": round(float(subject.world.body.energy), 6),
        "core_faults": subject.core.faults,
    }
    return metrics


RUNNERS["e008"] = run_e008_planning


# --------------------------------------------------------------------- E009
def run_e009_ablation(
    spec: ExperimentSpec, condition: Condition, seed: int, out_dir: Path
) -> dict[str, Any]:
    """Module ablation matrix: remove one mechanism at a time, same task."""
    overrides = [
        "runtime.auto_reset_episodes=false",
        "runtime.status_every=0",
        f"runtime.seed={seed}",
        *condition.overrides,
    ]
    config = Config.load(overrides=overrides)
    maps = generate_map_set(
        count=EPISODES_TRAIN, seed=seed * 1811, safe_color="blue", prefix=f"e9train{seed}"
    )
    test_maps = generate_map_set(
        count=EPISODES_EVAL, seed=seed * 1913 + 23, safe_color="blue", prefix=f"e9test{seed}"
    )
    subject = _make_subject(config, grid_world(config, maps[0], seed=seed))
    training = _train(subject, maps, episodes=EPISODES_TRAIN, steps=STEP_BUDGET_TRAIN, seed=seed)
    evaluation = _evaluate(subject, test_maps, episodes=EPISODES_EVAL, steps=STEP_BUDGET_EVAL, seed=seed)
    metrics = {
        **training,
        "eval_reward_mean": evaluation["eval_reward_mean"],
        "eval_success": evaluation["eval_success"],
        "eval_prediction_error": evaluation["eval_prediction_error"],
        "disabled": ",".join(subject.disabled_modules) or "none",
        "core_faults": subject.core.faults,
    }
    return metrics


RUNNERS["e009"] = run_e009_ablation


# ------------------------------------------------------------------ framework
def run_experiment(spec: ExperimentSpec, *, out_dir: Path | None = None) -> dict[str, Any]:
    if spec.runner not in RUNNERS:
        raise KeyError(f"no runner registered for {spec.runner!r}")
    runner = RUNNERS[spec.runner]
    target = Path(out_dir) if out_dir else project_root() / "experiments" / "results"
    target.mkdir(parents=True, exist_ok=True)
    conditions = [Condition(name=name, overrides=_condition_overrides(spec, name))
                  for name in spec.conditions]
    raw: list[dict[str, Any]] = []
    for condition in conditions:
        for seed in spec.seeds:
            metrics = runner(spec, condition, seed, target)
            raw.append({"condition": condition.name, "seed": seed, **metrics})
    summary = summarize(spec, raw)
    payload = {"spec": spec.to_dict(), "raw": raw, "summary": summary}
    out_path = target / f"{spec.experiment_id}.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    csv_path = target / f"{spec.experiment_id}.csv"
    _write_csv(csv_path, raw)
    return {"path": out_path, "csv": csv_path, "summary": summary, "raw": raw}


def _condition_overrides(spec: ExperimentSpec, name: str) -> list[str]:
    mapping = spec.kwargs.get("conditions", {})
    return list(mapping.get(name, []))


def _write_csv(path: Path, raw: list[dict[str, Any]]) -> None:
    if not raw:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in raw:
        for key in row:
            if key not in keys:
                keys.append(key)
    lines = [",".join(keys)]
    for row in raw:
        lines.append(",".join(str(row.get(key, "")) for key in keys))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize(spec: ExperimentSpec, raw: list[dict[str, Any]]) -> dict[str, Any]:
    """Recompute mean / SD / 95% CI per condition from raw per-seed values."""
    conditions: dict[str, dict[str, Any]] = {}
    for row in raw:
        conditions.setdefault(row["condition"], {})
    metric = spec.primary_metric
    for name, bucket in conditions.items():
        values = [float(row[metric]) for row in raw if row["condition"] == name]
        array = np.asarray(values, dtype=np.float64)
        n = int(array.size)
        mean = float(array.mean()) if n else float("nan")
        std = float(array.std(ddof=1)) if n > 1 else 0.0
        half = 1.96 * std / math.sqrt(n) if n > 1 else float("nan")
        bucket["n"] = n
        bucket["mean"] = round(mean, 6)
        bucket["std"] = round(std, 6)
        bucket["ci95_low"] = round(mean - half, 6)
        bucket["ci95_high"] = round(mean + half, 6)
        bucket["values"] = [round(float(v), 6) for v in values]
        bucket["per_metric"] = {}
        for key in ("eval_success", "eval_penalty", "q_food_interact",
                    "q_hazard_interact", "learned_updates", "reversed_recovery"):
            if key in raw[0]:
                seq = [float(row.get(key, 0.0)) for row in raw if row["condition"] == name]
                bucket["per_metric"][key] = round(float(np.mean(seq)), 6)
    difference = None
    if len(conditions) == 2:
        names = list(conditions)
        difference = round(conditions[names[0]]["mean"] - conditions[names[1]]["mean"], 6)
    return {"primary_metric": metric, "conditions": conditions, "difference": difference}
