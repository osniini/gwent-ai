"""Greedy, no-learning policy evaluations."""

import numpy as np

from src.ai.agent import DQNAgent
from src.ai.model import DuelingQNetwork
from src.ai.opponents import LEARNER_PLAYER, dummy_action, random_action
from src.engine.gwent_env import GwentEnv

EVAL_PARALLEL_ENVS = 128


def evaluate_opponent(
    agent: DQNAgent,
    *,
    opponent: str,
    matches: int,
    opponent_net: DuelingQNetwork | None = None,
) -> dict[str, int]:
    """Play balanced, greedy matches without updating the learner."""
    if matches <= 0 or matches % 2:
        raise ValueError("matches must be a positive, even number")
    if opponent == "frozen" and opponent_net is None:
        raise ValueError("A frozen evaluation requires an opponent network")

    trackers = []
    matches_started = 0
    for _ in range(min(EVAL_PARALLEL_ENVS, matches)):
        trackers.append(_new_tracker(matches_started))
        matches_started += 1

    results = {"wins": 0, "losses": 0, "draws": 0}
    matches_done = 0
    while matches_done < matches:
        active = [tracker for tracker in trackers if not tracker["done"]]
        actions = _select_actions(agent, active, opponent, opponent_net)

        for tracker, action in zip(active, actions):
            _, _, done = tracker["env"].step(action)
            if not done:
                continue

            _record_result(results, tracker["env"])
            matches_done += 1
            if matches_started < matches:
                _reset_tracker(tracker, matches_started)
                matches_started += 1
            else:
                tracker["done"] = True

    return results


def _new_tracker(match_index: int) -> dict:
    env = GwentEnv()
    env.reset(starting_player=_starting_player(match_index))
    return {"env": env, "done": False}


def _reset_tracker(tracker: dict, match_index: int) -> None:
    tracker["env"].reset(starting_player=_starting_player(match_index))
    tracker["done"] = False


def _starting_player(match_index: int) -> int:
    return LEARNER_PLAYER if match_index % 2 == 0 else 3 - LEARNER_PLAYER


def _select_actions(
    agent: DQNAgent,
    active: list[dict],
    opponent: str,
    opponent_net: DuelingQNetwork | None,
) -> list[int]:
    learner_states: list[np.ndarray] = []
    learner_masks: list[np.ndarray] = []
    learner_slots: list[int] = []
    opponent_states: list[np.ndarray] = []
    opponent_masks: list[np.ndarray] = []
    opponent_slots: list[int] = []
    actions: list[int | None] = [None] * len(active)

    for slot, tracker in enumerate(active):
        env = tracker["env"]
        legal = env.get_legal_actions()
        if env.current_player == LEARNER_PLAYER:
            learner_slots.append(slot)
            learner_states.append(env.get_state_for_player(LEARNER_PLAYER))
            learner_masks.append(legal)
        elif opponent == "random":
            actions[slot] = random_action(legal)
        elif opponent == "dummy":
            actions[slot] = dummy_action(env, legal)
        else:
            opponent_slots.append(slot)
            opponent_states.append(env.get_state_for_player(env.current_player))
            opponent_masks.append(legal)

    for slot, action in zip(
        learner_slots,
        agent.select_greedy_actions_batch(learner_states, learner_masks),
    ):
        actions[slot] = action

    if opponent_states:
        assert opponent_net is not None
        for slot, action in zip(
            opponent_slots,
            agent.select_greedy_actions_batch(
                opponent_states,
                opponent_masks,
                policy_net=opponent_net,
            ),
        ):
            actions[slot] = action

    return [int(action) for action in actions]


def _record_result(results: dict[str, int], env: GwentEnv) -> None:
    if env.match_draw:
        results["draws"] += 1
    elif env.lives[LEARNER_PLAYER - 1] == 0:
        results["losses"] += 1
    else:
        results["wins"] += 1
