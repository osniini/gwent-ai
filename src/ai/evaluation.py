"""Greedy, no-learning policy evaluations."""

import copy
from pathlib import Path

import numpy as np

from src.ai.agent import DQNAgent
from src.ai.model import DuelingQNetwork
from src.ai.opponents import LEARNER_PLAYER, dummy_action, random_action
from src.engine.gwent_env import GwentEnv

EVAL_PARALLEL_ENVS = 128
RANDOM_EVALUATION_MATCHES = 300
DUMMY_EVALUATION_MATCHES = 300
FROZEN_EVALUATION_MATCHES = 300
FROZEN_EVALUATION_LAG = 5000
ANCHOR_EVALUATION_MATCHES = 2000


class AnchorPool:
    """Permanent evaluation snapshots for measuring long-term progress."""

    def __init__(self, directory: str | Path = "models"):
        self.directory = Path(directory)
        self.anchors: dict[int, DuelingQNetwork] = {}

    @staticmethod
    def is_anchor_episode(episode: int) -> bool:
        """Save anchors at 25k, 50k, then every 100k episodes."""
        return episode in (25000, 50000) or (
            episode >= 100000 and episode % 100000 == 0
        )

    @staticmethod
    def anchor_episodes_up_to(num_episodes: int) -> tuple[int, ...]:
        """Return all anchor milestones reached during a training run."""
        early_anchors = tuple(
            episode for episode in (25000, 50000) if episode <= num_episodes
        )
        return early_anchors + tuple(range(100000, num_episodes + 1, 100000))

    @staticmethod
    def is_evaluation_episode(episode: int) -> bool:
        """Evaluate existing anchors at 50k, then every 100k episodes."""
        return episode == 50000 or (episode >= 100000 and episode % 100000 == 0)

    def save(self, agent: DQNAgent, episode: int) -> None:
        """Save and retain a policy snapshot for all later anchor evaluations."""
        if not self.is_anchor_episode(episode):
            return

        path = self.directory / f"gwent_agent_{episode // 1000}k.pth"
        self.directory.mkdir(parents=True, exist_ok=True)
        agent.save(str(path))

        anchor = copy.deepcopy(agent.policy_net)
        anchor.eval()
        for parameter in anchor.parameters():
            parameter.requires_grad_(False)
        self.anchors[episode] = anchor

    def evaluate(self, agent: DQNAgent) -> dict[str, dict[str, int]]:
        """Run fixed-size greedy evaluations against prior anchor policies."""
        return {
            f"anchor_{episode // 1000}k": evaluate_opponent(
                agent,
                opponent="frozen",
                matches=ANCHOR_EVALUATION_MATCHES,
                opponent_net=opponent_net,
            )
            for episode, opponent_net in self.anchors.items()
        }


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
