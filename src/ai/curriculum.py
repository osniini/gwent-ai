import copy
import random

import torch

from src.ai.agent import DQNAgent
from src.ai.model import DuelingQNetwork

PHASE1_FRAC = 0.0
PHASE2_FRAC = 0.0
OPPO_CHECKPOINT_KEEP = 8


def training_phase(episodes_done: int, num_episodes: int) -> str:
    phase1_end = int(num_episodes * PHASE1_FRAC)
    phase2_end = int(num_episodes * (PHASE1_FRAC + PHASE2_FRAC))
    if episodes_done < phase1_end:
        return "random"
    if episodes_done < phase2_end:
        return "greedy"
    return "frozen"


def frozen_episode_count(num_episodes: int) -> int:
    phase2_end = int(num_episodes * (PHASE1_FRAC + PHASE2_FRAC))
    return max(1, num_episodes - phase2_end)


def begin_frozen_exploration(agent: DQNAgent, num_episodes: int) -> None:
    """Frozen phase uses the epsilon already decaying over full training."""
    _ = agent, num_episodes

class FrozenOpponentPool:
    """Laggy snapshots of the learner — opponent plays an older version of itself."""

    def __init__(self, agent: DQNAgent):
        self.device = agent.device
        self.net = DuelingQNetwork(agent.state_size, agent.action_size).to(self.device)
        self.net.eval()
        self.checkpoints: list[dict] = []

    def on_enter_frozen_phase(self, agent: DQNAgent) -> None:
        self._save_and_pick_opponent(agent)

    def maybe_refresh(self, agent: DQNAgent) -> None:
        self._save_and_pick_opponent(agent)

    def _save_and_pick_opponent(self, agent: DQNAgent) -> None:
        self.checkpoints.append(copy.deepcopy(agent.policy_net.state_dict()))
        if len(self.checkpoints) > OPPO_CHECKPOINT_KEEP:
            self.checkpoints.pop(0)

        if len(self.checkpoints) >= 2:
            state_dict = random.choice(self.checkpoints[:-1])
        else:
            state_dict = self.checkpoints[-1]

        self.net.load_state_dict(state_dict)
        self.net.eval()
