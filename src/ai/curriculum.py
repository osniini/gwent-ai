import copy
import random
from dataclasses import dataclass

from src.ai.agent import DQNAgent
from src.ai.model import DuelingQNetwork

PHASE1_FRAC = 0.05
PHASE2_FRAC = 0.1
OPPO_CHECKPOINT_KEEP = 32


def training_phase(episodes_done: int, num_episodes: int) -> str:
    phase1_end = int(num_episodes * PHASE1_FRAC)
    phase2_end = int(num_episodes * (PHASE1_FRAC + PHASE2_FRAC))
    if episodes_done < phase1_end:
        return "random"
    if episodes_done < phase2_end:
        return "dummy"
    return "frozen"


def frozen_episode_count(num_episodes: int) -> int:
    phase2_end = int(num_episodes * (PHASE1_FRAC + PHASE2_FRAC))
    return max(1, num_episodes - phase2_end)


def begin_frozen_exploration(agent: DQNAgent, num_episodes: int) -> None:
    """Frozen phase uses the epsilon already decaying over full training."""
    _ = agent, num_episodes

class FrozenOpponentPool:
    """Laggy learner snapshots sampled as fixed, per-match opponents."""

    def __init__(self, agent: DQNAgent):
        _ = agent
        self.checkpoints: list[OpponentSnapshot] = []

    def on_enter_frozen_phase(self, agent: DQNAgent, episode: int) -> None:
        self._save_snapshot(agent, episode)

    def maybe_refresh(self, agent: DQNAgent, episode: int) -> None:
        self._save_snapshot(agent, episode)

    def sample_opponent(self) -> DuelingQNetwork:
        """Choose an older snapshot, favouring recent eligible opponents."""
        if not self.checkpoints:
            raise RuntimeError("Cannot sample an opponent from an empty pool.")

        # Never sample the most recently saved policy when an older snapshot
        # exists; it is too close to the actively training learner.
        candidates = self.checkpoints[:-1] or self.checkpoints
        weights = list(range(1, len(candidates) + 1))
        return random.choices(candidates, weights=weights, k=1)[0].net

    def snapshot_episodes_ago(
        self,
        current_episode: int,
        episodes_ago: int,
    ) -> DuelingQNetwork | None:
        """Return the newest snapshot no later than the requested age."""
        target_episode = current_episode - episodes_ago
        eligible = [
            snapshot
            for snapshot in self.checkpoints
            if snapshot.episode <= target_episode
        ]
        return eligible[-1].net if eligible else None

    def _save_snapshot(self, agent: DQNAgent, episode: int) -> None:
        snapshot = copy.deepcopy(agent.policy_net)
        snapshot.eval()
        for parameter in snapshot.parameters():
            parameter.requires_grad_(False)
        self.checkpoints.append(OpponentSnapshot(episode=episode, net=snapshot))
        if len(self.checkpoints) > OPPO_CHECKPOINT_KEEP:
            self.checkpoints.pop(0)


@dataclass(frozen=True)
class OpponentSnapshot:
    episode: int
    net: DuelingQNetwork
