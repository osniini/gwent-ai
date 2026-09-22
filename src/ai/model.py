import torch
import torch.nn as nn
from src.engine.gwent_env import (
    BOARD_FEATURES,
    BOARD_COMPOSITION_FEATURES,
    HERO_POWER_FEATURES,
    REDRAWS_REMAINING_STATE_INDEX,
    SCORE_DIFF_STATE_INDEX,
)


def _state_input_scales(state_size: int) -> torch.Tensor:
    """Return fixed per-feature multipliers for raw environment states."""
    if state_size != SCORE_DIFF_STATE_INDEX + 1:
        raise ValueError(f"Unexpected state size: {state_size}")

    scales = torch.ones(state_size, dtype=torch.float32)

    # Hero powers come first, followed by (unit count, base-power total) pairs.
    scales[:HERO_POWER_FEATURES] /= 50.0
    composition_start = HERO_POWER_FEATURES
    composition_end = composition_start + BOARD_COMPOSITION_FEATURES
    scales[composition_start:composition_end:2] /= 10.0
    scales[composition_start + 1:composition_end:2] /= 50.0

    # Global numeric features after board features. Per-card counts stay raw.
    scales[BOARD_FEATURES:BOARD_FEATURES + 2] /= 2.0  # Lives remaining.
    scales[BOARD_FEATURES + 2] /= 10.0  # Opponent hand count.
    scales[REDRAWS_REMAINING_STATE_INDEX] /= 2.0
    scales[SCORE_DIFF_STATE_INDEX] /= 100.0
    return scales

class DuelingQNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int):
        super(DuelingQNetwork, self).__init__()
        # Non-persistent so existing model checkpoints remain loadable; the
        # deterministic scales are rebuilt from the state layout on creation.
        self.register_buffer(
            "state_input_scales",
            _state_input_scales(state_size),
            persistent=False,
        )

        self.feature_network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
        )

    def forward(
        self,
        state: torch.Tensor,
        legal_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features = self.feature_network(state * self.state_input_scales)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)

        if legal_mask is None:
            mean_advantages = advantages.mean(dim=1, keepdim=True)
        else:
            mask = legal_mask.to(dtype=advantages.dtype)
            legal_count = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            mean_advantages = (advantages * mask).sum(dim=1, keepdim=True) / legal_count

        # Q(s,a) = V(s) + (A(s,a) - mean(A(s, a) over legal actions))
        q_values = values + (advantages - mean_advantages)
        return q_values