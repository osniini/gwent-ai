import torch
import torch.nn as nn

class DuelingQNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int):
        super(DuelingQNetwork, self).__init__()

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
        features = self.feature_network(state)
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