import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from src.ai.model import DuelingQNetwork
from src.ai.replay_buffer import ReplayBuffer
from src.engine.gwent_env import GwentEnv

class DQNAgent:
    def __init__(self, state_size: int, action_size: int, device: torch.device | None = None):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Hyperparameters
        self.memory = ReplayBuffer(capacity=200000) # Max transitions to store in memory
        self.gamma = 0.995 # Discount factor for future rewards
        self.epsilon = 1.0 # Exploration rate
        self.epsilon_min = 0.05 # Minimum exploration rate
        self.epsilon_decay = 1.0  # set via configure_epsilon_decay()
        self.batch_size = 256 # Memory batch size for training
        self.learning_rate = 0.000125 # Learning rate

        # 2 Networks for DQN: Main and Target
        self.policy_net = DuelingQNetwork(state_size, action_size).to(self.device)
        self.target_net = DuelingQNetwork(state_size, action_size).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

    
    def select_action(self, state: np.ndarray, legal_actions: np.ndarray) -> int:
        """Epsilon-greedy action selection."""
        return self.select_actions_batch([state], [legal_actions])[0]

    def select_actions_batch(
        self,
        states: list[np.ndarray],
        legal_masks: list[np.ndarray],
    ) -> list[int]:
        """Epsilon-greedy action selection for multiple envs in one forward pass."""
        n = len(states)
        actions = [0] * n
        explore_indices: list[int] = []
        exploit_indices: list[int] = []

        for i in range(n):
            if random.random() <= self.epsilon:
                explore_indices.append(i)
            else:
                exploit_indices.append(i)

        for i in explore_indices:
            legal = np.where(legal_masks[i])[0]
            actions[i] = int(random.choice(legal))

        if not exploit_indices:
            return actions

        states_tensor = torch.as_tensor(
            np.stack([states[i] for i in exploit_indices]),
            dtype=torch.float32,
            device=self.device,
        )
        masks_tensor = torch.as_tensor(
            np.stack([legal_masks[i] for i in exploit_indices]),
            dtype=torch.bool,
            device=self.device,
        )

        self.policy_net.eval()
        with torch.no_grad():
            q_values = self.policy_net(states_tensor, masks_tensor)

        neg_inf = torch.tensor(float("-inf"), device=self.device) # -inf for illegal actions
        q_values = torch.where(masks_tensor, q_values, neg_inf) 
        best_actions = torch.argmax(q_values, dim=1).cpu().numpy() # argmax the q-values for the best action

        for j, i in enumerate(exploit_indices):
            actions[i] = int(best_actions[j])

        return actions

    def select_greedy_actions_batch(
        self,
        states: list[np.ndarray],
        legal_masks: list[np.ndarray],
        policy_net: DuelingQNetwork | None = None,
    ) -> list[int]:
        """Greedy action selection (epsilon=0) using policy_net or the learner net."""
        if not states:
            return []

        net = policy_net or self.policy_net
        states_tensor = torch.as_tensor(
            np.stack(states),
            dtype=torch.float32,
            device=self.device,
        )
        masks_tensor = torch.as_tensor(
            np.stack(legal_masks),
            dtype=torch.bool,
            device=self.device,
        )

        net.eval()
        with torch.no_grad():
            q_values = net(states_tensor, masks_tensor)

        neg_inf = torch.tensor(float("-inf"), device=self.device)
        q_values = torch.where(masks_tensor, q_values, neg_inf)
        best_actions = torch.argmax(q_values, dim=1).cpu().numpy()
        return [int(a) for a in best_actions]

    def configure_epsilon_decay(self, num_episodes: int) -> None:
        """Per-episode decay so epsilon reaches epsilon_min after num_episodes."""
        if num_episodes <= 0:
            return
        self.epsilon_decay = (self.epsilon_min / self.epsilon) ** (1.0 / num_episodes)

    def decay_epsilon(self) -> None:
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train_step(self):
        """Pick a random batch from memory and train the DQN."""
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states_tensor = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_tensor = torch.as_tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards_tensor = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_tensor = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones_tensor = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        current_legal = torch.as_tensor(
            np.stack([GwentEnv.legal_mask_from_state(s) for s in states]),
            dtype=torch.bool,
            device=self.device,
        )
        next_legal = torch.as_tensor(
            np.stack([GwentEnv.legal_mask_from_state(s) for s in next_states]),
            dtype=torch.bool,
            device=self.device,
        )

        # What is the Q-value of the action we took?
        current_q = self.policy_net(states_tensor, current_legal).gather(1, actions_tensor).squeeze(1)

        # What was actually the best action in the next state?
        with torch.no_grad():
            neg_inf = torch.tensor(float("-inf"), device=self.device)

            next_q_policy = self.policy_net(next_states_tensor, next_legal)
            next_q_policy = torch.where(next_legal, next_q_policy, neg_inf)
            best_next_actions = next_q_policy.argmax(dim=1, keepdim=True)

            next_q_target = self.target_net(next_states_tensor, next_legal).gather(1, best_next_actions).squeeze(1)

            # Q(s,a) = r + γ * Qt(s', argmax_a' Qp(s', a'))
            expected_q = rewards_tensor + (1 - dones_tensor) * self.gamma * next_q_target

        # Calculate loss by comparing the predicted Q-value with the expected Q-value
        loss_fn = nn.MSELoss()
        loss = loss_fn(current_q, expected_q)

        # Backpropagate loss
        self.optimizer.zero_grad()
        loss.backward()

        # Track training metrics
        grad_norm_sq = 0.0
        gradients_finite = True
        for parameter in self.policy_net.parameters():
            if parameter.grad is None:
                continue
            gradient = parameter.grad.detach()
            gradients_finite &= bool(torch.isfinite(gradient).all())
            grad_norm_sq += gradient.norm(2).item() ** 2
        self.optimizer.step()
        td_error = expected_q - current_q
        return {
            "loss": loss.item(),
            "q_mean": current_q.mean().item(),
            "q_std": current_q.std(unbiased=False).item(),
            "target_mean": expected_q.mean().item(),
            "target_std": expected_q.std(unbiased=False).item(),
            "td_abs_mean": td_error.abs().mean().item(),
            "td_std": td_error.std(unbiased=False).item(),
            "grad_norm": grad_norm_sq ** 0.5,
            "gradients_finite": float(gradients_finite),
        }

    def update_target_network(self):
        """Update the target network with the policy network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())


    def save(self, path: str):
        """Save the model to a file."""
        torch.save(self.policy_net.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path: str):
        """Load the model from a file."""
        self.policy_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        print(f"Model loaded from {path}")
        