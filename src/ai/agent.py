import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from src.ai.model import DuelingQNetwork
from src.ai.replay_buffer import ReplayBuffer

class DQNAgent:
    def __init__(self, state_size: int, action_size: int, device: torch.device | None = None):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Hyperparameters
        self.memory = ReplayBuffer(capacity=50000)
        self.gamma = 0.99 # Discount factor for future rewards
        self.epsilon = 1.0 # Exploration rate
        self.epsilon_min = 0.05 # Minimum exploration rate
        self.epsilon_decay = 0.9995 # Exploration rate decay
        self.batch_size = 64 # Memory batch size for training
        self.learning_rate = 0.0005 # Learning rate

        # 2 Networks for DQN: Main and Target
        self.policy_net = DuelingQNetwork(state_size, action_size).to(self.device)
        self.target_net = DuelingQNetwork(state_size, action_size).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

    
    def select_action(self, state: np.ndarray, legal_actions: np.ndarray) -> int:
        """Epsilon-greedy action selection."""
        # Exploration
        if random.random() <= self.epsilon:
            legal_actions = np.where(legal_actions)[0]
            return int(random.choice(legal_actions))

        # Exploitation
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        self.policy_net.eval()
        with torch.no_grad():
            q_values = self.policy_net(state_tensor).squeeze(0)

        # Mask out illegal actions
        mask_tensor = torch.as_tensor(legal_actions, dtype=torch.bool, device=self.device)
        q_values = torch.where(mask_tensor, q_values, torch.tensor(float('-inf'), device=self.device))
        
        # Select action with highest Q-value
        return int(torch.argmax(q_values).item())

    def train_step(self):
        """Pick a random batch from memory and train the DQN."""
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states_tensor = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_tensor = torch.as_tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards_tensor = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_tensor = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones_tensor = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        # What is the Q-value of the action we took?
        current_q = self.policy_net(states_tensor).gather(1, actions_tensor).squeeze(1)

        # What was actually the best action in the next state?
        with torch.no_grad():
            max_next_q = self.target_net(next_states_tensor).max(1)[0]
            expected_q = rewards_tensor + (1 - dones_tensor) * self.gamma * max_next_q

        # Calculate loss by comparing the predicted Q-value with the expected Q-value
        loss_fn = nn.MSELoss()
        loss = loss_fn(current_q, expected_q)

        # Backpropagate loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Reduce exploration rate as training progresses
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
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
        