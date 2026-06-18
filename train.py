import numpy as np
from src.engine.gwent_env import GwentEnv
from src.ai.agent import DQNAgent

def train_gwent(num_episodes: int = 1000):
    env = GwentEnv()

    agent = DQNAgent(env.state_size, env.action_size)

    print("Training Gwent...")
    print(f"Device: {agent.device}")
    print("---------------------------------------")

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        done = False

        last_state = {1: None, 2: None}
        last_action = {1: None, 2: None}
        
        while not done:
            current_player = env.current_player

            legal_actions = env.get_legal_actions()

            action = agent.select_action(state, legal_actions)

            if last_state[current_player] is not None:
                agent.memory.push(
                    last_state[current_player],
                    last_action[current_player],
                    0.0,
                    state,
                    False
                )
            
            last_state[current_player] = state
            last_action[current_player] = action

            next_state, reward, done = env.step(action)
            state = next_state

            agent.train_step()

        p1_reward = reward if env.current_player == 1 else -reward
        p2_reward = -reward if env.current_player == 1 else reward

        if last_state[1] is not None:
            agent.memory.push(
                last_state[1],
                last_action[1],
                p1_reward,
                state,
                True
            )
        if last_state[2] is not None:
            agent.memory.push(
                last_state[2],
                last_action[2],
                p2_reward,
                state,
                True
            )

        if episode % 20 == 0:
            agent.update_target_network()
            print(f"Episode {episode}/{num_episodes} | epsilon: {agent.epsilon:.3f}")

if __name__ == "__main__":
    train_gwent()