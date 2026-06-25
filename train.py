import numpy as np
from src.engine.gwent_env import GwentEnv
from src.ai.agent import DQNAgent

def train_gwent(num_episodes: int = 5000):
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
        pending_reward = {1: 0.0, 2: 0.0}
        last_acting_player = None

        while not done:
            acting_player = env.current_player
            pending_reward[acting_player] += env.consume_deferred_round_reward(acting_player)

            legal_actions = env.get_legal_actions()

            action = agent.select_action(state, legal_actions)

            if last_state[acting_player] is not None:
                agent.memory.push(
                    last_state[acting_player],
                    last_action[acting_player],
                    pending_reward[acting_player],
                    state,
                    False,
                )
                pending_reward[acting_player] = 0.0

            last_state[acting_player] = state
            last_action[acting_player] = action

            state, reward, done = env.step(action)
            pending_reward[acting_player] += reward
            last_acting_player = acting_player

            agent.train_step()

        for player in (1, 2):
            if last_state[player] is None:
                continue

            terminal_reward = pending_reward[player]
            if player != last_acting_player:
                terminal_reward += env.get_match_reward_for_player(player)

            agent.memory.push(
                last_state[player],
                last_action[player],
                terminal_reward,
                state,
                True,
            )

        if episode % 100 == 0:
            agent.update_target_network()
            print(f"Episode {episode}/{num_episodes} | epsilon: {agent.epsilon:.3f}")

    agent.save("models/gwent_agent_alpha.pth")

if __name__ == "__main__":
    train_gwent()
