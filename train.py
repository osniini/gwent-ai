import numpy as np

from src.engine.gwent_env import GwentEnv
from src.ai.agent import DQNAgent
from src.ai.curriculum import (
    FrozenOpponentPool,
    begin_frozen_exploration,
    training_phase,
)
from src.ai.opponents import LEARNER_PLAYER, greedy_action, random_action

NUM_EPISODES = 60000
NUM_ENVS = 32
TRAIN_EVERY = 4
TRAIN_STEPS_PER_UPDATE = 3
TARGET_UPDATE_EVERY = 250


def _learner_state(env: GwentEnv) -> np.ndarray:
    return env.get_state_for_player(LEARNER_PLAYER)


def _new_tracker() -> dict:
    env = GwentEnv()
    env.reset()
    return {
        "env": env,
        "state": _learner_state(env),
        "done": False,
        "last_state": None,
        "last_action": None,
        "pending_reward": 0.0,
        "last_acting_player": None,
    }


def _reset_tracker(tracker: dict) -> None:
    tracker["env"].reset()
    tracker["state"] = _learner_state(tracker["env"])
    tracker["done"] = False
    tracker["last_state"] = None
    tracker["last_action"] = None
    tracker["pending_reward"] = 0.0
    tracker["last_acting_player"] = None


def _push_terminal_transitions(tracker: dict, agent: DQNAgent) -> None:
    if tracker["last_state"] is None:
        return

    env = tracker["env"]
    terminal_reward = tracker["pending_reward"]
    if LEARNER_PLAYER != tracker["last_acting_player"]:
        terminal_reward += env.get_match_reward_for_player(LEARNER_PLAYER)

    agent.memory.push(
        tracker["last_state"],
        tracker["last_action"],
        terminal_reward,
        _learner_state(env),
        True,
    )


def _select_actions(
    active: list[dict],
    agent: DQNAgent,
    phase: str,
    frozen_pool: FrozenOpponentPool,
) -> list[int]:
    learner_states: list[np.ndarray] = []
    learner_masks: list[np.ndarray] = []
    learner_slots: list[int] = []

    frozen_states: list[np.ndarray] = []
    frozen_masks: list[np.ndarray] = []
    frozen_slots: list[int] = []

    actions: list[int | None] = [None] * len(active)

    for slot, tracker in enumerate(active):
        env = tracker["env"]
        legal = env.get_legal_actions()

        if env.current_player == LEARNER_PLAYER:
            learner_slots.append(slot)
            learner_states.append(_learner_state(env))
            learner_masks.append(legal)
            continue

        if phase == "random":
            actions[slot] = random_action(legal)
        elif phase == "greedy":
            actions[slot] = greedy_action(env, legal)
        else:
            frozen_slots.append(slot)
            frozen_states.append(env.get_state_for_player(env.current_player))
            frozen_masks.append(legal)

    for slot, action in zip(learner_slots, agent.select_actions_batch(learner_states, learner_masks)):
        actions[slot] = action

    if frozen_states:
        frozen_actions = agent.select_greedy_actions_batch(
            frozen_states,
            frozen_masks,
            policy_net=frozen_pool.net,
        )
        for slot, action in zip(frozen_slots, frozen_actions):
            actions[slot] = action

    return [int(a) for a in actions]


def train_gwent(
    num_episodes: int = NUM_EPISODES,
    num_envs: int = NUM_ENVS,
    train_every: int = TRAIN_EVERY,
    train_steps: int = TRAIN_STEPS_PER_UPDATE,
):
    sample_env = GwentEnv()
    agent = DQNAgent(sample_env.state_size, sample_env.action_size)
    frozen_pool = FrozenOpponentPool(agent)

    # agent.load("models/gwent_agent_alpha.pth")

    print("Training Gwent...")
    print(f"Device: {agent.device}")
    print(f"Parallel envs: {num_envs}")
    print(f"Learner: player {LEARNER_PLAYER}")
    print(f"Curriculum: brief random warmup → frozen self (epsilon decays over full run)")
    print(f"Train every {train_every} steps, {train_steps} gradient steps per update")
    print(f"Batch size: {agent.batch_size}")
    print("---------------------------------------")

    trackers = [_new_tracker() for _ in range(num_envs)]
    episodes_done = 0
    global_step = 0
    agent.epsilon = 1.0
    agent.configure_epsilon_decay(num_episodes)
    current_phase = training_phase(episodes_done, num_episodes)
    print(f"Phase: {current_phase}")

    while episodes_done < num_episodes:
        phase = training_phase(episodes_done, num_episodes)
        if phase != current_phase:
            current_phase = phase
            print(f"Phase: {current_phase}")
            if phase == "frozen":
                frozen_pool.on_enter_frozen_phase(agent)
                begin_frozen_exploration(agent, num_episodes)

        active = [t for t in trackers if not t["done"]]

        for tracker in active:
            env = tracker["env"]
            acting_player = env.current_player
            if acting_player == LEARNER_PLAYER:
                tracker["pending_reward"] += env.consume_deferred_round_reward(acting_player)

        actions = _select_actions(active, agent, phase, frozen_pool)

        for tracker, action in zip(active, actions):
            env = tracker["env"]
            acting_player = env.current_player
            state = tracker["state"]

            if acting_player == LEARNER_PLAYER:
                if tracker["last_state"] is not None:
                    agent.memory.push(
                        tracker["last_state"],
                        tracker["last_action"],
                        tracker["pending_reward"],
                        _learner_state(env),
                        False,
                    )
                    tracker["pending_reward"] = 0.0

                tracker["last_state"] = _learner_state(env)
                tracker["last_action"] = action

            state, reward, done = env.step(action)

            if acting_player == LEARNER_PLAYER:
                tracker["pending_reward"] += reward

            tracker["last_acting_player"] = acting_player
            tracker["state"] = _learner_state(env)
            tracker["done"] = done

        global_step += 1
        if global_step % train_every == 0:
            for _ in range(train_steps):
                agent.train_step()

        for tracker in trackers:
            if not tracker["done"]:
                continue

            _push_terminal_transitions(tracker, agent)
            episodes_done += 1
            agent.decay_epsilon()

            if episodes_done % TARGET_UPDATE_EVERY == 0:
                agent.update_target_network()
                if training_phase(episodes_done, num_episodes) == "frozen":
                    frozen_pool.maybe_refresh(agent)
                print(
                    f"Episode {episodes_done}/{num_episodes} | "
                    f"phase: {training_phase(episodes_done, num_episodes)} | "
                    f"epsilon: {agent.epsilon:.3f}"
                )

            if episodes_done < num_episodes:
                _reset_tracker(tracker)

    agent.save("models/gwent_agent_gamma.pth")


if __name__ == "__main__":
    train_gwent()
