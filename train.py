import numpy as np

from src.engine.gwent_env import GwentEnv
from src.ai.agent import DQNAgent
from src.ai.curriculum import (
    FrozenOpponentPool,
    begin_frozen_exploration,
    training_phase,
)
from src.ai.evaluation import evaluate_opponent
from src.ai.metrics import TrainingMetricsLogger
from src.ai.opponents import LEARNER_PLAYER, dummy_action, random_action

NUM_EPISODES = 500000 # Total number of episodes to train for
NUM_ENVS = 128 # Number of parallel environments to train on
TRAIN_EVERY = 4 # Run train_step every N global steps
TRAIN_STEPS_PER_UPDATE = 1 # Number of gradient steps per update
TARGET_UPDATE_EVERY = 250 # Sync target net, log metrics, refresh frozen pool (every N completed episodes)
EVALUATION_EVERY = 25000 # Evaluate every N episodes
FROZEN_EVALUATION_LAG = 5000  # Eval vs newest snapshot saved at least N episodes ago

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
        "opponent_net": None,
    }


def _reset_tracker(tracker: dict) -> None:
    tracker["env"].reset()
    tracker["state"] = _learner_state(tracker["env"])
    tracker["done"] = False
    tracker["last_state"] = None
    tracker["last_action"] = None
    tracker["pending_reward"] = 0.0
    tracker["last_acting_player"] = None
    tracker["opponent_net"] = None


def _push_terminal_transitions(tracker: dict, agent: DQNAgent) -> None:
    if tracker["last_state"] is None:
        return

    env = tracker["env"]
    terminal_reward = (
        tracker["pending_reward"]
        + env.consume_deferred_round_reward(LEARNER_PLAYER)
    )
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

    frozen_nets: dict[int, object] = {}
    frozen_states: dict[int, list[np.ndarray]] = {}
    frozen_masks: dict[int, list[np.ndarray]] = {}
    frozen_slots: dict[int, list[int]] = {}

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
        elif phase == "dummy":
            actions[slot] = dummy_action(env, legal)
        else:
            opponent_net = tracker["opponent_net"]
            if opponent_net is None:
                opponent_net = frozen_pool.sample_opponent()
                tracker["opponent_net"] = opponent_net
            net_id = id(opponent_net)
            frozen_nets[net_id] = opponent_net
            frozen_slots.setdefault(net_id, []).append(slot)
            frozen_states.setdefault(net_id, []).append(
                env.get_state_for_player(env.current_player)
            )
            frozen_masks.setdefault(net_id, []).append(legal)

    for slot, action in zip(learner_slots, agent.select_actions_batch(learner_states, learner_masks)):
        actions[slot] = action

    for net_id, states in frozen_states.items():
        frozen_actions = agent.select_greedy_actions_batch(
            states,
            frozen_masks[net_id],
            policy_net=frozen_nets[net_id],
        )
        for slot, action in zip(frozen_slots[net_id], frozen_actions):
            actions[slot] = action

    return [int(a) for a in actions]


def _run_evaluation(
    agent: DQNAgent,
    frozen_pool: FrozenOpponentPool,
    metrics_logger: TrainingMetricsLogger,
    episode: int,
) -> None:
    """Evaluate greedily without adding data to replay memory."""
    benchmarks = [("random", 300, None), ("dummy", 300, None)]
    frozen_opponent = frozen_pool.snapshot_episodes_ago(
        episode,
        FROZEN_EVALUATION_LAG,
    )
    if frozen_opponent is None:
        benchmarks[-1] = ("dummy", 700, None)
        print("Evaluation: no frozen snapshot old enough; using 700 dummy matches.")
    else:
        benchmarks.append(("frozen", 400, frozen_opponent))

    for opponent, matches, opponent_net in benchmarks:
        results = evaluate_opponent(
            agent,
            opponent=opponent,
            matches=matches,
            opponent_net=opponent_net,
        )
        metrics_logger.write_evaluation(
            episode=episode,
            opponent=opponent,
            results=results,
        )
        total = sum(results.values())
        print(
            f"  eval vs {opponent}: "
            f"{results['wins'] / total:.1%} "
            f"(W/L/D {results['wins']}/{results['losses']}/{results['draws']})"
        )


def train_gwent(
    num_episodes: int = NUM_EPISODES,
    num_envs: int = NUM_ENVS,
    train_every: int = TRAIN_EVERY,
    train_steps: int = TRAIN_STEPS_PER_UPDATE,
):
    sample_env = GwentEnv()
    agent = DQNAgent(sample_env.state_size, sample_env.action_size)
    frozen_pool = FrozenOpponentPool(agent)
    metrics_logger = TrainingMetricsLogger()

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
    metric_sums: dict[str, float] = {}
    metric_count = 0
    window_results = {"wins": 0, "losses": 0, "draws": 0}
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
                frozen_pool.on_enter_frozen_phase(agent, episodes_done)
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
                metrics = agent.train_step()
                if metrics is not None:
                    metric_count += 1
                    for name, value in metrics.items():
                        metric_sums[name] = metric_sums.get(name, 0.0) + value

        for tracker in trackers:
            if not tracker["done"]:
                continue

            env = tracker["env"]
            _push_terminal_transitions(tracker, agent)
            episodes_done += 1
            agent.decay_epsilon()
            if env.match_draw:
                window_results["draws"] += 1
            elif env.lives[LEARNER_PLAYER - 1] == 0:
                window_results["losses"] += 1
            else:
                window_results["wins"] += 1

            if episodes_done % TARGET_UPDATE_EVERY == 0:
                agent.update_target_network()
                if training_phase(episodes_done, num_episodes) == "frozen":
                    frozen_pool.maybe_refresh(agent, episodes_done)
                window_total = sum(window_results.values())
                win_rate = window_results["wins"] / window_total if window_total else 0.0
                print(
                    f"Episode {episodes_done}/{num_episodes} | "
                    f"phase: {training_phase(episodes_done, num_episodes)} | "
                    f"epsilon: {agent.epsilon:.3f} | "
                    f"win rate: {win_rate:.1%} "
                    f"(W/L/D {window_results['wins']}/"
                    f"{window_results['losses']}/{window_results['draws']})"
                )
                averages: dict[str, float] = {}
                if metric_count:
                    averages = {
                        name: total / metric_count
                        for name, total in metric_sums.items()
                    }
                    print(
                        "  train: "
                        f"loss={averages['loss']:.4f} | "
                        f"Q={averages['q_mean']:.3f}±{averages['q_std']:.3f} | "
                        f"target={averages['target_mean']:.3f}±{averages['target_std']:.3f} | "
                        f"|TD|={averages['td_abs_mean']:.3f} "
                        f"(std={averages['td_std']:.3f}) | "
                        f"grad norm={averages['grad_norm']:.3f} | "
                        f"finite gradients={averages['gradients_finite']:.1%}"
                    )
                metrics_logger.write_checkpoint(
                    episode=episodes_done,
                    global_step=global_step,
                    phase=training_phase(episodes_done, num_episodes),
                    epsilon=agent.epsilon,
                    results=window_results,
                    averages=averages,
                )
                metric_sums = {}
                metric_count = 0
                window_results = {"wins": 0, "losses": 0, "draws": 0}

            if episodes_done % EVALUATION_EVERY == 0:
                _run_evaluation(
                    agent,
                    frozen_pool,
                    metrics_logger,
                    episodes_done,
                )

            if episodes_done < num_episodes:
                _reset_tracker(tracker)

    agent.save("models/gwent_agent_delta.pth")
    metrics_logger.close()


if __name__ == "__main__":
    train_gwent()
