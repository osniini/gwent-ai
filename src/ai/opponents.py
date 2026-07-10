import random

import numpy as np

from src.engine.gwent_env import GwentEnv

LEARNER_PLAYER = 2


def random_action(legal_mask: np.ndarray) -> int:
    return int(random.choice(np.where(legal_mask)[0]))


def greedy_action(env: GwentEnv, legal_mask: np.ndarray) -> int:
    """Play the strongest card; pass only to seal a round already won."""
    player = env.current_player
    hand = env.hand1 if player == 1 else env.hand2
    legal = np.where(legal_mask)[0]
    card_actions = [a for a in legal if a != env.pass_action]
    if not card_actions:
        return env.pass_action

    if env._opponent_board(player).passed and env._score_diff_for_player(player) > 0:
        return env.pass_action

    def best_power(type_id: int) -> int:
        return max(
            (card.current_power for card in hand if card.type_id == type_id),
            default=-1,
        )

    return max(card_actions, key=best_power)
