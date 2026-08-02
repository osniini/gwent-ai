import random

import numpy as np

from src.engine.card import CARD_CATALOG, NUM_CARD_TYPES
from src.engine.gwent_env import (
    DECOY_ACTIONS,
    HORN_ACTIONS,
    MEDIC_ACTIONS,
    MEDIC_NO_TARGET_ACTIONS,
    REDRAW_DONE_ACTION,
    GwentEnv,
)

LEARNER_PLAYER = 2


def random_action(legal_mask: np.ndarray) -> int:
    return int(random.choice(np.where(legal_mask)[0]))


def dummy_action(env: GwentEnv, legal_mask: np.ndarray) -> int:
    """Choose a low-value unit unless a special card has clear tactical value."""
    player = env.current_player
    hand = env.hand1 if player == 1 else env.hand2
    board = env.board.player1 if player == 1 else env.board.player2
    opponent_board = env._opponent_board(player)
    legal = np.where(legal_mask)[0]
    non_pass_actions = [action for action in legal if action != env.pass_action]
    if not non_pass_actions:
        return env.pass_action

    if opponent_board.passed and env._score_diff_for_player(player) > 0:
        return env.pass_action

    if env.redraw_active:
        redraw_cards = [action for action in legal if action != REDRAW_DONE_ACTION]
        if not redraw_cards:
            return REDRAW_DONE_ACTION
        return min(redraw_cards, key=lambda action: _card_power(hand, action))

    basic_actions = [
        action
        for action in non_pass_actions
        if action < NUM_CARD_TYPES and CARD_CATALOG[action].get("effect") is None
        and CARD_CATALOG[action].get("weather_row") is None
    ]
    least_basic_power = min(
        (_card_power(hand, action) for action in basic_actions),
        default=float("inf"),
    )
    special_actions = [
        (action, _special_value(env, action, player, board, opponent_board, hand))
        for action in non_pass_actions
        if action not in basic_actions
    ]
    best_special_action, best_special_value = max(
        special_actions,
        key=lambda candidate: candidate[1],
        default=(None, float("-inf")),
    )

    # A special must do more now than spending the weakest ordinary unit.
    if best_special_action is not None and best_special_value > least_basic_power:
        return best_special_action
    if basic_actions:
        return min(basic_actions, key=lambda action: _card_power(hand, action))
    return int(best_special_action)


def _card_power(hand: list, type_id: int) -> int:
    return min(
        (card.current_power for card in hand if card.type_id == type_id),
        default=0,
    )


def _special_value(
    env: GwentEnv,
    action: int,
    player: int,
    board,
    opponent_board,
    hand: list,
) -> float:
    """Estimate a special action's immediate tactical value."""
    if action in HORN_ACTIONS.values():
        row = next(row for row, horn_action in HORN_ACTIONS.items() if horn_action == action)
        return sum(
            card.current_power
            for card in board.rows[row]
            if card.unit and not card.hero
        )

    decoy_targets = {
        decoy_action: target_type_id
        for target_type_id, decoy_action in DECOY_ACTIONS.items()
    }
    if action in decoy_targets:
        target_type_id = decoy_targets[action]
        target = next(
            card
            for cards in board.rows.values()
            for card in cards
            if card.type_id == target_type_id and card.unit and not card.hero
        )
        replay_bonus = {
            "spy": 6,
            "medic": 5,
            "muster": 4,
        }.get(target.effect, 0)
        return replay_bonus - target.current_power

    medic_targets = {
        medic_action: target_type_id
        for (_, target_type_id), medic_action in MEDIC_ACTIONS.items()
    }
    if action in medic_targets:
        return CARD_CATALOG[medic_targets[action]].get("power", 0)
    if action in MEDIC_NO_TARGET_ACTIONS.values():
        medic_type_id = next(
            type_id
            for type_id, medic_action in MEDIC_NO_TARGET_ACTIONS.items()
            if medic_action == action
        )
        # Without a revival target, a Medic is only an ordinary unit.
        return -_card_power(hand, medic_type_id)

    if action >= NUM_CARD_TYPES:
        return float("-inf")

    card = CARD_CATALOG[action]
    effect = card.get("effect")
    weather_row = card.get("weather_row")

    if weather_row == "clear":
        return _weather_recovery_value(board) - _weather_recovery_value(opponent_board)
    if weather_row is not None:
        return _weather_damage_value(opponent_board, weather_row) - _weather_damage_value(
            board, weather_row
        )
    if effect == "scorch":
        return _scorch_value(opponent_board) - _scorch_value(board)
    if effect == "spy":
        deck = env.deck1 if player == 1 else env.deck2
        return 3 * min(2, len(deck)) - _card_power(hand, action)
    if effect == "muster":
        deck = env.deck1 if player == 1 else env.deck2
        name = card["name"]
        return sum(
            candidate.current_power
            for candidate in hand + deck
            if candidate.name == name
        )
    if effect == "tight_bond":
        same_name_count = sum(
            candidate.name == card["name"]
            for candidate in board.rows[card["row"]]
        )
        return card["power"] * (2 * same_name_count + 1)
    if effect == "morale_boost":
        return len(board.rows[card["row"]])
    return _card_power(hand, action)


def _weather_damage_value(board, row: str) -> int:
    return sum(
        max(card.current_power - 1, 0)
        for card in board.rows[row]
        if card.unit and not card.hero
    )


def _weather_recovery_value(board) -> int:
    return sum(
        max(card.base_power - card.current_power, 0)
        for row in board.rows.values()
        for card in row
        if card.unit and not card.hero
    )


def _scorch_value(board) -> int:
    non_heroes = [
        card
        for cards in board.rows.values()
        for card in cards
        if not card.hero
    ]
    if not non_heroes:
        return 0
    highest_power = max(card.current_power for card in non_heroes)
    return sum(
        card.current_power
        for card in non_heroes
        if card.current_power == highest_power
    )
