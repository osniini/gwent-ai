import numpy as np
from src.engine.board import GameBoard
from src.engine.card import (
    CARD_BY_NAME,
    NUM_CARD_TYPES,
    create_random_deck,
    hand_counts,
)

ROWS = ("melee", "ranged", "siege")
HORN_CARD_TYPE = CARD_BY_NAME["Commander's Horn"]
HORN_ACTION_START = NUM_CARD_TYPES
HORN_ACTIONS = {
    row: HORN_ACTION_START + index
    for index, row in enumerate(ROWS)
}
PASS_ACTION = HORN_ACTION_START + len(HORN_ACTIONS)
# Per side: current hero power for each of melee/ranged/siege (3), times 2 sides.
HERO_POWER_FEATURES = 2 * len(ROWS)
# Per side: non-hero unit count and base power total per row (6), times 2 sides.
BOARD_COMPOSITION_FEATURES = 2 * 2 * len(ROWS)
# Per side and row: count of each card type. This lets the agent identify
# same-name units already present for effects such as Tight Bond.
BOARD_CARD_COUNT_FEATURES = 2 * len(ROWS) * NUM_CARD_TYPES
BOARD_FEATURES = (
    HERO_POWER_FEATURES
    + BOARD_COMPOSITION_FEATURES
    + BOARD_CARD_COUNT_FEATURES
)
# [board features...] my_lives, opp_lives, opp_hand_len, my_passed, opp_passed, weather×3, horn×6
MY_PASSED_STATE_INDEX = BOARD_FEATURES + 3
MY_HORN_STATE_INDEX = BOARD_FEATURES + 5 + len(ROWS)
GLOBAL_STATE_SIZE = MY_HORN_STATE_INDEX + 2 * len(ROWS)
STARTING_LIVES = 2

ROUND_WIN_REWARD = 0.5
MATCH_WIN_REWARD = 1.0
SCORE_DIFF_SCALE = 0.01
ROUND_SEALED_PASS_SCALE = 0.06
CARD_PLAY_COST_SCALE = 0.02
CARD_PLAY_AHEAD_PENALTY_SCALE = 0.06
ROUND_WIN_HAND_SAVE_SCALE = 0.06
PASS_WITHOUT_LEAD_PENALTY = 0.15
PASS_WHILE_LEADING_OPEN_PENALTY = 0.15
WEATHER_RESERVE_VALUE = 4
HORN_RESERVE_VALUE = 5


class GwentEnv:
    def __init__(self):
        self.board = GameBoard()
        self.deck1 = []
        self.deck2 = []
        self.hand1 = []
        self.hand2 = []

        self.current_player = 1
        self.lives = [STARTING_LIVES, STARTING_LIVES]
        self.deferred_round_rewards: dict[int, float] = {}
        self.last_round_was_tie = False
        self.match_draw = False

        self.pass_action = PASS_ACTION
        self.action_size = PASS_ACTION + 1
        self.state_size = GLOBAL_STATE_SIZE + NUM_CARD_TYPES

    @staticmethod
    def _board_power_features(my_board, opp_board) -> list[int]:
        """Visible board power, composition, and per-row card-type counts."""
        features = []
        for board in (my_board, opp_board):
            features.extend(board.get_hero_power(row) for row in ROWS)
        for board in (my_board, opp_board):
            for row in ROWS:
                unit_count, base_power_total = board.get_non_hero_composition(row)
                features.extend((unit_count, base_power_total))
        for board in (my_board, opp_board):
            for row in ROWS:
                row_counts = [0] * NUM_CARD_TYPES
                for card in board.rows[row]:
                    row_counts[card.type_id] += 1
                features.extend(row_counts)
        return features

    def _get_state(self):
        """Numeric vector from the current player's perspective."""
        active_hand = self.hand1 if self.current_player == 1 else self.hand2
        opponent_hand = self.hand2 if self.current_player == 1 else self.hand1

        if self.current_player == 1:
            my_board, opp_board = self.board.player1, self.board.player2
            my_lives, opp_lives = self.lives[0], self.lives[1]
            my_passed = self.board.player1.passed
            opp_passed = self.board.player2.passed
        else:
            my_board, opp_board = self.board.player2, self.board.player1
            my_lives, opp_lives = self.lives[1], self.lives[0]
            my_passed = self.board.player2.passed
            opp_passed = self.board.player1.passed

        weather_bits = [
            1 if self.board.weather_rows[row] else 0
            for row in ROWS
        ]
        horn_bits = [
            *(1 if my_board.horn_rows[row] else 0 for row in ROWS),
            *(1 if opp_board.horn_rows[row] else 0 for row in ROWS),
        ]
        state = [
            *self._board_power_features(my_board, opp_board),
            my_lives,
            opp_lives,
            len(opponent_hand),
            1 if my_passed else 0,
            1 if opp_passed else 0,
            *weather_bits,
            *horn_bits,
        ] + hand_counts(active_hand)

        return np.array(state, dtype=np.float32)

    def get_state_for_player(self, player: int) -> np.ndarray:
        """State vector from a specific player's perspective (without switching turn order)."""
        saved = self.current_player
        self.current_player = player
        state = self._get_state()
        self.current_player = saved
        return state

    @staticmethod
    def legal_mask_from_state(state: np.ndarray) -> np.ndarray:
        """Rebuild legal actions from a learner-perspective state vector."""
        mask = np.zeros(PASS_ACTION + 1, dtype=bool)
        if state[MY_PASSED_STATE_INDEX] >= 0.5:
            return mask
        for type_id in range(NUM_CARD_TYPES):
            if type_id != HORN_CARD_TYPE and state[GLOBAL_STATE_SIZE + type_id] > 0:
                mask[type_id] = True
        if state[GLOBAL_STATE_SIZE + HORN_CARD_TYPE] > 0:
            for index, row in enumerate(ROWS):
                if state[MY_HORN_STATE_INDEX + index] < 0.5:
                    mask[HORN_ACTIONS[row]] = True
        mask[PASS_ACTION] = True
        return mask

    def reset(self):
        self.board.reset()
        self.lives = [STARTING_LIVES, STARTING_LIVES]
        self.deferred_round_rewards = {}
        self.last_round_was_tie = False
        self.match_draw = False
        self.current_player = 1

        self.deck1 = create_random_deck()
        self.deck2 = create_random_deck()

        self.hand1 = [self.deck1.pop() for _ in range(min(8, len(self.deck1)))]
        self.hand2 = [self.deck2.pop() for _ in range(min(8, len(self.deck2)))]

        return self._get_state()

    def get_legal_actions(self) -> np.ndarray:
        """Return a True/False list of legal actions for the current player."""
        active_hand = self.hand1 if self.current_player == 1 else self.hand2
        active_board = self.board.player1 if self.current_player == 1 else self.board.player2

        mask = np.zeros(self.action_size, dtype=bool)

        if active_board.passed:
            return mask

        for type_id, count in enumerate(hand_counts(active_hand)):
            if type_id != HORN_CARD_TYPE and count > 0:
                mask[type_id] = True
            elif type_id == HORN_CARD_TYPE and count > 0:
                for row, action in HORN_ACTIONS.items():
                    if not active_board.horn_rows[row]:
                        mask[action] = True

        mask[self.pass_action] = True

        return mask

    def _remove_card_by_type(self, hand: list, type_id: int):
        for i, card in enumerate(hand):
            if card.type_id == type_id:
                return hand.pop(i)
        raise ValueError(f"No card of type {type_id} in hand")

    def _score_diff_for_player(self, player: int) -> float:
        p1_score, p2_score = self.board.get_scores()
        diff = p1_score - p2_score
        return diff if player == 1 else -diff

    @staticmethod
    def _card_reserve_value(card) -> int:
        """Value a card's future strategic use for reward shaping."""
        if card.weather_row is not None:
            return WEATHER_RESERVE_VALUE
        if card.effect == "horn":
            return HORN_RESERVE_VALUE
        return card.current_power

    @classmethod
    def _hand_value(cls, hand: list) -> int:
        return sum(cls._card_reserve_value(card) for card in hand)

    def _opponent_board(self, player: int):
        return self.board.player1 if player == 2 else self.board.player2

    def _hand_save_value_scale(self, player: int) -> float:
        """Saved cards only matter when another round can follow a loss."""
        lives = self.lives[0] if player == 1 else self.lives[1]
        if lives <= 1 or STARTING_LIVES <= 1:
            return 0.0
        return (lives - 1) / (STARTING_LIVES - 1)

    def _pass_without_lead_penalty(self, player: int, *, passed: bool) -> float:
        """Discourage passing while tied or behind before the round is decided."""
        if not passed:
            return 0.0
        if self._opponent_board(player).passed:
            return 0.0
        if self._score_diff_for_player(player) > 0:
            return 0.0
        return PASS_WITHOUT_LEAD_PENALTY

    def _pass_while_leading_open_penalty(self, player: int, *, passed: bool) -> float:
        """On last life, passing while ahead lets a still-active opponent overtake you."""
        if not passed:
            return 0.0
        if self._opponent_board(player).passed:
            return 0.0
        if self._score_diff_for_player(player) <= 0:
            return 0.0
        lives = self.lives[0] if player == 1 else self.lives[1]
        if lives > 1:
            return 0.0
        return PASS_WHILE_LEADING_OPEN_PENALTY

    def _card_play_ahead_penalty(
        self,
        player: int,
        played_value: int,
        score_after: float,
    ) -> float:
        """Discourage dumping cards while ahead only when saving them for another round matters."""
        if played_value == 0 or score_after <= 0:
            return 0.0
        if self._opponent_board(player).passed:
            return 0.0
        scale = self._hand_save_value_scale(player)
        if scale == 0.0:
            return 0.0
        return CARD_PLAY_AHEAD_PENALTY_SCALE * played_value * scale

    def _round_sealed_pass_bonus(self, player: int, hand: list, *, passed: bool) -> float:
        """Opponent passed and we are ahead — reward closing the round with cards saved."""
        if not passed or self._hand_value(hand) == 0:
            return 0.0
        if not self._opponent_board(player).passed:
            return 0.0
        if self._score_diff_for_player(player) <= 0:
            return 0.0
        scale = self._hand_save_value_scale(player)
        if scale == 0.0:
            return 0.0
        return ROUND_SEALED_PASS_SCALE * self._hand_value(hand) * scale

    @staticmethod
    def _perspective_from_p1(player: int, p1_positive: float) -> float:
        return p1_positive if player == 1 else -p1_positive

    def _round_win_hand_bonus(self, player: int, outcome: int) -> float:
        """Bonus for winning the round with unused hand value."""
        if outcome == 0:
            return 0.0
        won = (outcome == 1 and player == 1) or (outcome == -1 and player == 2)
        if not won:
            return 0.0
        hand = self.hand1 if player == 1 else self.hand2
        hand_value = self._hand_value(hand)
        if hand_value == 0:
            return 0.0
        scale = self._hand_save_value_scale(player)
        if scale == 0.0:
            return 0.0
        return ROUND_WIN_HAND_SAVE_SCALE * hand_value * scale

    def consume_deferred_round_reward(self, player: int) -> float:
        """Apply deferred round-end reward for a player who already ended their turn."""
        return self.deferred_round_rewards.pop(player, 0.0)

    def _match_is_over(self) -> bool:
        return self.match_draw or self.lives[0] == 0 or self.lives[1] == 0

    def get_match_reward_for_player(self, player: int) -> float:
        """Terminal match reward from the given player's perspective."""
        if self.match_draw or (self.lives[0] == 0 and self.lives[1] == 0):
            return -MATCH_WIN_REWARD
        if self.lives[0] == 0:
            return self._perspective_from_p1(player, -MATCH_WIN_REWARD)
        if self.lives[1] == 0:
            return self._perspective_from_p1(player, MATCH_WIN_REWARD)
        return 0.0

    def _set_deferred_round_rewards(self, outcome: int) -> None:
        self.deferred_round_rewards = {}
        for player in (1, 2):
            if outcome == 1:
                reward = ROUND_WIN_REWARD if player == 1 else -ROUND_WIN_REWARD
            elif outcome == -1:
                reward = -ROUND_WIN_REWARD if player == 1 else ROUND_WIN_REWARD
            else:
                reward = -ROUND_WIN_REWARD
            reward += self._round_win_hand_bonus(player, outcome)
            self.deferred_round_rewards[player] = reward

    def step(self, action: int):
        """Execute a card, targeted Horn, or pass action."""

        acting_player = self.current_player
        active_board = self.board.player1 if acting_player == 1 else self.board.player2
        active_hand = self.hand1 if acting_player == 1 else self.hand2
        legal_actions = self.get_legal_actions()
        if action < 0 or action >= self.action_size or not legal_actions[action]:
            raise ValueError(f"Illegal action: {action}")

        score_before = self._score_diff_for_player(acting_player)
        played_value = 0

        if action == self.pass_action:
            active_board.passed = True
        elif 0 <= action < NUM_CARD_TYPES:
            card = self._remove_card_by_type(active_hand, action)
            played_value = self._card_reserve_value(card)
            if card.weather_row is not None:
                self.board.apply_weather(card.weather_row)
            else:
                self.board.place_card(acting_player, card)
                self.board.recompute_powers()
        elif action in HORN_ACTIONS.values():
            row = next(row for row, horn_action in HORN_ACTIONS.items() if horn_action == action)
            card = self._remove_card_by_type(active_hand, HORN_CARD_TYPE)
            played_value = self._card_reserve_value(card)
            self.board.apply_horn(acting_player, row)
        else:
            raise ValueError(f"Invalid action: {action}")

        score_after = self._score_diff_for_player(acting_player)
        reward = SCORE_DIFF_SCALE * (score_after - score_before)

        if action != self.pass_action:
            # Tax spending a card's future strategic value when saving it can matter.
            reward -= (
                CARD_PLAY_COST_SCALE
                * played_value
                * self._hand_save_value_scale(acting_player)
            )
            reward -= self._card_play_ahead_penalty(
                acting_player,
                played_value,
                score_after,
            )

        round_ending = self.board.player1.passed and self.board.player2.passed

        if action == self.pass_action and not round_ending:
            reward += self._round_sealed_pass_bonus(
                acting_player,
                active_hand,
                passed=True,
            )
            reward -= self._pass_without_lead_penalty(acting_player, passed=True)
            reward -= self._pass_while_leading_open_penalty(acting_player, passed=True)

        if round_ending:
            round_outcome = self._check_round_end()
            if not self.match_draw:
                self._set_deferred_round_rewards(round_outcome)
            reward += self.deferred_round_rewards.pop(acting_player, 0.0)

        next_player = 2 if self.current_player == 1 else 1
        next_board = self.board.player2 if next_player == 2 else self.board.player1

        if not next_board.passed:
            self.current_player = next_player
        elif active_board.passed:
            pass

        done = False

        if self._match_is_over():
            done = True
            reward += self.get_match_reward_for_player(acting_player)

        return self._get_state(), reward, done

    def _check_round_end(self) -> int:
        """Return +1 if P1 won the round, -1 if P2 won, 0 on a tie."""
        p1_score, p2_score = self.board.get_scores()
        self.last_round_was_tie = False
        self.match_draw = False

        if p1_score > p2_score:
            self.lives[1] -= 1
            outcome = 1
        elif p2_score > p1_score:
            self.lives[0] -= 1
            outcome = -1
        else:
            outcome = 0
            self.last_round_was_tie = True
            if self.lives[0] == 1 and self.lives[1] == 1:
                self.match_draw = True
                self.lives[0] = 0
                self.lives[1] = 0
            else:
                self.lives[0] -= 1
                self.lives[1] -= 1

        self.board.reset()
        return outcome
