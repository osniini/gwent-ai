import random
import numpy as np
from src.engine.board import GameBoard
from src.engine.card import create_starter_deck, Card

class GwentEnv:
    def __init__(self):
        self.board = GameBoard()
        self.deck1 = []
        self.deck2 = []
        self.hand1 = []
        self.hand2 = []
    
        self.current_player = 1
        self.round_wins = [0, 0]

        self.action_size = 9
        self.state_size = 30

    def _get_state(self):
        """Numeric vector representation of the state."""
        p1_score, p2_score = self.board.get_scores()
        active_hand = self.hand1 if self.current_player == 1 else self.hand2
        opponent_hand = self.hand2 if self.current_player == 1 else self.hand1

        state = [
            p1_score, p2_score,
            self.round_wins[0], self.round_wins[1],
            len(active_hand), len(opponent_hand),
            1 if self.board.player1.passed else 0,
            1 if self.board.player2.passed else 0,
        ]

        while len(state) < self.state_size:
            state.append(0.0)
        
        return np.array(state)

    def reset(self):
        self.board.reset()
        self.round_wins = [0, 0]
        self.current_player = 1

        self.deck1 = create_starter_deck()
        self.deck2 = create_starter_deck()

        self.hand1 = [self.deck1.pop() for _ in range(min(8, len(self.deck1)))]
        self.hand2 = [self.deck2.pop() for _ in range(min(8, len(self.deck2)))]
        
        return self._get_state()

    def step(self, action: int):
        """Execute single action: 0-7 play a card from hand, 8 pass."""

        active_board = self.board.player1 if self.current_player == 1 else self.board.player2
        active_hand = self.hand1 if self.current_player == 1 else self.hand2

        if action == 8:
            active_board.passed = True
        else:
            if action < len(active_hand):
                card = active_hand.pop(action)
                self.board.place_card(self.current_player, card)
            else:
                active_board.passed = True
        
        if self.board.player1.passed and self.board.player2.passed:
            self._check_round_end()

        next_player = 2 if self.current_player == 1 else 1
        next_board = self.board.player2 if next_player == 2 else self.board.player1

        if not next_board.passed:
            self.current_player = next_player
        elif active_board.passed:
            pass
        
        done = False
        reward = 0.0

        if self.round_wins[0] >= 2:
            done = True
            reward = 1.0 if self.current_player == 1 else -1.0
        elif self.round_wins[1] >= 2:
            done = True
            reward = -1.0 if self.current_player == 1 else 1.0
        
        # If the game is not done but no one can do anything anymore
        if not done and len(self.hand1) == 0 and len(self.hand2) == 0 and self.board.player1.passed and self.board.player2.passed:
            done = True
            if self.round_wins[0] > self.round_wins[1]:
                reward = 1.0 if self.current_player == 1 else -1.0
            else:
                reward = -1.0 if self.current_player == 1 else 1.0
        
        return self._get_state(), reward, done

    def _check_round_end(self):
        p1_score, p2_score = self.board.get_scores()

        if p1_score > p2_score:
            self.round_wins[0] += 1
        elif p2_score > p1_score:
            self.round_wins[1] += 1
        else:
            self.round_wins[0] += 1
            self.round_wins[1] += 1
        
        self.board.reset()