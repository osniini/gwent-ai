from typing import List, Dict
from src.engine.card import Card


class PlayerBoard:
    def __init__(self):
        self.rows: Dict[str, List[Card]] = {
            "melee": [],
            "ranged": [],
            "siege": [],
        }
        self.passed: bool = False

    def reset(self):
        self.rows = {'melee': [], 'ranged': [], 'siege': []}
        self.passed = False

    def get_row_score(self, row: str) -> int:
        return sum(card.current_power for card in self.rows[row])

    def get_total_score(self) -> int:
        return sum(self.get_row_score(row) for row in self.rows)


class GameBoard:
    def __init__(self):
        self.player1 = PlayerBoard()
        self.player2 = PlayerBoard()
        
    def reset(self):
        self.player1.reset()
        self.player2.reset()

    def place_card(self, player_num: int, card: Card):
        target_board = self.player1 if player_num == 1 else self.player2

        if card.row in target_board.rows:
            target_board.rows[card.row].append(card)
        else:
            raise ValueError(f"Invalid row: {card.row}")
        
    def get_scores(self) -> tuple[int, int]:
        return (self.player1.get_total_score(), self.player2.get_total_score())
    