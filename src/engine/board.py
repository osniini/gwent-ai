from typing import List, Dict
from src.engine.card import Card


class PlayerBoard:
    def __init__(self):
        self.rows: Dict[str, List[Card]] = {
            "melee": [],
            "ranged": [],
            "siege": [],
        }
        self.horn_rows: Dict[str, bool] = {
            "melee": False,
            "ranged": False,
            "siege": False,
        }
        self.passed: bool = False

    def reset(self):
        self.rows = {'melee': [], 'ranged': [], 'siege': []}
        self.horn_rows = {
            "melee": False,
            "ranged": False,
            "siege": False,
        }
        self.passed = False

    def get_row_score(self, row: str) -> int:
        return sum(card.current_power for card in self.rows[row])

    def get_row_power_split(self, row: str) -> tuple[int, int]:
        """Return (non_hero_power, hero_power) for a row using current powers."""
        non_hero = 0
        hero = 0
        for card in self.rows[row]:
            if card.hero:
                hero += card.current_power
            else:
                non_hero += card.current_power
        return non_hero, hero

    def get_total_score(self) -> int:
        return sum(self.get_row_score(row) for row in self.rows)


class GameBoard:
    def __init__(self):
        self.player1 = PlayerBoard()
        self.player2 = PlayerBoard()
        self.weather_rows = {
            "melee": False,
            "ranged": False,
            "siege": False,
        }
        
    def reset(self):
        self.player1.reset()
        self.player2.reset()
        self.weather_rows = {
            "melee": False,
            "ranged": False,
            "siege": False,
        }

    def place_card(self, player_num: int, card: Card):
        target_board = self.player1 if player_num == 1 else self.player2

        if card.row in target_board.rows:
            target_board.rows[card.row].append(card)
        else:
            raise ValueError(f"Invalid row: {card.row}")

    def apply_horn(self, player_num: int, row: str):
        """Double non-hero unit power on one of the player's rows."""
        target_board = self.player1 if player_num == 1 else self.player2

        if row not in target_board.horn_rows:
            raise ValueError(f"Invalid horn row: {row}")
        if target_board.horn_rows[row]:
            raise ValueError(f"Commander's Horn is already active on {row}")

        target_board.horn_rows[row] = True
        self.recompute_powers()

    def apply_weather(self, weather_row: str):
        """Apply a weather effect, then recompute unit powers."""
        if weather_row == "clear":
            for row in self.weather_rows:
                self.weather_rows[row] = False
        elif weather_row in self.weather_rows:
            self.weather_rows[weather_row] = True
        else:
            raise ValueError(f"Unknown weather_row: {weather_row}")
        self.recompute_powers()

    def recompute_powers(self):
        """Reset units, apply weather, then double horned non-hero rows."""
        for board in (self.player1, self.player2):
            for row, cards in board.rows.items():
                for card in cards:
                    card.reset()
                    if self.weather_rows[row] and not card.hero:
                        card.current_power = 1
                    if board.horn_rows[row] and not card.hero:
                        card.current_power *= 2

    def get_scores(self) -> tuple[int, int]:
        return (self.player1.get_total_score(), self.player2.get_total_score())
    