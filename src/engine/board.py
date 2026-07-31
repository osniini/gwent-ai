from collections import Counter
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

    def get_hero_power(self, row: str) -> int:
        """Return the current total power of a row's hero units."""
        return sum(card.current_power for card in self.rows[row] if card.hero)

    def get_non_hero_composition(self, row: str) -> tuple[int, int]:
        """Return (unit_count, base_power_total) for a row's non-hero units."""
        non_heroes = [
            card for card in self.rows[row]
            if card.unit and not card.hero
        ]
        return len(non_heroes), sum(card.base_power for card in non_heroes)

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

    def place_card(self, player_num: int, card: Card, row: str | None = None):
        target_board = self.player1 if player_num == 1 else self.player2
        row = row or card.row

        if row in target_board.rows:
            target_board.rows[row].append(card)
        else:
            raise ValueError(f"Invalid row: {row}")

    def replace_with_decoy(self, player_num: int, target_type_id: int, decoy: Card) -> Card:
        """Return an eligible unit to hand and put Decoy in its row."""
        target_board = self.player1 if player_num == 1 else self.player2
        for row, cards in target_board.rows.items():
            for index, card in enumerate(cards):
                if (
                    card.type_id == target_type_id
                    and card.unit
                    and not card.hero
                ):
                    cards[index] = decoy
                    card.reset()
                    self.recompute_powers()
                    return card
        raise ValueError(f"No eligible Decoy target with type_id {target_type_id}")

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

    def scorch(self) -> list[tuple[int, Card]]:
        """Destroy every non-hero card tied for highest current power."""
        cards_on_board = [
            card
            for board in (self.player1, self.player2)
            for cards in board.rows.values()
            for card in cards
            if not card.hero
        ]
        if not cards_on_board:
            return []

        highest_power = max(card.current_power for card in cards_on_board)
        destroyed = []
        for player, board in ((1, self.player1), (2, self.player2)):
            for row, cards in board.rows.items():
                survivors = []
                for card in cards:
                    if not card.hero and card.current_power == highest_power:
                        destroyed.append((player, card))
                    else:
                        survivors.append(card)
                board.rows[row] = survivors
        return destroyed

    def recompute_powers(self):
        """Reset units, then apply weather, Tight Bond, Horn, and Morale Boost."""
        for board in (self.player1, self.player2):
            for row, cards in board.rows.items():
                for card in cards:
                    card.reset()
                    if card.unit and self.weather_rows[row] and not card.hero:
                        card.current_power = 1

                tight_bond_counts = Counter(
                    card.name
                    for card in cards
                    if card.unit and card.effect == "tight_bond" and not card.hero
                )
                for card in cards:
                    if card.unit and card.effect == "tight_bond" and not card.hero:
                        card.current_power *= tight_bond_counts[card.name]

                if board.horn_rows[row]:
                    for card in cards:
                        if card.unit and not card.hero:
                            card.current_power *= 2

                morale_boost_count = sum(
                    card.unit and card.effect == "morale_boost"
                    for card in cards
                )
                if morale_boost_count:
                    for card in cards:
                        if not card.unit:
                            continue
                        # Every Morale Boost unit adds +1 to every other unit
                        # on its row; boosts can therefore affect each other.
                        own_boost = 1 if card.effect == "morale_boost" else 0
                        card.current_power += morale_boost_count - own_boost

    def get_scores(self) -> tuple[int, int]:
        return (self.player1.get_total_score(), self.player2.get_total_score())
    