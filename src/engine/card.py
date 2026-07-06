import random

CARD_CATALOG = [
    {"name": "Geralt", "power": 15, "row": "melee"},
    {"name": "Yennefer", "power": 7, "row": "ranged"},
    {"name": "Dandelion", "power": 2, "row": "melee"},
    {"name": "Trebuchet", "power": 6, "row": "siege"},
    {"name": "Redanian Knight", "power": 4, "row": "melee"},
    {"name": "Archer", "power": 5, "row": "ranged"},
    {"name": "Catapult", "power": 8, "row": "siege"},
    {"name": "Ciri", "power": 15, "row": "melee"},
    {"name": "Poor Fucking Infantry", "power": 1, "row": "melee"},
    {"name": "Vesemir", "power": 6, "row": "melee"},
    {"name": "Triss", "power": 7, "row": "melee"},
    {"name": "Philippa Eilhart", "power": 10, "row": "ranged"},
    {"name": "Thaler", "power": 1, "row": "siege"},
    {"name": "Roach", "power": 3, "row": "melee"},
    {"name": "Dethmold", "power": 6, "row": "ranged"},
    {"name": "Sheldon Skaggs", "power": 4, "row": "ranged"},
    {"name": "Keiza Metz", "power": 5, "row": "ranged"},
]

CARD_BY_NAME = {entry["name"]: i for i, entry in enumerate(CARD_CATALOG)}
NUM_CARD_TYPES = len(CARD_CATALOG)
PASS_ACTION = NUM_CARD_TYPES
DECK_SIZE = 10


class Card:
    def __init__(self, type_id: int):
        if type_id < 0 or type_id >= NUM_CARD_TYPES:
            raise ValueError(f"Invalid card type_id: {type_id}")

        stats = CARD_CATALOG[type_id]
        self.type_id = type_id
        self.name = stats["name"]
        self.row = stats["row"]
        self.base_power = stats["power"]
        self.current_power = stats["power"]

    def reset(self):
        """Palauttaa kortin voiman alkutilaan erän päättyessä."""
        self.current_power = self.base_power

    def __repr__(self):
        return f"{self.name} ({self.current_power} [Row: {self.row}])"


def hand_counts(hand: list[Card]) -> list[int]:
    counts = [0] * NUM_CARD_TYPES
    for card in hand:
        counts[card.type_id] += 1
    return counts


def create_random_deck():
    """Build a deck by sampling card types with replacement (duplicates allowed)."""
    type_ids = random.choices(range(NUM_CARD_TYPES), k=DECK_SIZE)
    return [Card(type_id) for type_id in type_ids]
