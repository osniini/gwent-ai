import random

CARD_CATALOG = [
    {"name": "Geralt", "power": 15, "row": "melee", "hero": True, "unique": True},
    {"name": "Yennefer", "power": 7, "row": "ranged", "hero": True, "unique": True},
    {"name": "Dandelion", "power": 2, "row": "melee", "unique": True},
    {"name": "Trebuchet", "power": 6, "row": "siege"},
    {"name": "Redanian Knight", "power": 4, "row": "melee"},
    {"name": "Archer", "power": 5, "row": "ranged"},
    {"name": "Catapult", "power": 8, "row": "siege"},
    {"name": "Ciri", "power": 15, "row": "melee", "hero": True, "unique": True},
    {"name": "Poor Fucking Infantry", "power": 1, "row": "melee"},
    {"name": "Vesemir", "power": 6, "row": "melee", "unique": True},
    {"name": "Triss", "power": 7, "row": "melee", "hero": True, "unique": True},
    {"name": "Philippa Eilhart", "power": 10, "row": "ranged", "hero": True, "unique": True},
    {"name": "Thaler", "power": 1, "row": "siege", "unique": True},
    {"name": "Roach", "power": 3, "row": "melee", "unique": True},
    {"name": "Dethmold", "power": 6, "row": "ranged", "unique": True},
    {"name": "Sheldon Skaggs", "power": 4, "row": "ranged", "unique": True},
    {"name": "Keira Metz", "power": 5, "row": "ranged", "unique": True},
    {"name": "Biting Frost", "weather_row": "melee"},
    {"name": "Impenetrable Fog", "weather_row": "ranged"},
    {"name": "Torrential Rain", "weather_row": "siege"},
    {"name": "Clear Weather", "weather_row": "clear"},
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
        self.row = stats.get("row")
        self.base_power = stats.get("power", 0)
        self.current_power = self.base_power
        self.weather_row = stats.get("weather_row")
        self.hero = stats.get("hero", False)
        self.unique = stats.get("unique", False)

    def reset(self):
        """Reset card power to its base value."""
        self.current_power = self.base_power

    def __repr__(self):
        if self.weather_row is not None:
            return f"{self.name} [weather: {self.weather_row}]"
        hero = " hero" if self.hero else ""
        return f"{self.name} ({self.current_power} [Row: {self.row}]{hero})"


def hand_counts(hand: list[Card]) -> list[int]:
    counts = [0] * NUM_CARD_TYPES
    for card in hand:
        counts[card.type_id] += 1
    return counts


def create_random_deck():
    """Build a deck; unique cards appear at most once, others may duplicate."""
    available = list(range(NUM_CARD_TYPES))
    type_ids = []
    for _ in range(DECK_SIZE):
        type_id = random.choice(available)
        type_ids.append(type_id)
        if CARD_CATALOG[type_id].get("unique", False):
            available.remove(type_id)
    return [Card(type_id) for type_id in type_ids]
