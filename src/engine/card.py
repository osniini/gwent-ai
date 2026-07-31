import random

CARD_CATALOG = [  # Defines the deck: each entry contributes `count` copies.
    {"name": "Geralt", "power": 15, "row": "melee", "hero": True, "count": 1},
    {"name": "Yennefer", "power": 7, "row": "ranged", "hero": True, "effect": "medic", "count": 1},
    {"name": "Dandelion", "power": 2, "row": "melee", "count": 1},
    {"name": "Trebuchet", "power": 6, "row": "siege", "count": 1},
    {"name": "Dun Banner Medic", "power": 3, "row": "siege", "effect": "medic", "count": 2},
    {"name": "Redanian Knight", "power": 4, "row": "melee", "count": 3},
    {"name": "Archer", "power": 5, "row": "ranged", "count": 3},
    {"name": "Catapult", "power": 8, "row": "siege", "effect": "tight_bond", "count": 3},
    {"name": "Blue Stripes Commando", "power": 4, "row": "melee", "effect": "tight_bond", "count": 3},
    {"name": "Ciri", "power": 15, "row": "melee", "hero": True, "count": 1},
    {"name": "Poor Fucking Infantry", "power": 1, "row": "melee", "effect": "tight_bond", "count": 2},
    {"name": "Vesemir", "power": 6, "row": "melee", "count": 1},
    {"name": "Kaedweni Siege Expert", "power": 1, "row": "siege", "effect": "morale_boost", "count": 3},
    {"name": "Triss", "power": 7, "row": "melee", "hero": True, "count": 1},
    {"name": "Philippa Eilhart", "power": 10, "row": "ranged", "hero": True, "count": 1},
    {"name": "Thaler", "power": 1, "row": "siege", "effect": "spy", "count": 1},
    {"name": "Mysterious Elf", "power": 0, "row": "melee", "effect": "spy", "count": 8},
    {"name": "Sigismund Dijkstra", "power": 4, "row": "melee", "effect": "spy", "count": 1},
    {"name": "Roach", "power": 3, "row": "melee", "count": 1},
    {"name": "Dethmold", "power": 6, "row": "ranged", "count": 1},
    {"name": "Sheldon Skaggs", "power": 4, "row": "ranged", "effect": "medic", "count": 1},
    {"name": "Gaunter O'Dimm: Darkness", "power": 4, "row": "ranged", "effect": "muster", "count": 8},
    {"name": "Keira Metz", "power": 5, "row": "ranged", "count": 1},
    {"name": "Biting Frost", "weather_row": "melee", "count": 2},
    {"name": "Impenetrable Fog", "weather_row": "ranged", "count": 2},
    {"name": "Torrential Rain", "weather_row": "siege", "count": 2},
    {"name": "Clear Weather", "weather_row": "clear", "count": 2},
    {"name": "Commander's Horn", "effect": "horn", "count": 3},
    {"name": "Decoy", "effect": "decoy", "count": 3},
]

CARD_BY_NAME = {entry["name"]: i for i, entry in enumerate(CARD_CATALOG)}
NUM_CARD_TYPES = len(CARD_CATALOG)
PASS_ACTION = NUM_CARD_TYPES


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
        self.effect = stats.get("effect")
        self.unit = stats.get("unit", self.row is not None)

    def reset(self):
        """Reset card power to its base value."""
        self.current_power = self.base_power

    def __repr__(self):
        if self.weather_row is not None:
            return f"{self.name} [weather: {self.weather_row}]"
        if self.effect is not None:
            return f"{self.name} [effect: {self.effect}]"
        hero = " hero" if self.hero else ""
        return f"{self.name} ({self.current_power} [Row: {self.row}]{hero})"


def hand_counts(hand: list[Card]) -> list[int]:
    counts = [0] * NUM_CARD_TYPES
    for card in hand:
        counts[card.type_id] += 1
    return counts


def create_deck() -> list[Card]:
    """Build and shuffle the complete deck declared by ``CARD_CATALOG``."""
    deck = []
    for type_id, card in enumerate(CARD_CATALOG):
        count = card.get("count", 1)
        if not isinstance(count, int) or count < 0:
            raise ValueError(f"Invalid count for {card['name']}: {count!r}")
        deck.extend(Card(type_id) for _ in range(count))
    random.shuffle(deck)
    return deck
