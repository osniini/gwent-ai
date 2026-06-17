class Card:
    def __init__(self, name: str, power: int, row: str):
        self.name = name
        self.current_power = power
        self.row = row
        self.base_power = power

    def reset(self):
        """Palauttaa kortin voiman alkutilaan erän päättyessä."""
        self.current_power = self.base_power

    def __repr__(self):
        return f"{self.name} ({self.current_power} [Row: {self.row}])"


def create_starter_deck():
    return [
        Card("Geralt", 10, "melee"),
        Card("Yennefer", 7, "ranged"),
        Card("Dandelion", 2, "melee"),
        Card("Trebuchet", 6, "siege"),
        Card("Redanian Knight", 4, "melee"),
        Card("Archer", 5, "ranged"),
        Card("Catapult", 8, "siege"),
        Card("Ciri", 10, "melee"),
    ]