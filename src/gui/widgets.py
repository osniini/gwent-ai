from collections.abc import Callable

import customtkinter as ctk
from src.engine.card import Card
from src.engine.board import PlayerBoard


ROW_COLORS = {
    "melee": "#5c3a1e",
    "ranged": "#1e4d3a",
    "siege": "#1e3a5c",
}

ROW_LABELS = {
    "melee": "Melee",
    "ranged": "Ranged",
    "siege": "Siege",
}

WEATHER_CARD_COLOR = "#3d4f66"
WEATHER_ROW_COLOR = "#9a9a9a"
WEATHER_ACTIVE_COLOR = "#7eb8da"
HORN_COLOR = "#d8a137"
ROW_SELECTION_COLOR = "#4ade80"
HERO_COLOR = "#6b5420"
HERO_BORDER = "#c9a227"

WEATHER_LABELS = {
    "melee": "Frost",
    "ranged": "Fog",
    "siege": "Rain",
    "clear": "Clear",
}

# Bottom player: melee nearest the center line, siege farthest.
BOTTOM_ROW_ORDER = ("melee", "ranged", "siege")
# Top player: mirrored
TOP_ROW_ORDER = ("siege", "ranged", "melee")


ROW_HEIGHT = 100
CARD_WIDTH = 72
CARD_HEIGHT = ROW_HEIGHT - 12
PLAYER_PANEL_WIDTH = 88
BOARD_HEIGHT = 3 * ROW_HEIGHT + 4
LIVES_PER_PLAYER = 2
POWER_COLOR_LEADING = "#4ade80"
POWER_COLOR_DEFAULT = ("#ebebeb", "#ebebeb")
POWER_COLOR_BEHIND = "#666666"


class CardWidget(ctk.CTkFrame):
    def __init__(
        self,
        master,
        card: Card,
        *,
        command: Callable[[], None] | None = None,
        enabled: bool = True,
    ):
        is_weather = card.weather_row is not None
        if is_weather:
            fg_color = WEATHER_CARD_COLOR
            border_width = 0
            border_color = None
        elif card.hero:
            fg_color = ROW_COLORS.get(card.row, "#333333")
            border_width = 2
            border_color = HERO_BORDER
        else:
            fg_color = ROW_COLORS.get(card.row, "#333333")
            border_width = 0
            border_color = None

        if not enabled:
            fg_color = "#3a3a3a"
            border_color = "#555555" if card.hero else border_color

        kwargs = {
            "fg_color": fg_color,
            "corner_radius": 4,
            "width": CARD_WIDTH,
            "height": CARD_HEIGHT,
            "border_width": border_width,
        }
        if border_color is not None:
            kwargs["border_color"] = border_color

        super().__init__(master, **kwargs)
        self.pack_propagate(False)

        muted = "#888888" if not enabled else None
        ctk.CTkLabel(
            self,
            text=card.name,
            font=ctk.CTkFont(size=10, weight="bold"),
            wraplength=CARD_WIDTH - 8,
            text_color=muted,
        ).pack(padx=4, pady=(6, 0))

        if is_weather:
            power_text = WEATHER_LABELS.get(card.weather_row, "W")
        elif card.effect == "horn":
            power_text = "x2"
        elif card.effect == "tight_bond":
            power_text = f"TB · {card.current_power}"
        elif card.effect == "morale_boost":
            power_text = f"MB · {card.current_power}"
        elif card.effect == "decoy":
            power_text = "⇄"
        else:
            power_text = str(card.current_power)

        ctk.CTkLabel(
            self,
            text=power_text,
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=muted if muted else (WEATHER_ACTIVE_COLOR if is_weather else None),
        ).pack(pady=(2, 6))

        if command and enabled:
            self._bind_click(command)

    def _bind_click(self, command: Callable[[], None]) -> None:
        def on_click(_event):
            command()

        def bind_recursive(widget):
            widget.bind("<Button-1>", on_click)
            widget.configure(cursor="hand2")
            for child in widget.winfo_children():
                bind_recursive(child)

        bind_recursive(self)


class RowWidget(ctk.CTkFrame):
    def __init__(self, master, row_name: str):
        super().__init__(master, fg_color=ROW_COLORS[row_name], corner_radius=4, height=ROW_HEIGHT)
        self.pack_propagate(False)

        self.label = ctk.CTkLabel(
            self,
            text=ROW_LABELS[row_name],
            width=52,
            height=ROW_HEIGHT - 4,
            font=ctk.CTkFont(size=11, weight="bold"),
        )
        self.label.pack(side="left", padx=(4, 2), pady=2)

        self.slots = ctk.CTkFrame(self, fg_color="#1a1a1a", corner_radius=3, height=ROW_HEIGHT - 6)
        self.slots.pack(side="left", fill="x", expand=True, padx=(0, 4), pady=3)
        self.slots.pack_propagate(False)

        self.score_label = ctk.CTkLabel(
            self,
            text="0",
            width=36,
            height=ROW_HEIGHT - 4,
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        self.score_label.pack(side="right", padx=(2, 4), pady=2)

        self.horn_label = ctk.CTkLabel(
            self,
            text="",
            width=42,
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=HORN_COLOR,
        )
        self.horn_label.pack(side="right", padx=(2, 0), pady=2)

        self.row_name = row_name

    def update(
        self,
        board: PlayerBoard,
        row_name: str,
        *,
        weather_active: bool = False,
        selectable: bool = False,
        on_click: Callable[[str], None] | None = None,
        selectable_card_types: set[int] | None = None,
        on_card_click: Callable[[int], None] | None = None,
    ):
        for child in self.slots.winfo_children():
            child.destroy()

        for card in board.rows[row_name]:
            card_is_selectable = (
                selectable_card_types is not None
                and card.type_id in selectable_card_types
            )
            command = None
            if selectable and on_click is not None:
                command = lambda: on_click(row_name)
            elif card_is_selectable and on_card_click is not None:
                command = lambda type_id=card.type_id: on_card_click(type_id)
            CardWidget(self.slots, card, command=command).pack(side="left", padx=3, pady=2)

        self.score_label.configure(text=str(board.get_row_score(row_name)))
        self.horn_label.configure(text="HORN" if board.horn_rows[row_name] else "")

        if selectable:
            border_color = ROW_SELECTION_COLOR
        elif weather_active:
            border_color = WEATHER_ROW_COLOR
        elif board.horn_rows[row_name]:
            border_color = HORN_COLOR
        else:
            border_color = None

        if border_color:
            self.configure(border_width=2, border_color=border_color)
        else:
            self.configure(border_width=0)
        label_color = WEATHER_ROW_COLOR if weather_active else ["#1a1a1a", "#ebebeb"]
        self.label.configure(text_color=label_color)
        self.score_label.configure(text_color=label_color)
        self._set_click_handler(on_click if selectable else None)

    def _set_click_handler(self, on_click: Callable[[str], None] | None) -> None:
        def bind_widget(widget):
            if on_click is None:
                widget.unbind("<Button-1>")
                widget.configure(cursor="")
            else:
                widget.bind("<Button-1>", lambda _event: on_click(self.row_name))
                widget.configure(cursor="hand2")

        # Card widgets manage their own commands. Binding recursively here
        # would remove the Decoy target handlers created during update().
        for widget in (self, self.label, self.slots, self.score_label, self.horn_label):
            bind_widget(widget)


class PlayerWidget(ctk.CTkFrame):
    def __init__(self, master, name: str):
        super().__init__(
            master,
            fg_color="#2a2a2a",
            corner_radius=4,
            width=PLAYER_PANEL_WIDTH,
            height=BOARD_HEIGHT,
        )
        self.pack_propagate(False)

        ctk.CTkLabel(
            self,
            text=name,
            font=ctk.CTkFont(size=11, weight="bold"),
            wraplength=PLAYER_PANEL_WIDTH - 8,
        ).pack(pady=(10, 6))

        ctk.CTkLabel(
            self,
            text="Power",
            font=ctk.CTkFont(size=10),
            text_color="#888888",
        ).pack()
        self.power_label = ctk.CTkLabel(
            self,
            text="0",
            font=ctk.CTkFont(size=22, weight="bold"),
        )
        self.power_label.pack(pady=(0, 4))

        ctk.CTkLabel(
            self,
            text="Cards",
            font=ctk.CTkFont(size=10),
            text_color="#888888",
        ).pack()
        self.cards_label = ctk.CTkLabel(
            self,
            text="0",
            font=ctk.CTkFont(size=16, weight="bold"),
        )
        self.cards_label.pack(pady=(0, 6))

        ctk.CTkLabel(
            self,
            text="Match",
            font=ctk.CTkFont(size=10),
            text_color="#888888",
        ).pack()
        ctk.CTkLabel(
            self,
            text=f"{LIVES_PER_PLAYER} lives",
            font=ctk.CTkFont(size=9),
            text_color="#666666",
        ).pack(pady=(0, 4))

        self.round_pips_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.round_pips_frame.pack(pady=(0, 8))
        self.round_pips: list[ctk.CTkLabel] = []
        for _ in range(LIVES_PER_PLAYER):
            pip = ctk.CTkLabel(
                self.round_pips_frame,
                text="○",
                font=ctk.CTkFont(size=18),
                width=22,
                text_color="#555555",
            )
            pip.pack(side="left")
            self.round_pips.append(pip)

        self.pass_label = ctk.CTkLabel(
            self,
            text="",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#e0a040",
        )
        self.pass_label.pack()

    def update(
        self,
        board: PlayerBoard,
        lives: int,
        *,
        hand_count: int = 0,
        power_leading: bool | None = None,
    ):
        power = board.get_total_score()
        self.power_label.configure(text=str(power))
        self.cards_label.configure(text=str(hand_count))

        if power_leading is True:
            self.power_label.configure(text_color=POWER_COLOR_LEADING)
        elif power_leading is False:
            self.power_label.configure(text_color=POWER_COLOR_BEHIND)
        else:
            self.power_label.configure(text_color=POWER_COLOR_DEFAULT)

        for i, pip in enumerate(self.round_pips):
            alive = i < lives
            pip.configure(
                text="●" if alive else "○",
                text_color=POWER_COLOR_LEADING if alive else "#555555",
            )

        self.pass_label.configure(text="Passed" if board.passed else "")


class BoardWidget(ctk.CTkFrame):
    def __init__(self, master, row_order: tuple[str, ...]):
        super().__init__(master, fg_color="transparent")

        self.row_order = row_order
        self.row_widgets = {}

        for name in row_order:
            row = RowWidget(self, name)
            row.pack(fill="x", pady=1)
            self.row_widgets[name] = row

    def update(
        self,
        board: PlayerBoard,
        weather_rows: dict[str, bool] | None = None,
        *,
        selectable_rows: set[str] | None = None,
        on_row_click: Callable[[str], None] | None = None,
        selectable_card_types: set[int] | None = None,
        on_card_click: Callable[[int], None] | None = None,
    ):
        weather_rows = weather_rows or {}
        selectable_rows = selectable_rows or set()
        for name in self.row_order:
            self.row_widgets[name].update(
                board,
                name,
                weather_active=weather_rows.get(name, False),
                selectable=name in selectable_rows,
                on_click=on_row_click,
                selectable_card_types=selectable_card_types,
                on_card_click=on_card_click,
            )


class HandWidget(ctk.CTkFrame):
    def __init__(self, master, title: str = "Hand"):
        super().__init__(master, fg_color="#2a2a2a", corner_radius=4, height=ROW_HEIGHT)
        self.pack_propagate(False)

        self.label = ctk.CTkLabel(
            self,
            text=title,
            width=52,
            height=ROW_HEIGHT - 4,
            font=ctk.CTkFont(size=11, weight="bold"),
        )
        self.label.pack(side="left", padx=(4, 2), pady=2)

        self.slots = ctk.CTkFrame(self, fg_color="#1a1a1a", corner_radius=3, height=ROW_HEIGHT - 6)
        self.slots.pack(side="left", fill="x", expand=True, padx=(0, 4), pady=3)
        self.slots.pack_propagate(False)

        self.controls = ctk.CTkFrame(
            self,
            fg_color="transparent",
            width=70,
            height=ROW_HEIGHT - 6,
        )
        self.controls.pack(side="right", padx=(0, 4), pady=3)
        self.controls.pack_propagate(False)

        self.pass_btn = ctk.CTkButton(
            self.controls,
            text="Pass",
            width=70,
            height=58,
            fg_color="#101010",
            hover_color="#202020",
            text_color="#d8a137",
        )
        self.pass_btn.pack(fill="x", pady=(0, 2))
        self.discard_btn = ctk.CTkButton(
            self.controls,
            text="Discard",
            width=70,
            height=30,
            fg_color="#555555",
            hover_color="#666666",
            text_color="#ffffff",
        )
        self.discard_btn.pack(fill="x")

    def update(
        self,
        cards: list[Card],
        on_click: Callable[[int], None] | None = None,
        legal=None,
        on_pass: Callable[[], None] | None = None,
        on_view_discard: Callable[[], None] | None = None,
        playable_card_types: set[int] | None = None,
        pass_action: int | None = None,
        pass_text: str = "Pass",
    ):
        for child in self.slots.winfo_children():
            child.destroy()

        for card in cards:
            if playable_card_types is not None:
                is_legal = card.type_id in playable_card_types
            else:
                is_legal = (
                    bool(legal[card.type_id])
                    if legal is not None and card.type_id < len(legal)
                    else True
                )
            command = (
                (lambda type_id=card.type_id: on_click(type_id))
                if on_click and is_legal
                else None
            )
            CardWidget(self.slots, card, command=command, enabled=is_legal).pack(
                side="left", padx=3, pady=2
            )

        can_pass = (
            bool(legal[pass_action]) if pass_action is not None
            else bool(legal[-1])
            if legal is not None and len(legal) > 0
            else True
        )
        self.pass_btn.configure(
            text=pass_text,
            command=on_pass if on_pass and can_pass else None,
            state="normal" if on_pass and can_pass else "disabled",
        )
        self.discard_btn.configure(
            command=on_view_discard,
            state="normal" if on_view_discard else "disabled",
        )
