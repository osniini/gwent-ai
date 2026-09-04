import numpy as np
import customtkinter as ctk
from src.ai.agent import DQNAgent
from src.engine.card import NUM_CARD_TYPES
from src.engine.gwent_env import (
    DECOY_ACTIONS,
    DECOY_CARD_TYPE,
    GwentEnv,
    HORN_ACTIONS,
    HORN_CARD_TYPE,
    MEDIC_ACTIONS,
    MEDIC_CARD_TYPES,
    MEDIC_NO_TARGET_ACTIONS,
    PASS_ACTION,
    REDRAW_DONE_ACTION,
)
from src.gui.widgets import (
    BoardWidget,
    BOTTOM_ROW_ORDER,
    CardWidget,
    HandWidget,
    PlayerWidget,
    TOP_ROW_ORDER,
)

ctk.set_appearance_mode("dark")


class GwentApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.env = GwentEnv()
        self.env.reset()
        self.game_over = False
        self.round_paused = False
        self.pending_horn = False
        self.pending_decoy = False
        self.pending_medic_type: int | None = None
        self._ai_job = None
        self._round_continue_job = None
        self._discard_window = None

        self.agent = DQNAgent(self.env.state_size, self.env.action_size)
        try:
            self.agent.load("models/gwent_agent_gamma.pth")
        except Exception as exc:    
            print(f"Could not load model (retrain needed): {exc}")
        self.agent.epsilon = 0  # no random exploration in the GUI

        self.title("Gwent")
        self.geometry("1200x780")
        self.resizable(False, False)

        self.build_ui()
        self.bind("<Escape>", self._cancel_horn_selection)
        self.refresh()

    def refresh(self):
        p1_score = self.env.board.player1.get_total_score()
        p2_score = self.env.board.player2.get_total_score()
        if p1_score > p2_score:
            p1_leading, p2_leading = True, False
        elif p2_score > p1_score:
            p1_leading, p2_leading = False, True
        else:
            p1_leading = p2_leading = None

        self.opponent_player.update(
            self.env.board.player2,
            self.env.lives[1],
            hand_count=len(self.env.hand2),
            power_leading=p2_leading,
        )
        self.player_player.update(
            self.env.board.player1,
            self.env.lives[0],
            hand_count=len(self.env.hand1),
            power_leading=p1_leading,
        )
        can_act = (
            self.env.current_player == 1
            and not self.game_over
            and not self.round_paused
        )
        legal = self.env.get_legal_actions() if can_act else np.zeros(self.env.action_size, dtype=bool)
        redrawing = self.env.redraw_active and can_act
        selecting_horn = self.pending_horn and can_act
        selecting_decoy = self.pending_decoy and can_act
        selecting_medic = self.pending_medic_type is not None and can_act
        selectable_rows = {
            row for row, action in HORN_ACTIONS.items()
            if selecting_horn and legal[action]
        }
        selectable_card_types = {
            type_id for type_id, action in DECOY_ACTIONS.items()
            if selecting_decoy and legal[action]
        }
        selectable_discard_types = {
            target_type_id
            for (medic_type_id, target_type_id), action in MEDIC_ACTIONS.items()
            if (
                selecting_medic
                and medic_type_id == self.pending_medic_type
                and legal[action]
            )
        }
        playable_card_types = {
            type_id for type_id in range(NUM_CARD_TYPES)
            if legal[type_id]
        }
        if any(legal[action] for action in HORN_ACTIONS.values()):
            playable_card_types.add(HORN_CARD_TYPE)
        if any(legal[action] for action in DECOY_ACTIONS.values()):
            playable_card_types.add(DECOY_CARD_TYPE)
        for medic_type_id in MEDIC_CARD_TYPES:
            medic_actions = (
                action
                for (source_type_id, _), action in MEDIC_ACTIONS.items()
                if source_type_id == medic_type_id
            )
            if (
                any(legal[action] for action in medic_actions)
                or legal[MEDIC_NO_TARGET_ACTIONS[medic_type_id]]
            ):
                playable_card_types.add(medic_type_id)
        if selecting_horn or selecting_decoy or selecting_medic:
            playable_card_types = set()

        weather = self.env.board.weather_rows
        self.opponent_board.update(self.env.board.player2, weather)
        self.player_board.update(
            self.env.board.player1,
            weather,
            selectable_rows=selectable_rows,
            on_row_click=self.on_select_horn_row if selecting_horn else None,
            selectable_card_types=selectable_card_types,
            on_card_click=self.on_select_decoy_target if selecting_decoy else None,
        )
        self.horn_instruction.configure(
            text=(
                f"Redraw up to 2 cards, then press Done."
                if redrawing
                else "Select a highlighted row for Commander's Horn (Esc to cancel)."
                if selecting_horn
                else "Select one of your units to replace with Decoy (Esc to cancel)."
                if selecting_decoy
                else "Select a unit from your discard pile to revive (Esc to cancel)."
                if selecting_medic
                else ""
            )
        )

        self._refresh_discard_window()
        displayed_cards = self.env.discard1 if selecting_medic else self.env.hand1
        self.player_hand.update(
            displayed_cards,
            on_click=(
                self.on_select_medic_target
                if selecting_medic
                else self.on_play_card
                if can_act and not selecting_horn and not selecting_decoy
                else None
            ),
            legal=legal,
            on_pass=(
                self.on_pass
                if can_act and not selecting_horn and not selecting_decoy and not selecting_medic
                else None
            ),
            on_view_discard=self.show_discard_piles,
            playable_card_types=selectable_discard_types if selecting_medic else playable_card_types,
            pass_action=REDRAW_DONE_ACTION if redrawing else PASS_ACTION,
            pass_text="Done" if redrawing else "Pass",
        )
        self.player_hand.label.configure(text="Discard" if selecting_medic else "Hand")

        if self.env.current_player == 2 and not self.game_over and not self.round_paused:
            self._schedule_ai_turn()

    def _schedule_ai_turn(self):
        if self._ai_job is not None:
            self.after_cancel(self._ai_job)
        self._ai_job = self.after(400, self.ai_turn)

    def ai_turn(self):
        self._ai_job = None

        if self.game_over or self.round_paused or self.env.current_player != 2:
            return

        legal = self.env.get_legal_actions()
        if not legal.any():
            return

        state = self.env._get_state()
        action = self.agent.select_action(state, legal)
        self._apply_action(action)

    def _apply_action(self, action: int):
        prev_lives = list(self.env.lives)
        _, _, done = self.env.step(action)
        self._on_step_complete(prev_lives, done)
        self.refresh()

    def _on_step_complete(self, prev_lives: list[int], done: bool):
        # Lives always change when a round ends (including ties); do not use
        # last_round_was_tie here — it stays True until the next round ends.
        round_changed = self.env.lives != prev_lives

        if done:
            self.game_over = True
            self._show_overlay(self._game_over_message(), show_new_game=True)
        elif round_changed:
            self.round_paused = True
            self._show_overlay(self._round_end_message(prev_lives), show_new_game=False)
            if self._round_continue_job is not None:
                self.after_cancel(self._round_continue_job)
            self._round_continue_job = self.after(2000, self._continue_round)

    def _round_end_message(self, prev_lives: list[int]) -> str:
        if self.env.match_draw:
            return "Round tied at last life — match drawn."
        if self.env.last_round_was_tie:
            return "Round tied — both players lose a life."
        if self.env.lives[0] < prev_lives[0]:
            return "You lost the round!"
        return "You won the round!"

    def _game_over_message(self) -> str:
        if self.env.match_draw:
            return "Match drawn."
        if self.env.lives[0] == 0 and self.env.lives[1] > 0:
            return "Opponent wins the match!"
        if self.env.lives[1] == 0 and self.env.lives[0] > 0:
            return "You win the match!"
        if self.env.lives[0] == 0 and self.env.lives[1] == 0:
            return "Match drawn."
        return "Match over."

    def _show_overlay(self, message: str, *, show_new_game: bool):
        self.overlay_message.configure(text=message)
        if show_new_game:
            self.continue_btn.pack_forget()
            self.new_game_btn.pack(pady=(8, 0))
        else:
            self.new_game_btn.pack_forget()
            self.continue_btn.pack(pady=(8, 0))
        self.overlay.place(relx=0.5, rely=0.5, anchor="center")

    def _hide_overlay(self):
        self.overlay.place_forget()

    def _continue_round(self):
        if self._round_continue_job is not None:
            self.after_cancel(self._round_continue_job)
            self._round_continue_job = None
        if not self.round_paused:
            return
        self.round_paused = False
        self._hide_overlay()
        self.refresh()

    def new_game(self):
        if self._ai_job is not None:
            self.after_cancel(self._ai_job)
            self._ai_job = None
        if self._round_continue_job is not None:
            self.after_cancel(self._round_continue_job)
            self._round_continue_job = None

        self.env.reset()
        self.game_over = False
        self.round_paused = False
        self.pending_horn = False
        self.pending_decoy = False
        self.pending_medic_type = None
        self._hide_overlay()
        self.refresh()

    def build_ui(self):
        container = ctk.CTkFrame(self, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=12, pady=10)

        # Opponent
        opponent_row = ctk.CTkFrame(container, fg_color="transparent")
        opponent_row.pack(fill="x", pady=(0, 4))
        self.opponent_player = PlayerWidget(opponent_row, "Opponent")
        self.opponent_player.pack(side="left", padx=(0, 4))
        self.opponent_board = BoardWidget(opponent_row, TOP_ROW_ORDER)
        self.opponent_board.pack(side="left", fill="both", expand=True)

        ctk.CTkFrame(container, height=1, fg_color="#666666").pack(fill="x", pady=2)

        # Player
        player_row = ctk.CTkFrame(container, fg_color="transparent")
        player_row.pack(fill="x", pady=(4, 0))
        self.player_player = PlayerWidget(player_row, "You")
        self.player_player.pack(side="left", padx=(0, 4))
        self.player_board = BoardWidget(player_row, BOTTOM_ROW_ORDER)
        self.player_board.pack(side="left", fill="both", expand=True)

        # Player hand
        self.horn_instruction = ctk.CTkLabel(
            container,
            text="",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#d8a137",
        )
        self.horn_instruction.pack(fill="x", pady=(8, 0))
        self.player_hand = HandWidget(container)
        self.player_hand.pack(fill="x", pady=(4, 0))

        self.overlay = ctk.CTkFrame(self, fg_color="#1a1a1a", corner_radius=8, border_width=1, border_color="#666666")
        self.overlay_message = ctk.CTkLabel(
            self.overlay,
            text="",
            font=ctk.CTkFont(size=18, weight="bold"),
            wraplength=360,
            justify="center",
        )
        self.overlay_message.pack(padx=24, pady=(20, 8))

        self.continue_btn = ctk.CTkButton(self.overlay, text="Continue", width=120, command=self._continue_round)
        self.new_game_btn = ctk.CTkButton(self.overlay, text="New Game", width=120, command=self.new_game)

    def on_play_card(self, type_id: int):
        if (
            self.game_over
            or self.round_paused
            or self.pending_horn
            or self.pending_decoy
            or self.pending_medic_type is not None
            or self.env.current_player != 1
        ):
            return

        legal = self.env.get_legal_actions()
        if self.env.redraw_active:
            if type_id < len(legal) and legal[type_id]:
                self._apply_action(type_id)
            return
        if type_id == HORN_CARD_TYPE:
            if any(legal[action] for action in HORN_ACTIONS.values()):
                self.pending_horn = True
                self.refresh()
            return
        if type_id == DECOY_CARD_TYPE:
            if any(legal[action] for action in DECOY_ACTIONS.values()):
                self.pending_decoy = True
                self.refresh()
            return
        if type_id in MEDIC_CARD_TYPES:
            target_actions = (
                action
                for (medic_type_id, _), action in MEDIC_ACTIONS.items()
                if medic_type_id == type_id
            )
            if any(legal[action] for action in target_actions):
                self.pending_medic_type = type_id
                self.refresh()
            elif legal[MEDIC_NO_TARGET_ACTIONS[type_id]]:
                self._apply_action(MEDIC_NO_TARGET_ACTIONS[type_id])
            return
        if type_id >= len(legal) or not legal[type_id]:
            return

        self._apply_action(type_id)

    def on_select_horn_row(self, row: str):
        if not self.pending_horn:
            return
        action = HORN_ACTIONS[row]
        legal = self.env.get_legal_actions()
        if not legal[action]:
            return
        self.pending_horn = False
        self._apply_action(action)

    def on_select_decoy_target(self, type_id: int):
        if not self.pending_decoy:
            return
        action = DECOY_ACTIONS.get(type_id)
        legal = self.env.get_legal_actions()
        if action is None or not legal[action]:
            return
        self.pending_decoy = False
        self._apply_action(action)

    def on_select_medic_target(self, type_id: int):
        if self.pending_medic_type is None:
            return
        action = MEDIC_ACTIONS.get((self.pending_medic_type, type_id))
        legal = self.env.get_legal_actions()
        if action is None or not legal[action]:
            return
        self.pending_medic_type = None
        self._apply_action(action)

    def _cancel_horn_selection(self, _event=None):
        if self.pending_horn or self.pending_decoy or self.pending_medic_type is not None:
            self.pending_horn = False
            self.pending_decoy = False
            self.pending_medic_type = None
            self.refresh()

    def on_pass(self):
        if self.pending_horn or self.pending_decoy or self.pending_medic_type is not None:
            return
        action = self.env.redraw_done_action if self.env.redraw_active else self.env.pass_action
        self._apply_action(action)

    def show_discard_piles(self):
        if self._discard_window is None or not self._discard_window.winfo_exists():
            self._discard_window = ctk.CTkToplevel(self)
            self._discard_window.title("Discard piles")
            self._discard_window.geometry("780x390")
            self._discard_window.resizable(False, False)
            self._discard_window.protocol(
                "WM_DELETE_WINDOW",
                lambda: self._discard_window.destroy(),
            )
        self._discard_window.deiconify()
        self._discard_window.lift()
        self._refresh_discard_window()

    def _refresh_discard_window(self):
        if self._discard_window is None or not self._discard_window.winfo_exists():
            return
        for child in self._discard_window.winfo_children():
            child.destroy()

        for column, (title, cards) in enumerate((
            ("Your discard", self.env.discard1),
            ("Opponent discard", self.env.discard2),
        )):
            pile = ctk.CTkFrame(self._discard_window, fg_color="transparent")
            pile.grid(row=0, column=column, sticky="nsew", padx=12, pady=12)
            ctk.CTkLabel(
                pile,
                text=f"{title} ({len(cards)})",
                font=ctk.CTkFont(size=14, weight="bold"),
            ).pack(pady=(0, 8))
            cards_frame = ctk.CTkFrame(pile, fg_color="#1a1a1a", corner_radius=4)
            cards_frame.pack(fill="both", expand=True)
            if not cards:
                ctk.CTkLabel(cards_frame, text="Empty", text_color="#888888").pack(
                    padx=120,
                    pady=120,
                )
                continue
            for index, card in enumerate(cards):
                CardWidget(cards_frame, card).grid(
                    row=index // 4,
                    column=index % 4,
                    padx=4,
                    pady=4,
                )

        self._discard_window.grid_columnconfigure((0, 1), weight=1)
        self._discard_window.grid_rowconfigure(0, weight=1)


if __name__ == "__main__":
    app = GwentApp()
    app.mainloop()
