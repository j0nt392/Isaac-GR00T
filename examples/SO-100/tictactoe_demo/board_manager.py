import threading
import time
from collections.abc import Callable

from utils import detect_piece_type, extract_cell_image, is_piece_in_cell

POSITION_TO_COORDS = {
    "top-left": (0, 0),
    "top-center": (0, 1),
    "top-right": (0, 2),
    "center-left": (1, 0),
    "center": (1, 1),
    "center-right": (1, 2),
    "bottom-left": (2, 0),
    "bottom-center": (2, 1),
    "bottom-right": (2, 2),
}


class BoardManager:
    """
    Thread-safe manager for Tic-Tac-Toe board state and turn management.
    Detects new piece placements and updates logical board.
    """

    def __init__(self, get_latest_frame: Callable, robot_move_event: threading.Event):
        # Logical board: 3x3 array, 'X', 'O', or None
        self.logical_board = [[None] * 3 for _ in range(3)]
        self.lock = threading.Lock()  # protects logical_board

        # Camera frame getter
        self.get_latest_frame = get_latest_frame

        # Event set when robot is performing a move
        self.robot_move_event = robot_move_event
        self.next_robot_move = (None, None)

        # Thread control flags
        self.running = False

        # Turn tracking: True=human, False=robot
        self.player_turn = True

    # ---------------- Thread Management ----------------
    def start(self):
        self.running = True
        threading.Thread(target=self._monitor_loop, daemon=True).start()

    def stop(self):
        self.running = False

    # ---------------- Main Loop ----------------
    def _monitor_loop(self):
        while self.running:
            if self.player_turn:
                self._check_human_move()
            else:
                self._check_robot_move(*self.next_robot_move)
            time.sleep(0.1)  # adjust polling interval

    # ---------------- Move Detection ----------------
    def _check_human_move(self):
        """Detects a new piece placed by the human on empty cells."""
        value = "O"
        for row in range(3):
            for col in range(3):
                if self.logical_board[row][col] is None:
                    if self._detect_piece_in_cell(row, col, value):
                        self.update_board(row, col, value)
                        return

    def _check_robot_move(self, row: int, col: int):
        """Detects the robot's piece in the position (row, col) after a move is executed."""
        if row is None or col is None:
            return  # no move pending

        # Wait until robot finishes motion (robot_move_event cleared)
        while self.robot_move_event.is_set():
            time.sleep(0.1)

        # Confirm piece placement with stability check
        stable_count = 0
        required_stability = 3  # number of consecutive frames
        while stable_count < required_stability:
            if self._detect_piece_in_cell(row, col, "X"):
                stable_count += 1
            else:
                stable_count = 0
            time.sleep(0.2)  # check interval

        # Piece confirmed: update board
        self.next_robot_move = (None, None)
        self.update_board(row, col, "X")

    # ---------------- Utility Methods ----------------
    def _detect_piece_in_cell(self, row: int, col: int, piece_type: str) -> bool:
        frame = self.get_latest_frame()
        img_raw = frame.get("front")
        img_cell = extract_cell_image(img_raw, row, col)
        return (
            is_piece_in_cell(img_cell, occupancy_thresh=0.02)
            and detect_piece_type(img_cell) == piece_type
        )

    def update_board(self, row: int, col: int, value: str):
        """Thread-safe update of logical board and switch turn."""
        with self.lock:
            self.logical_board[row][col] = value
            self.player_turn = not self.player_turn  # switch turn
