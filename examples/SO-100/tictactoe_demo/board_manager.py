import threading
import time
from collections import defaultdict
from collections.abc import Callable
from datetime import timedelta

from backend_client import set_board_state
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

    def __init__(self, get_latest_frame: Callable):
        # Logical board: 3x3 array, 'X', 'O', or None
        self.logical_board = [[None] * 3 for _ in range(3)]
        self.lock = threading.Lock()  # protects logical_board

        # Camera frame getter
        self.get_latest_frame = get_latest_frame

        # Position where robot is expected to move next
        self.next_robot_move = (None, None)

        # Thread control flags
        self.running = False

        # State tracking: 'human_turn', 'robot_turn', or 'analyzing'
        self.state = "human_turn"

    # ---------------- Thread Management ----------------
    def start(self):
        self.running = True
        self._perform_initial_scan()
        threading.Thread(target=self._monitor_loop, daemon=True).start()

    def stop(self):
        self.running = False

    # ---------------- Main Loop ----------------
    def _monitor_loop(self):
        time.sleep(3.0)  # initial delay to let system stabilize
        while self.running:
            state = self.get_state()
            if state == "human_turn":
                time.sleep(3.0)  # delay to allow camera thread to capture stable frame
                self._check_human_move()
                time.sleep(0.1)
            elif state == "robot_turn":
                self._check_robot_move(*self.next_robot_move)
                time.sleep(0.1)
            elif state == "analyzing":
                time.sleep(0.5)

    # ---------------- Move Detection ----------------
    def _perform_initial_scan(self):
        """Scans the board to initialize logical state."""
        time.sleep(3.0)  # allow system to stabilize
        print("🔍 Performing initial board scan...")
        n_scans = 70
        counts = defaultdict(lambda: defaultdict(int))
        start_time = time.time()
        for _ in range(n_scans):  # multiple passes for reliability
            for row in range(3):
                for col in range(3):
                    # print(f"Scanning cell ({row}, {col})...")
                    if self._detect_piece_in_cell(row, col, "O"):
                        value = "O"
                    elif self._detect_piece_in_cell(row, col, "X"):
                        value = "X"
                    else:
                        value = "empty"
                    counts[(row, col)][value] += 1

        # Classify each cell based on majority vote
        for row in range(3):
            for col in range(3):
                piece_type = max(counts[(row, col)], key=counts[(row, col)].get)
                self.update_board(
                    row, col, None if piece_type == "empty" else piece_type, print_update=False
                )
        end_time = time.time()
        scan_time = end_time - start_time
        print(f"⏱️ Initial scan took {timedelta(seconds=scan_time)}.")
        print("✅ Initial board scan complete.")
        self.print_board()

    def _check_human_move(self):
        """Detects a new piece placed by the human on empty cells."""
        value = "O"
        for row in range(3):
            for col in range(3):
                if self.logical_board[row][col] is None:
                    if self._detect_piece_in_cell(row, col, value):
                        print(f"👨 Human placed piece {value} at:", (row, col))
                        self.state = "analyzing"  # prepare to evaluate board, then robot's turn
                        self.update_board(row, col, value)
                        return

    def _check_robot_move(self, row: int, col: int):
        """Detects the robot's piece in the position (row, col) after a move is executed."""
        value = "X"
        if row is None or col is None:
            return  # no move pending

        # Give time for the robot to place the piece
        time.sleep(8.0)

        # Confirm piece placement with stability check
        stable_count = 0
        required_stability = 30  # number of consecutive frames
        n_prints = required_stability // 3
        while stable_count < required_stability:
            if self._detect_piece_in_cell(row, col, value):
                stable_count += 1
                if (stable_count % n_prints) == 0 or stable_count == required_stability:
                    print(f"⏳ Robot piece stability count: {stable_count}/{required_stability}")
            else:
                stable_count = 0
            time.sleep(0.1)  # check interval

        # Piece confirmed: update board
        self.next_robot_move = (None, None)
        print(f"🤖 Robot placed piece {value} at:", (row, col))
        self.state = "analyzing"  # stop checking, VLM will analyze next
        self.update_board(row, col, value)

    # ---------------- Utility Methods ----------------
    def _detect_piece_in_cell(self, row: int, col: int, piece_type: str) -> bool:
        img_cell = None
        while img_cell is None:
            frame = self.get_latest_frame()
            img_raw = frame.get("front")
            img_cell = extract_cell_image(img_raw, row, col)
        return (
            is_piece_in_cell(img_cell, min_occ=0.04, max_occ=0.09)
            and detect_piece_type(img_cell) == piece_type
        )

    def update_board(self, row: int, col: int, value: str, print_update: bool = True) -> None:
        """Thread-safe update of logical board and switch turn."""
        with self.lock:
            self.logical_board[row][col] = value
            if print_update:
                self.print_board()
            set_board_state(self.logical_board)

    def get_state(self) -> str:
        """Thread-safe getter for current state. Returns 'human_turn', 'robot_turn', or 'analyzing'."""
        with self.lock:
            return self.state

    def print_board(self) -> None:
        """Prints the current logical board state."""
        print("-" * 10 + "Current Board State:" + "-" * 10)
        for row in self.logical_board:
            print([" " if cell is None else cell for cell in row])
        print("-" * 40)
