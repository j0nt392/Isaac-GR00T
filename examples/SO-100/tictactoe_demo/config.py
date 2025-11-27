# Standard library
import os
from dataclasses import dataclass

# Third-party libraries
from dotenv import load_dotenv

load_dotenv()

# Local modules
from eval_lerobot import EvalConfig

BACKEND_URL = os.getenv("BACKEND_URL")
OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")

EXPECTED_RESPONSE_FORMAT = {
    "observation": "<short text describing your observation of the board state>",
    "reasoning": "<short text describing your reasoning for the next move>",
    "action": "<your next move in the format described below, or 'N/A' if no moves remain>",
    "game_state": "<one of ['win', 'draw', 'loss', 'ongoing'] representing the state of the game from X’s perspective>",
}
DEFAULT_MOVE = {
    "observation": "Could not parse board",
    "reasoning": "Error in GPT response",
    "action": "N/A",
    "game_state": "Unknown",
}


@dataclass
class TicTacToeConfig(EvalConfig):
    action_smoothing_factor: float = 0.6  # 60% old action, 40% new action (tune as needed)
    camera_fps: int = 30
    control_hz: int = 30  # speed of the robot's movements
    enhance_images: bool = False  # whether to apply image enhancement preprocessing
    save_images: bool = (
        False  # whether to save captured board images to disk (Isaac-GR00T/board_images/)
    )
    show_board_images: bool = (
        False  # whether to show preprocessed board images until user closes the window
    )
    reasoning_effort: str = "low"  # amount of effort GPT should use when analyzing the game
