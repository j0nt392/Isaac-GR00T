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
    """
    Configuration for real-time Tic-Tac-Toe robot evaluation.

    This class organizes all key parameters controlling:
    - robot motion behavior (smoothing, RTC overlap, action refresh threshold)
    - perception (camera FPS, image enhancements)
    - UX/debugging options (saving/showing images)
    - reasoning configuration for the policy model

    These fields are read by TicTacToeBot and other sub-components to ensure
    consistent robot behavior and reproducibility.
    """

    # ---------------------------
    # 🟦 Real-Time Control Settings
    # ---------------------------
    action_smoothing_factor: float = 0.6
    """
    Exponential smoothing between consecutive actions.
    0.6 → 60% of the previous action, 40% of the new one.
    Higher values = smoother but more sluggish motion.
    """

    action_queue_threshold: float = 0.5
    """
    Fraction of the action horizon at which the controller triggers a new inference.
    Example: horizon=12, threshold=0.5 → request new chunk when ≤ 6 actions remain.
    """

    control_hz: int = 30
    """
    Frequency (Hz) of robot control updates.
    Determines `control_dt = 1/control_hz`.
    Higher = smoother but more CPU load.
    """

    rtc_overlap: int = 4
    """
    Number of timesteps blended between old and new action chunks during RTC merging.
    Prevents jumps when mid-horizon inference completes.
    """

    # ---------------------------
    # 🟩 Camera / Perception Settings
    # ---------------------------
    camera_fps: int = 30
    """Frames per second for board observation capture."""

    enhance_images: bool = False
    """Whether to apply preprocessing to improve board visibility."""

    save_images: bool = False
    """
    If True, store captured board images on disk under:
    Isaac-GR00T/board_images/
    """

    show_board_images: bool = False
    """
    If True, display board images in a popup window until the user closes it.
    Useful for debugging image preprocessing.
    """

    # ---------------------------
    # 🟨 Model Reasoning Settings
    # ---------------------------
    reasoning_effort: str = "low"
    """
    Controls how hard GPT/Gemini should think about the board state.
    Options could be: 'low', 'medium', 'high' depending on implementation.
    More effort → slower but more accurate reasoning.
    """
