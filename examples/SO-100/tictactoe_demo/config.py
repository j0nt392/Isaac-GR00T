# Standard library
import os
from dataclasses import dataclass

# Third-party libraries
from dotenv import load_dotenv

load_dotenv()

# Local modules
from eval_lerobot import EvalConfig

BACKEND_URL = os.getenv("BACKEND_URL")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")
XAI_API_KEY = os.getenv("XAI_API_KEY")

EXPECTED_RESPONSE_FORMAT = {
    "observation": "<short text describing your observation of the board state in words (no ASCII art or such, just semantically), maximum 15 words>",
    "reasoning": "<short text describing your reasoning for the next move, maximum 15 words>",
    "action": "<your next move in the format described below, or 'N/A' if no moves remain>",
}
DEFAULT_MOVE = {
    "observation": "Could not parse board",
    "reasoning": "Error in GPT response",
    "action": "N/A",
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
    action_smoothing_factor: float = 0.7
    """
    Exponential smoothing between consecutive actions.
    E.g. 0.8 → 80% of the previous action, 20% of the new one.
    Higher values = smoother but more sluggish motion.
    """

    action_queue_threshold: float = 0.5
    """
    Fraction of the action horizon at which the controller triggers a new inference.
    Example: horizon=12, threshold=0.5 → request new chunk when ≤ 6 actions remain.
    """

    control_hz: int = 25
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
    # 🟨 Model Settings
    # ---------------------------
    vlm_model_name: str = "gemini-2.5-flash"
    """
    The model that will be used to analyze the board and decide the next move.
    Supported: 
        'gemini-2.5-flash'
        'gemini-2.5-pro'
        'grok-4'
        'grok-4-fast-non-reasoning'
        'gpt-5-nano'
        'o3'
    """

    reasoning_effort: str = "medium"
    """
    Controls how hard the model should think about the board state.
    Options are: 'low', 'medium', 'high', which are mapped to appropriate levels of reasoning specific to the selected model.
    More effort → slower but more accurate reasoning.
    """
