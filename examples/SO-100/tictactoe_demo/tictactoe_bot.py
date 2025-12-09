# Standard library
import json
import threading
import time
from queue import Queue

# Local modules
from backend_client import get_player_turn, send_reasoning, set_player_turn
from camera import CameraSystem
from config import TicTacToeConfig
from debug_display import DebugDisplay
from eval_lerobot import Gr00tRobotInferenceClient
from lerobot.robots import make_robot_from_config  # noqa: F401
from rtc_controller import RTCMotionController
from utils import prepare_frame_for_vlm, print_blue, print_green, print_yellow
from vlm_client import VLMClient

GAME_RESULT = {
    "win": " 🤖 Bot has defeated 👨 Player!",
    "loss": " 👨 Player has defeated 🤖 Bot!",
    "draw": "It's a draw 🤖-👨!",
}


class TicTacToeBot:
    """
    High-level controller for the Tic-Tac-Toe robot system.

    This class ties together:
    - robot hardware
    - camera system
    - vision-language model and policy inference clients
    - real-time control components (RTC)
    - game state and turn synchronization

    It does not execute motion itself; motion is handled by RTCMotionController.
    This class acts as the orchestrator for gameplay.
    """

    def __init__(self, cfg: TicTacToeConfig):
        # Configuration and game state
        self.cfg = cfg
        self.game_state = "ongoing"  # one of: ["win", "loss", "draw", "ongoing"]

        # Vision-language client (board analysis and reasoning)
        self.vlm_client = VLMClient()

        # Robot hardware and synchronization
        self.robot = make_robot_from_config(cfg.robot)
        self.robot_lock = threading.Lock()  # prevents concurrent robot access

        # Policy/action inference client (GR00T)
        self.client = Gr00tRobotInferenceClient(
            host=cfg.policy_host,
            port=cfg.policy_port,
            camera_keys=list(cfg.robot.cameras.keys()),
            robot_state_keys=list(self.robot._motors_ft.keys()),
        )

        # Camera subsystem (captures images in sync with robot motion for interface)
        self.camera_system = CameraSystem(self.robot, self.robot_lock, cfg.camera_fps)

        # Shared action queue (used by inference and control loops)
        self.action_queue: Queue[dict[str, float]] = Queue(maxsize=cfg.action_horizon)

        # Optional debugging interface (visualizations, diagnostics)
        self.debug_display = DebugDisplay() if cfg.show_board_images else None

    # ------------------ Game State Helpers ------------------
    def wait_until_player_turn(self, expected: bool) -> None:
        while get_player_turn() != expected:
            time.sleep(0.2)

    def handle_pause(self) -> None:
        print_yellow(" 👨 Player's turn! Make a move, then press SPACE to resume.")
        while not self.action_queue.empty():
            self.action_queue.get_nowait()  # clear action queue before robot's next turn

    def handle_resume(self) -> None:
        """Robot's turn: VLM predicts move; then RTC control executes it."""
        print_green(" 🤖 Bot's turn! Thinking...")

        # --- STEP 1: PRE-MOVE ANALYSIS ---
        # Retrieve latest camera frame for VLM inference
        obs = self.camera_system.get_latest_obs()
        img = prepare_frame_for_vlm(obs, self.cfg, self.debug_display)

        # VLM decision (move or game over)
        move_dict_1 = self.vlm_client.get_move_decision(img, self.cfg.reasoning_effort)
        send_reasoning(move_dict_1)
        print_green(f" 🤖 Bot's decision (step 1): {json.dumps(move_dict_1, indent=4)}")

        state, action = move_dict_1["game_state"], move_dict_1["action"]
        self.game_state = state
        # Check if game is already over (win, loss, or draw)
        if state != "ongoing" and action == "N/A":
            return

        # Execute move. Perform the action steps using real-time chunking
        self._start_rtc_for_move(action)

        # System busy point: give the VLM time to analyze the game post-move before player's turn -
        # set_processing_status(True)

        # --- STEP 2: POST-MOVE ANALYSIS ---
        # Retrieve camera frame after move
        obs_post = self.camera_system.get_latest_obs()
        print_green("Sending second frame to vlm")
        img_post = prepare_frame_for_vlm(obs_post, self.cfg, self.debug_display)

        # Get the game state after the move
        move_dict_2 = self.vlm_client.get_post_move_state(img_post, self.cfg.reasoning_effort)
        send_reasoning(move_dict_2)
        print_green(f" 🤖 Bot's post-move analysis (step 2): {json.dumps(move_dict_2, indent=4)}")
        # Update the game state and check if game is over
        self.game_state = move_dict_2.get("game_state", "ongoing")
        if self.game_state != "ongoing":
            return

        # End the bot's turn
        set_player_turn(True)

    def _start_rtc_for_move(self, language_instruction: str):
        """
        Executes the robot's predicted move using RTC-style threading.
        Continuously requests new actions from the policy server (inference loop),
        executes them on the robot (control loop), and updates telemetry until the turn ends.
        """
        rtc = RTCMotionController(
            robot=self.robot,
            client=self.client,
            robot_lock=self.robot_lock,
            get_obs=self.camera_system.get_latest_obs,
            action_queue=self.action_queue,
            action_horizon=self.cfg.action_horizon,
            control_loop_interval=1.0 / self.cfg.control_hz,
            action_smoothing_factor=self.cfg.action_smoothing_factor,
            rtc_overlap=self.cfg.rtc_overlap,
            action_queue_threshold=self.cfg.action_queue_threshold,
        )

        rtc.start(language_instruction)

    # ------------------------ Main Loop ------------------------
    def run(self) -> None:
        """
        Main loop of the TicTacToeBot.

        - Starts the robot and camera system.
        - Alternates turns between the human player and the robot.
        - On the player's turn: waits for input, clears action queue.
        - On the robot's turn: retrieves latest camera observation, uses VLM to predict move,
        and executes the move using RTCMotionController.
        - Continues until the game reaches a terminal state (win, loss, or draw).
        - Cleans up threads and disconnects the robot on completion.
        """
        self.robot.connect()
        self.camera_system.start()

        set_player_turn(True)

        while self.game_state == "ongoing":
            if get_player_turn():
                self.handle_pause()
                self.wait_until_player_turn(False)  # blocks until turn status changes to False
                time.sleep(0.1)
            else:
                self.handle_resume()

        # Cleanup
        self.camera_system.stop()
        self.robot.disconnect()
        print_blue(GAME_RESULT[self.game_state])
        print_blue(" 🏆 Game over! Thanks for playing.")
