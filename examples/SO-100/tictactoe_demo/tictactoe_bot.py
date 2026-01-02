# Standard library
import json
import threading
import time
from queue import Queue

# Local modules
from board_manager import POSITION_TO_COORDS, BoardManager
from camera import CameraSystem
from config import TicTacToeConfig
from debug_display import DebugDisplay
from eval_lerobot import Gr00tRobotInferenceClient
from lerobot.robots import make_robot_from_config  # noqa: F401
from rtc_controller import RTCMotionController
from utils import (
    analyze_board_status,
    get_position_from_action,
    prepare_frame_for_vlm,
    print_blue,
    print_green,
    print_yellow,
)
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
        self.vlm_client = VLMClient(model_name=cfg.vlm_model_name)

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

        # Board game subsystem (manages turns by detecting piece placement)
        self.board_manager = BoardManager(self.camera_system.get_latest_obs)

        # Shared action queue (used by inference and control loops)
        self.action_queue: Queue[dict[str, float]] = Queue(maxsize=cfg.action_horizon)

        # Optional debugging interface (visualizations, diagnostics)
        self.debug_display = DebugDisplay() if cfg.show_board_images else None

    # ------------------ Game State Helpers ------------------
    def handle_player_turn(self) -> None:
        print_yellow(" 👨 Player's turn! Make a move.")
        while not self.action_queue.empty():
            self.action_queue.get_nowait()  # clear action queue before robot's next turn

        # BoardManager will detect the piece placed by the human and change state to "analyzing". Block here until then.
        while self.board_manager.state == "human_turn":
            time.sleep(0.9)

    def handle_robot_turn(self) -> None:
        """Robot's turn: VLM predicts move; then RTC control executes it."""
        print_green(" 🤖 Bot's turn! Thinking...")
        self.board_manager.state = "analyzing"  # pause  BoardManage until VLM decision is made

        # Retrieve latest camera frame for VLM inference
        obs = self.camera_system.get_latest_obs()
        img = prepare_frame_for_vlm(obs, self.cfg, self.debug_display)

        # VLM decision (move or N/A if game over)
        move_dict = self.vlm_client.get_move_decision(img, self.cfg.reasoning_effort)
        send_reasoning(move_dict)
        print_green(f" 🤖 Bot's decision: {json.dumps(move_dict, indent=4)}")

        action = move_dict["action"]
        self.board_manager.state = "robot_turn"
        # Signal the robot is about to perform an action in a certain box
        if action != "N/A":
            row, col = POSITION_TO_COORDS[get_position_from_action(action)]
            self.board_manager.next_robot_move = (row, col)
            # Execute move using real-time chunking. When piece is detected by BoardManager, state will be "analyzing" and move ends.
            self._start_rtc_for_move(action)

    def _start_rtc_for_move(self, language_instruction: str):
        """
        Executes the robot's predicted move using RTC-style threading.
        Continuously requests new actions from the policy server (inference loop),
        executes them on the robot (control loop), and updates telemetry until the turn ends.
        """
        rtc = RTCMotionController(
            board_manager=self.board_manager,
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

        - Starts the robot, camera system and board manager.
        - Alternates turns between the human player and the robot.
        - On the player's turn: clears action queue and waits for BoardManager to detect move.
        - On the robot's turn: retrieves latest camera observation, uses VLM to predict move,
        and executes the move using RTCMotionController. Waits for piece detection, then analyzes board state again.
        - Continues until the game reaches a terminal state (win, loss, or draw).
        - Cleans up threads and disconnects the robot on completion.
        """
        self.robot.connect()
        self.camera_system.start()
        self.board_manager.start()
        self.game_state = analyze_board_status(
            self.board_manager.logical_board
        )  # perform initial analysis (board could be non-empty)

        while self.game_state == "ongoing":
            if self.board_manager.state == "human_turn":
                self.handle_player_turn()
                self.game_state = analyze_board_status(self.board_manager.logical_board)
                print_blue(f" 🔍 Current game state: {self.game_state}")
                self.board_manager.state = "robot_turn"
            elif self.board_manager.state == "robot_turn":
                self.handle_robot_turn()
                self.game_state = analyze_board_status(self.board_manager.logical_board)
                print_blue(f" 🔍 Current game state: {self.game_state}")
                self.board_manager.state = "human_turn"

        send_reasoning({"game_over": True, "game_state": self.game_state, "visible": False})
        # Cleanup
        self.camera_system.stop()
        self.board_manager.stop()
        self.robot.disconnect()
        print_blue(GAME_RESULT[self.game_state])
        print_blue(" 🏆 Game over! Thanks for playing.")
