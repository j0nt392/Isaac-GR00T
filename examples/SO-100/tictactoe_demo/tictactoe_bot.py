# Standard library
import threading
import time
from queue import Queue

# Local modules
from backend_client import get_player_turn, send_reasoning, set_player_turn
from camera import CameraSystem
from config import TicTacToeConfig
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
    def __init__(self, cfg: TicTacToeConfig):
        self.cfg = cfg
        self.game_state = "ongoing"  # ["win", "loss", "draw", "ongoing"] Start as "ongoing" always
        self.turn_event = threading.Event()

        # Bot components
        self.vlm_client = VLMClient()
        self.robot = make_robot_from_config(cfg.robot)  # initialize robot arm
        self.client = Gr00tRobotInferenceClient(  # initialize gr00t client
            host=cfg.policy_host,
            port=cfg.policy_port,
            camera_keys=list(cfg.robot.cameras.keys()),
            robot_state_keys=list(self.robot._motors_ft.keys()),
        )
        self.camera_system = CameraSystem(self.robot, cfg.camera_fps)

        # Action queue
        self.action_queue: Queue[dict[str, float]] = Queue(maxsize=cfg.action_horizon)

        # Action control parameters
        self.control_dt = 1.0 / cfg.control_hz
        self.smoothing_factor = cfg.action_smoothing_factor

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

        # Retrieve latest camera frame for VLM inference
        obs = self.camera_system.get_latest_obs()
        img = prepare_frame_for_vlm(obs, self.cfg)

        move_dict = self.vlm_client.generate_vla_prompt(img, self.cfg.reasoning_effort)
        send_reasoning(move_dict)
        print_green(f" 🤖 Bot's decision: {move_dict['action']}")

        state, action = move_dict["game_state"], move_dict["action"]
        self.game_state = state
        if state != "ongoing" and action == "N/A":
            return

        # Perform the action steps using real-time chunking
        self._start_rtc_for_move(action)

    def _start_rtc_for_move(self, language_instruction: str):
        """
        Executes the robot's predicted move using RTC-style threading.
        Continuously requests new actions from the policy server (inference loop),
        executes them on the robot (control loop), and updates telemetry until the turn ends.
        """
        rtc = RTCMotionController(
            robot=self.robot,
            client=self.client,
            get_obs=self.camera_system.get_latest_obs,
            action_queue=self.action_queue,
            control_dt=self.control_dt,
            smoothing_factor=self.smoothing_factor,
            turn_event=self.turn_event,
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

        player_prompted = False
        set_player_turn(True)

        while self.game_state == "ongoing":
            if get_player_turn():
                self.turn_event.set()
                if not player_prompted:
                    self.handle_pause()
                    player_prompted = True
                    self.wait_until_player_turn(False)
                    time.sleep(0.1)
            else:
                self.turn_event.clear()
                self.handle_resume()
                player_prompted = False
                set_player_turn(True)

        # Cleanup
        self.camera_system.stop()
        self.robot.disconnect()
        print_blue(GAME_RESULT[self.game_state])
        print_blue(" 🏆 Game over! Thanks for playing.")
