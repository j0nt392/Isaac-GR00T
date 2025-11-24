# Standard library
import datetime
import time

# Third-party libraries
import matplotlib.pyplot as plt

# Local modules
from backend_client import (
    get_player_turn,
    send_frame,
    send_reasoning,
    send_telemetry,
    set_player_turn,
)
from config import TicTacToeConfig
from eval_lerobot import Gr00tRobotInferenceClient
from lerobot.robots import make_robot_from_config  # noqa: F401
from utils import (
    enhance_image,
    print_blue,
    print_green,
    print_yellow,
    project_image_to_bev,
    save_images,
)
from vlm_client import VLMClient

GAME_RESULT = {
    "win": " 🤖 Bot has defeated 👨 Player!",
    "loss": " 👨 Player has defeated 🤖 Bot!",
    "draw": "It's a draw 🤖-👨!",
}


class TicTacToeBot:
    def __init__(self, vlm_client: VLMClient, cfg: TicTacToeConfig):
        self.vlm_client = vlm_client
        self.cfg = cfg

        self.game_state = "ongoing"  #  ["win", "loss", "draw", "ongoing"]

        self.robot = make_robot_from_config(self.cfg.robot)  # initialize robot arm
        self.client = Gr00tRobotInferenceClient(  # initialize gr00t client
            host=cfg.policy_host,
            port=cfg.policy_port,
            camera_keys=list(cfg.robot.cameras.keys()),
            robot_state_keys=list(self.robot._motors_ft.keys()),
        )

    def wait_until_player_turn(self, expected: bool) -> None:
        while True:
            if get_player_turn() == expected:
                return
            time.sleep(0.2)

    def handle_pause(self) -> None:
        """Pause the robot and allow the player to make a move."""
        print_yellow(" 👨 Player's turn! Make a move, then press SPACE to resume.")

    def handle_resume(self) -> None:
        """Robot's turn. Capture an image of the board, predict the next move, and execute it."""
        print_green(" 🤖 Bot's turn! Thinking...")
        images = {"raw": None, "bev": None, "enhanced": None}

        # Capture image
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        _ = self.robot.get_observation()  # flush any previous latest_frame
        time.sleep(1.0 / 30.0)  # wait for the camera to settle
        observation_dict = self.robot.get_observation()
        img_raw = observation_dict["front"]
        images["raw"] = img_raw

        # Preprocess image for GPT
        img_bev = project_image_to_bev(img_raw)
        images["bev"] = img_bev
        img = img_bev

        # (Optional) Enhance image
        if self.cfg.enhance_images:
            img_enhanced = enhance_image(img_bev)
            images["enhanced"] = img_enhanced
            img = img_enhanced
        # (Optional) Show board img that will be sent to VLM
        if self.cfg.show_board_images:
            plt.imshow(img)
            print("showing board image")
            plt.show()

        # (Optional) Save all images used
        if self.cfg.save_images:
            save_images(images, timestamp)

        # Predict next move
        move_dict = self.vlm_client.generate_vla_prompt(img, self.cfg.reasoning_effort)
        send_reasoning(move_dict)  # Send reasoning to fastapi backend
        print_green(f" 🤖 Bot's decision: {move_dict['action']}")

        # Proceed or stop based on game state. Game can end before or after making a move
        state, action = move_dict["game_state"], move_dict["action"]
        self.game_state = state
        if state != "ongoing" and action == "N/A":
            return
        self.execute_move(action)
        time.sleep(0.5)

    def execute_move(self, language_instruction: str) -> None:
        """Execute the predicted move on the Tic-Tac-Toe board."""
        while True:
            if get_player_turn():
                break
            # Get the real-time image for action prediction
            observation_dict = self.robot.get_observation()

            if language_instruction == "N/A":
                print_yellow(" ⚠️ WARNING! No valid move to execute! Robot might act unexpectedly.")

            # Predict action horion from Gr00t client
            action_chunk = self.client.get_action(observation_dict, language_instruction)
            # Execute the action chunk
            for i in range(self.cfg.action_horizon):
                if get_player_turn():
                    break
                # Get frame per action step and send data to interface
                dict_for_interface = self.robot.get_observation()
                send_telemetry(dict_for_interface, self.client.robot_state_keys)
                send_frame("front", dict_for_interface["front"])
                send_frame("wrist", dict_for_interface["wrist"])

                action_dict = action_chunk[i]
                self.robot.send_action(action_dict)
                time.sleep(0.05)  # implicitly wait for the action to be executed

    def run(self) -> None:
        """Main loop to run the Tic-Tac-Toe bot."""
        self.robot.connect()  # activate robot

        player_prompted = False
        set_player_turn(True)

        while self.game_state == "ongoing":
            if get_player_turn():
                if not player_prompted:
                    self.handle_pause()
                    player_prompted = True
                    self.wait_until_player_turn(False)
                    time.sleep(0.1)
            else:
                self.handle_resume()
                player_prompted = False
                set_player_turn(True)

        print_blue(GAME_RESULT[self.game_state])
        print_blue(" 🏆 Game over! Thanks for playing.")
