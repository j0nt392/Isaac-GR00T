import atexit
import base64
import datetime
import io
import json
import logging
import os
import sys
import time
from dataclasses import asdict
from pprint import pformat

import cv2
import draccus
import numpy as np
from backend_client import (
    get_player_turn,
    send_frame,
    send_reasoning,
    send_telemetry,
    set_player_turn,
)
from dotenv import load_dotenv
from eval_lerobot import EvalConfig, Gr00tRobotInferenceClient
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    koch_follower,
    make_robot_from_config,
    so100_follower,
    so101_follower,
)
from lerobot.utils.utils import (
    init_logging,
)
from openai import OpenAI
from pynput import keyboard

load_dotenv()

OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")
BACKEND_URL = os.getenv("BACKEND_URL")

PAUSE = "PAUSE"
RESUME = "RESUME"
DEFAULT_MOVE = {
    "observation": "Could not parse board",
    "reasoning": "Error in GPT response",
    "action": "N/A",
    "game_over": False,
}


def print_green(text):
    print(f"\033[92m{text}\033[0m")


def print_yellow(text):
    print(f"\033[93m{text}\033[0m")


def print_blue(text):
    print(f"\033[94m{text}\033[0m")


class VLMClient:
    def __init__(self):
        self.client = OpenAI(api_key=OPEN_AI_API_KEY)
        self.prompt = self._build_prompt()

    def _build_prompt(self) -> str:

        prompt = """
        You are X in a game of Tic-Tac-Toe. The input image shows the current 3×3 board.
        Your goal is to choose the strongest legal move and win.

        Your response should be a JSON object as follows:
        {
            "observation": "<short text describing your observation of the board state>",
            "reasoning": "<short text describing your reasoning for the next move>",
            "action": "<your next move in the format described below>",
            "game_over": "<True/False, indicating if you believe the game is over because of win/draw/loss>",
        }

        Rules and constraints:
        - X is your mark; O is the opponent.
        - Legal moves are ONLY empty squares. Never place X on an occupied square.
        - The "action" format should be "Place the X in the <position>" box". Where <position> can be center, top-right, bottom-left, etc.
        - Decision policy (apply in order):
        1) If you can win this turn, do it.
        2) Else if the opponent can win next turn, block it.
        3) Else create a fork (two simultaneous threats).
        4) Else block the opponent’s fork.
        5) Else take center if empty.
        6) Else take the opposite corner of the opponent if available.
        7) Else take any empty corner.
        8) Else take any empty side.
        - Before finalizing, double-check the chosen square is empty in the image. If not, pick the next-best legal move.
        - If the game is already over (X or O has three-in-a-row, or no empty squares), set "game_over": True and "action": "N/A".

        Be concise. Output only the JSON, no extra text.
        """
        # prompt = """
        #     You are an expert Tic-Tac-Toe player.
        #     The image shows a robotic setup for playing Tic-Tac-Toe.
        #     The pieces are wooden blocks with 'O' and 'X' symbols on them.
        #     Circles represent 'O' and crosses represent 'X'.
        #     Analyze the board state and determine your next move as 'X'.
        #     Follow these rules strictly:
        #     1. Your response should be a JSON object as follows:
        #     {
        #         "observation": "<short text describing your observation of the board state>",
        #         "reasoning": "<short text describing your reasoning for the next move>",
        #         "action": "<your next move in the format described below>",
        #         "game_over": "<True/False, indicating if you believe the game is over because of win/draw/loss>",
        #     }
        #     2. The "action" format should be "Place the X in the <position>" box". Where <position> can be center, top-right, bottom-left, etc.
        #     3. If the game is over, set "game_over" to True, "action" to "N/A", and provide appropriate "observation" and "reasoning" for the game outcome.
        #     4. Be concise and to the point in your responses.
        #     5. The position of the block determines the move, not the location of the symbol on the block. The symbol may appear to be in a different square due to perspective, but the base position of the block is what matters!
        # """
        return prompt

    def _encode_image_to_base64(self, img: np.ndarray) -> str:
        # Convert array to JPEG bytes.
        success, encoded_img = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if not success:
            raise ValueError("Could not encode image")
        # Encode the JPEG bytes to base64.
        encoded_img = base64.b64encode(encoded_img.tobytes()).decode("utf-8")

        return encoded_img

    def generate_vla_prompt(self, img: np.ndarray) -> dict:
        """Generate a VLA prompt and get the bot's move prediction."""
        encoded_img = self._encode_image_to_base64(img)
        # Send to OpenAI API.
        response = self.client.responses.create(
            model="gpt-4.1",
            # reasoning={"effort": "minimal"},
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": self.prompt},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{encoded_img}",
                        },
                    ],
                }
            ],
        )
        # Retrieve and parse response.
        print_green(" 🤖 VLM response received")
        text = response.output_text
        print_green(f"Response: {text}")
        try:
            move_dict = json.loads(text)
            self._validate_move_dict(move_dict)
        except json.JSONDecodeError:
            print_yellow(f" ⚠️ GPT response could not be parsed as JSON:\n{text}")
            move_dict = DEFAULT_MOVE
        # move_dict = DEFAULT_MOVE
        return move_dict

    def _validate_move_dict(self, move_dict: dict) -> None:
        """Validate the structure of the move dictionary to ensure it contains all required keys."""
        for key in DEFAULT_MOVE:
            if key not in move_dict:
                print_yellow(f" ⚠️ Missing key '{key}' in GPT response. Using default.")
                move_dict[key] = DEFAULT_MOVE[key]


class TicTacToeBot:
    def __init__(self, vlm_client: VLMClient, cfg: EvalConfig):
        self.vlm_client = vlm_client
        self.cfg = cfg

        self.game_over = False
        self.state = "PLAYER_TURN"  # or "BOT_TURN"

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

    def on_press_key(self, key: keyboard.Key) -> None:
        """Pause or resume the robot control based on keyboard input."""
        if key == keyboard.Key.space and self.state == "PLAYER_TURN":
            time.sleep(1)
            self.state = "BOT_TURN"

    def handle_pause(self) -> None:
        """Pause the robot and allow the player to make a move."""
        # self.robot.go_home()  # <--- need to implement this in so101
        print_yellow(" 👨 Player's turn! Make a move, then press SPACE to resume.")

    def handle_resume(self) -> None:
        """Capture an image of the board, predict the next move, and execute it."""

        print_green(" 🤖 Bot's turn! Thinking...")
        # Capture image.
        print("Capturing image in handle resume...")
        import datetime

        import matplotlib.pyplot as plt

        timestamp = datetime.datetime.now().strftime("%H_%M_%S")
        print_yellow(f" 📸 Capturing image at {timestamp}")
        _ = self.robot.get_observation()  # flush any previous latest_frame
        time.sleep(1.0 / 30.0)  # wait for the camera to settle
        observation_dict = self.robot.get_observation()
        img = observation_dict["front"]
        img = np.rot90(img, k=2)
        cv2.imwrite(f"./board_images/board_{timestamp}.jpg", img)
        img = improve_image(img)
        plt.imshow(img)
        plt.show()

        # Predict next move.
        move_dict = self.vlm_client.generate_vla_prompt(
            img
        )  # ToDo: send move_dict to interface for logging/display.
        send_reasoning(move_dict)  # Send reasoning to fastapi backend.
        print_green(f" 🤖 Bot's decision: {move_dict['action']}")
        # Stop if game is over.
        if move_dict["game_over"] == True:
            self.game_over = True
            return
        # Execute move.
        self.execute_move(move_dict["action"])
        # self.robot.disconnect()
        time.sleep(0.5)

    def execute_move(self, language_instruction: str) -> None:
        """Execute the predicted move on the Tic-Tac-Toe board."""
        while True:
            if self.state == "PLAYER_TURN":
                break
            # Get the real-time image and predict action horizon.
            print("Getting observation in execute move...")
            observation_dict = self.robot.get_observation()

            if language_instruction == "N/A":
                print_yellow(
                    f" ⚠️ WARNING! No valid move to execute! Switching to cfg.lang_instruction: {self.cfg.lang_instruction}"
                )
                language_instruction = self.cfg.lang_instruction
            action_chunk = self.client.get_action(observation_dict, language_instruction)
            # Execute the action chunk.
            for i in range(self.cfg.action_horizon):
                if get_player_turn():
                    self.state = "PLAYER_TURN"
                    break
                # if self.state == "PLAYER_TURN": break
                action_dict = action_chunk[i]
                self.robot.send_action(action_dict)
                send_telemetry(observation_dict, self.client.robot_state_keys)
                send_frame("front", observation_dict["front"])
                send_frame("wrist", observation_dict["wrist"])
                time.sleep(0.05)  # Implicitly wait for the action to be executed

    def run(self) -> None:
        """Main loop to run the Tic-Tac-Toe bot."""
        # Start keyboard listener for pause/resume functionality.
        # listener = keyboard.Listener(on_press=self.on_press_key)
        # listener.start()
        # Start main control loop.
        self.robot.connect()  # self.robot.activate():
        player_prompted = False
        set_player_turn(True)
        while not self.game_over:
            if self.state == "PLAYER_TURN":
                if not player_prompted:
                    self.handle_pause()
                    player_prompted = True
                    self.wait_until_player_turn(False)
                    self.state = "BOT_TURN"
                    time.sleep(0.1)
            else:
                set_player_turn(False)
                self.handle_resume()
                player_prompted = False
                set_player_turn(True)
                self.state = "PLAYER_TURN"
        print_green(" 🏆 Game over! Thanks for playing.")


class _Tee(io.TextIOBase):
    def __init__(self, original_stream, shared_file):
        self._original = original_stream
        self._file = shared_file

    def write(self, s):
        self._original.write(s)
        self._file.write(s)
        self._file.flush()  # flush so logs are written even if program crashes

    def flush(self):
        self._original.flush()
        self._file.flush()

    def isatty(self):
        return self._original.isatty()

    @property
    def encoding(self):
        # mimic the original stream’s encoding
        return getattr(self._original, "encoding", "utf-8")


def setup_tee_logging(log_path: str | None = None):
    """
    Mirror everything written to stdout and stderr into a text file.
    If log_path is None, create a timestamped file next to this script.
    """
    if log_path is None:
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        logs_dir = os.path.join(os.path.dirname(__file__), "logs")
        os.makedirs(logs_dir, exist_ok=True)
        log_path = os.path.join(logs_dir, f"output_{ts}.txt")

    # Open a single shared file for both stdout and stderr
    log_file = open(log_path, "a", encoding="utf-8")

    # Replace sys.stdout and sys.stderr with tee'd streams
    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)

    # Ensure file closes on exit
    def _close_log():
        try:
            log_file.close()
        except Exception:
            pass

    atexit.register(_close_log)
    print(f"Logging to: {log_path}")


# The draccus wrapper handles argument parsing.
@draccus.wrap()
def main(cfg: EvalConfig):
    setup_tee_logging()
    init_logging()
    logging.info(pformat(asdict(cfg)))
    vlm_client = VLMClient()
    ttt_bot = TicTacToeBot(vlm_client, cfg)
    ttt_bot.run()


def improve_image(img: np.ndarray) -> np.ndarray:
    cropped = img[150:350, 200:425]
    # Source points (corners of the board in the original image)
    pts_src = np.float32(
        [
            [196, 158],  # top-left
            [432, 149],  # top-right
            [406, 336],  # bottom-right
            [224, 344],  # bottom-left
        ]
    )
    # Destination points (straight square board)
    # Let's make it 300x300 pixels
    pts_dst = np.float32([[0, 0], [300, 0], [300, 300], [0, 300]])
    # Compute perspective transform matrix
    M = cv2.getPerspectiveTransform(pts_src, pts_dst)

    # Apply warp
    warped = cv2.warpPerspective(img, M, (300, 300))
    # Smooth
    smoothed = cv2.edgePreservingFilter(warped, flags=1, sigma_s=60, sigma_r=0.4)
    # smoothed = warped
    # 2️⃣ Convert to HSV to adjust brightness/contrast more naturally
    hsv = cv2.cvtColor(smoothed, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    # Increase brightness and contrast on the value channel
    v = cv2.convertScaleAbs(v, alpha=1.5, beta=20)

    # Merge back and convert to RGB
    hsv_enhanced = cv2.merge([h, s, v])
    enhanced = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2RGB)

    # Optional: mild sharpening to make Xs/Os stand out
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    enhanced = cv2.filter2D(enhanced, -1, kernel)
    return enhanced


if __name__ == "__main__":
    main()
