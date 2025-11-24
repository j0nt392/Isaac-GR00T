# Standard library
import base64
import datetime
import json
import time

# Third-party libraries
import cv2
import numpy as np

# Local modules
from backend_client import send_reasoning_chunk
from config import DEFAULT_MOVE, EXPECTED_RESPONSE_FORMAT, OPEN_AI_API_KEY
from openai import OpenAI
from utils import print_yellow


class VLMClient:
    def __init__(self):
        self.client = OpenAI(api_key=OPEN_AI_API_KEY)
        self.prompt = self._build_prompt()

    def _build_prompt(self) -> str:
        json_format = json.dumps(EXPECTED_RESPONSE_FORMAT, indent=4)
        prompt = f"""
        You are X in a game of Tic-Tac-Toe. The input image shows the current 3×3 board.
        Your goal is to choose the strongest legal move and win.

        Your response should be a JSON object as follows:
        {json_format}

        Rules and constraints:
        - X is your mark; O is the opponent.
        - Legal moves are ONLY empty squares. Never place X on an occupied square.
        - The "action" format should be: "Place the X in the <position> box", where <position> can be center, top-right, bottom-left, etc.
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
        - If the game is already over (X or O has three-in-a-row, or no empty squares), set:
            - "game_state" to 'win', 'draw', or 'loss' (from X’s perspective)
            - "action" to "N/A"
        - If the game will be over after your move (X completes three-in-a-row or fills the board), set:
            - "game_state" to 'win' or 'draw'
            - "action" according to the required format

        Be concise. Output only the JSON — no extra text.
        """
        return prompt

    def _encode_image_to_base64(self, img: np.ndarray) -> str:
        # Convert array to JPEG bytes.
        success, encoded_img = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if not success:
            raise ValueError("Could not encode image")
        # Encode the JPEG bytes to base64.
        encoded_img = base64.b64encode(encoded_img.tobytes()).decode("utf-8")

        return encoded_img

    def generate_vla_prompt(self, img: np.ndarray, reasoning_effort: str) -> dict:
        """Generate a VLA prompt and get the bot's move prediction."""
        encoded_img = self._encode_image_to_base64(img)
        turn_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        move_dict = {}

        stream = self.client.responses.create(
            model="gpt-5",
            reasoning={"effort": reasoning_effort, "summary": "auto"},
            text={"verbosity": "low"},
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
            stream=True,
        )

        for ev in stream:
            t = getattr(ev, "type", "")

            # Live “thinking” summary
            if "response.reasoning_summary_text.delta" in t:
                delta = getattr(ev, "delta", "") or ""
                if delta:
                    send_reasoning_chunk(delta, done=False, turn_id=turn_id)
                    time.sleep(0.1)
                continue

            # Final JSON arrives as a single done text event
            if t == "response.output_text.done":
                final_json = getattr(ev, "text", "") or ""
                try:
                    move_dict = json.loads(final_json)
                    self._validate_move_dict(move_dict)
                except json.JSONDecodeError:
                    print_yellow(f" ⚠️ Could not parse JSON:\n{final_json}. Using default move.")
                    move_dict = DEFAULT_MOVE
                continue

            # Close the UI stream for reasoning
            if t == "response.completed":
                send_reasoning_chunk("", done=True, turn_id=turn_id)

        # # Send to OpenAI API.
        # response = self.client.responses.create(
        #     model="gpt-5",
        #     reasoning={"effort": reasoning_effort},
        #     input=[
        #         {
        #             "role": "user",
        #             "content": [
        #                 { "type": "input_text", "text": self.prompt },
        #                 {
        #                     "type": "input_image",
        #                     "image_url": f"data:image/jpeg;base64,{encoded_img}",
        #                 },
        #             ],
        #         }
        #     ],
        # )
        # Retrieve and parse response.
        # print_green(" 🤖 VLM response received")
        # text = response.output_text
        # print_green(f"Response: {text}")

        # try:
        #     move_dict = json.loads(text)
        #     self._validate_move_dict(move_dict)
        # except json.JSONDecodeError:
        #     print_yellow(f" ⚠️ GPT response could not be parsed as JSON:\n{text}. Using default move.")
        #     move_dict = DEFAULT_MOVE

        return move_dict

    def _validate_move_dict(self, move_dict: dict) -> None:
        """Validate the structure of the move dictionary to ensure it contains all required keys."""
        for key in EXPECTED_RESPONSE_FORMAT:
            if key not in move_dict:
                print_yellow(f" ⚠️ Missing key '{key}' in GPT response. Using default.")
                move_dict[key] = DEFAULT_MOVE[key]
