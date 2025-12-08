# Standard library
import base64
import datetime
import json
import os
import time

# Third-party libraries
import cv2
import numpy as np

# Local modules
from backend_client import send_reasoning_chunk
from config import DEFAULT_MOVE, EXPECTED_RESPONSE_FORMAT, OPEN_AI_API_KEY
from google import genai
from google.genai import types
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
        Your goal is to choose the strongest move and win.

        Your response should be a JSON object as follows:
        {json_format}

        Rules and constraints:
        - The "action" format should be: "Place the X in the <position> box", where <position> can be center, top-right, bottom-left, etc.
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
        model_name = "gemini-2.5-flash"
        # --- GEMINI IMPLEMENTATION ---
        if "gemini" in model_name.lower():
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                print_yellow(" ⚠️ GEMINI_API_KEY not found.")
                return DEFAULT_MOVE

            client = genai.Client(api_key=api_key)
            turn_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

            # Encode image
            success, encoded_jpg = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            if not success:
                print_yellow(" ⚠️ Failed to encode image")
                return DEFAULT_MOVE

            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=self.prompt),
                        types.Part.from_bytes(data=encoded_jpg.tobytes(), mime_type="image/jpeg"),
                    ],
                ),
            ]

            # Configure Thinking + JSON
            generate_content_config = types.GenerateContentConfig(
                response_mime_type="application/json",
                thinking_config=types.ThinkingConfig(include_thoughts=True, thinking_budget=10000),
            )

            try:
                stream = client.models.generate_content_stream(
                    model=model_name,
                    contents=contents,
                    config=generate_content_config,
                )

                full_json_text = ""
                for chunk in stream:
                    # Safety check for candidates structure
                    if (
                        not chunk.candidates
                        or not chunk.candidates[0].content
                        or not chunk.candidates[0].content.parts
                    ):
                        continue

                    for part in chunk.candidates[0].content.parts:
                        text_part = part.text or ""
                        if not text_part:
                            continue

                        # Check if this part is a thought
                        is_thought = getattr(part, "thought", False)

                        if is_thought:
                            # Stream thoughts to UI
                            send_reasoning_chunk(text_part, done=False, turn_id=turn_id)
                        else:
                            # Accumulate JSON answer (don't stream to UI)
                            full_json_text += text_part

                send_reasoning_chunk("", done=True, turn_id=turn_id)

                # Parse only the accumulated JSON answer
                cleaned = full_json_text.replace("", "").replace("```", "").strip()
                move_dict = json.loads(cleaned)
                self._validate_move_dict(move_dict)
                return move_dict

            except Exception as e:
                print_yellow(f" ⚠️ Gemini Error: {e}")
                return DEFAULT_MOVE

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
