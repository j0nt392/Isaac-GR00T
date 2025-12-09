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
    """
    Client for Vision-Language Models (VLM).
    Handles communication for both pre-move decision (Step 1) and post-move state check (Step 2).
    """

    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.client = OpenAI(api_key=OPEN_AI_API_KEY)
        self.model_name = model_name
        self.prompt_before_move = self._build_prompt_before_move()
        self.prompt_after_move = self._build_prompt_after_move()

    # --- Prompt Builders ---

    def _build_prompt_before_move(self) -> str:
        # Step 1: Decision and Pre-Move State Check
        json_format = json.dumps(EXPECTED_RESPONSE_FORMAT, indent=4)
        return f"""
        You are X in a game of Tic-Tac-Toe. The input image shows the current 3x3 board.
        Your goal is to choose the strongest move and win.

        Your response must be a JSON object using the following format:
        {json_format}

        Rules and constraints (Strictly before the move):
        - The "action" format must be: "Place the X in the <position> box".
        - The "action" must be "N/A" only if the game is already over (X or O has 3-in-a-row, or board is full).
        - If the game is already over: Set "game_state" to 'win', 'draw', or 'loss' (from X’s perspective) and "action" to "N/A".
        - If the game is ongoing, set "game_state" to 'ongoing' and provide the best "action".
        - Do NOT evaluate any state *after* your move.
        - Be concise. Output ONLY the JSON.
        """

    def _build_prompt_after_move(self) -> str:
        # Step 2: Post-Move State Check
        json_format = json.dumps(EXPECTED_RESPONSE_FORMAT, indent=4)

        return f"""
        Analyze the input image, which shows the board *after* the previous move by X.
        Your sole task is to determine the final game state.
        
        Your response must be a JSON object using the following format:
        {json_format}

        Rules and constraints (Strictly after the move):
        - Check if X has just won by completing 3-in-a-row. If so, set "game_state" to 'win'.
        - Check if the board is now full and X did not win. If so, set "game_state" to 'draw'.
        - Otherwise, set "game_state" to 'ongoing'.
        - Do NOT suggest any action, nor evaluate future moves. Set "action" to "N/A".
        - Be concise. Output ONLY the JSON.
        """

    # --- Utilities ---

    def _encode_image_to_base64(self, img: np.ndarray) -> str:
        # Convert array to JPEG bytes.
        success, encoded_img = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if not success:
            raise ValueError("Could not encode image")
        # Encode the JPEG bytes to base64.
        encoded_img = base64.b64encode(encoded_img.tobytes()).decode("utf-8")
        return encoded_img

    def _validate_move_dict(self, move_dict: dict) -> dict:
        """Validate the structure of the move dictionary."""
        keys_to_check = EXPECTED_RESPONSE_FORMAT.keys()

        for key in keys_to_check:
            if key not in move_dict:
                print_yellow(f" ⚠️ Missing required key '{key}' in VLM response. Using default.")
                move_dict[key] = DEFAULT_MOVE.get(key, "N/A")
        return move_dict

    # --- VLM Implementation Details ---

    def _call_gemini(self, img: np.ndarray, prompt: str, stream_reasoning: bool = True) -> dict:
        """Handles streaming call to the Gemini API."""
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print_yellow(" ⚠️ GEMINI_API_KEY not found.")
            return DEFAULT_MOVE

        client = genai.Client(api_key=api_key)
        turn_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        # Encode image to JPEG bytes
        success, encoded_jpg = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if not success:
            print_yellow(" ⚠️ Failed to encode image")
            return DEFAULT_MOVE

        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=prompt),
                    types.Part.from_bytes(data=encoded_jpg.tobytes(), mime_type="image/jpeg"),
                ],
            ),
        ]

        # Configure Thinking + JSON
        # Note: reasoning_effort is not used directly, but thinking_budget is relevant.
        generate_content_config = types.GenerateContentConfig(
            response_mime_type="application/json",
            thinking_config=types.ThinkingConfig(include_thoughts=True, thinking_budget=15000),
        )

        try:
            stream = client.models.generate_content_stream(
                model=self.model_name,
                contents=contents,
                config=generate_content_config,
            )

            full_json_text = ""
            for chunk in stream:
                # Basic check for candidates/content structure
                if not (
                    chunk.candidates
                    and chunk.candidates[0].content
                    and chunk.candidates[0].content.parts
                ):
                    continue

                for part in chunk.candidates[0].content.parts:
                    text_part = part.text or ""
                    if not text_part:
                        continue

                    is_thought = getattr(part, "thought", False)

                    if is_thought:
                        if stream_reasoning:
                            send_reasoning_chunk(text_part, done=False, turn_id=turn_id)
                    elif not is_thought:
                        # Accumulate JSON answer
                        full_json_text += text_part

            if stream_reasoning:
                send_reasoning_chunk("", done=True, turn_id=turn_id)

            # Parse and validate
            cleaned = full_json_text.replace("", "").replace("```", "").strip()
            move_dict = json.loads(cleaned)
            return self._validate_move_dict(move_dict)

        except Exception as e:
            print_yellow(f" ⚠️ Gemini Error: {e}")
            return DEFAULT_MOVE  # Fallback if error occurred

    def _call_openai(
        self, img: np.ndarray, prompt: str, reasoning_effort: str, stream_reasoning: bool = True
    ) -> dict:
        """Handles streaming call to the OpenAI API (or similar non-Gemini VLM)."""
        encoded_img = self._encode_image_to_base64(img)
        turn_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        move_dict = DEFAULT_MOVE.copy()  # Use a default copy to accumulate the final result

        stream = self.client.responses.create(
            model=self.model_name,
            reasoning={"effort": reasoning_effort, "summary": "auto"},
            text={"verbosity": "low"},
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
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

            # Live "thinking" summary (only for Step 1)
            if "response.reasoning_summary_text.delta" in t:
                delta = getattr(ev, "delta", "") or ""
                if delta and stream_reasoning:
                    send_reasoning_chunk(delta, done=False, turn_id=turn_id)
                    time.sleep(0.1)
                continue

            # Final JSON arrives as a single done text event
            if t == "response.output_text.done":
                final_json = getattr(ev, "text", "") or ""
                try:
                    move_dict = json.loads(final_json)
                    move_dict = self._validate_move_dict(move_dict)
                except json.JSONDecodeError:
                    print_yellow(f" ⚠️ Could not parse JSON:\n{final_json}. Using default move.")
                    move_dict = DEFAULT_MOVE.copy()
                continue

            # Close the UI stream for reasoning
            if t == "response.completed" and stream_reasoning:
                send_reasoning_chunk("", done=True, turn_id=turn_id)

        return move_dict

    def _call_vlm(
        self, img: np.ndarray, prompt: str, reasoning_effort: str, stream_reasoning: bool = True
    ) -> dict:
        """Routes the call to the appropriate VLM implementation."""
        if "gemini" in self.model_name.lower():
            return self._call_gemini(img, prompt, stream_reasoning)

        # Default to OpenAI implementation for other models
        return self._call_openai(img, prompt, reasoning_effort, stream_reasoning)

    # --- Public VLM Methods (Used by tictactoe_bot.py) ---

    def get_move_decision(self, img: np.ndarray, reasoning_effort: str) -> dict:
        """Step 1: Get the strategic decision (action and pre-move state)."""
        print_yellow(" ➡️ Calling VLM: Step 1 (Move Decision)")
        return self._call_vlm(img, self.prompt_before_move, reasoning_effort, stream_reasoning=True)

    def get_post_move_state(self, img: np.ndarray, reasoning_effort: str) -> dict:
        """Step 2: Get the post-move game state (win/draw/ongoing)."""
        print_yellow(" ➡️ Calling VLM: Step 2 (Post-Move State Check)")
        # Note: Reasoning is suppressed in the VLM implementations for Step 2
        return self._call_vlm(img, self.prompt_after_move, reasoning_effort, stream_reasoning=False)
