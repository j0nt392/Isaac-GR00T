# Standard library
import base64
import datetime
import json
import time

# Third-party libraries
import cv2
import numpy as np

# Local modules (google, openai and tenacity are incorrectly detected as local by isort)
from backend_client import send_reasoning_chunk
from config import (
    DEFAULT_MOVE,
    EXPECTED_RESPONSE_FORMAT,
    GEMINI_API_KEY,
    OPEN_AI_API_KEY,
    XAI_API_KEY,
)
from google import genai
from google.genai import types
from google.genai.errors import APIError as GeminiAPIError
from openai import APIError as OpenAIAPIError
from openai import OpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from utils import print_yellow

# Retry settings
RETRYABLE_EXCEPTIONS = (
    GeminiAPIError,
    OpenAIAPIError,
)
MAX_RETRIES = 5
INITIAL_DELAY = 1.0
MAX_WAIT_TIME = 10.0

# Reasoning effort mapping (based on GPT)
REASONING_MAPPING = {
    "low": {
        "gemini": 5000,
        "grok": {"temperature": 0.7, "max_tokens": 500},  # Faster, less deterministic
    },
    "medium": {
        "gemini": 15000,
        "grok": {"temperature": 0.4, "max_tokens": 1000},
    },
    "high": {
        "gemini": 30000,
        "grok": {"temperature": 0.2, "max_tokens": 2000},  # More reasoned/deterministic
    },
}


class VLMClient:
    """
    Unified, resilient client for Vision-Language Models.
    Handles communication for both pre-move decision (Step 1) and post-move state check (Step 2).
    """

    def __init__(self, model_name: str):
        self.model_name = model_name.lower()
        self.prompt_before_move = self._build_prompt_before_move()
        self.prompt_after_move = self._build_prompt_after_move()
        self._init_clients()

        # Map model names to provider functions
        self.provider_map = {
            "gemini-2.5-flash": self._call_gemini,
            "gemini-2.5-pro": self._call_gemini,
            "grok-4": self._call_grok,
            "grok-4-fast-non-reasoning": self._call_grok,
            "gpt-5-nano": self._call_openai,
            "o3": self._call_openai,
        }

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
        - The "action" format must be: "Place the X in the <position> box". Where <position> can only be: "top-left", "top-center", "top-right", "center-left", "center", "center-right", "bottom-left", "bottom-center", "bottom-right".
        - The "action" must be "N/A" only if the game is already over (X or O has 3-in-a-row, or board is full).
        - Double-check that the selected square is not already occupied!
        - If the game is already over: Set "game_state" to 'win', 'draw', or 'loss' (from X’s perspective) and "action" to "N/A".
        - If the game is ongoing, set "game_state" to 'ongoing' and provide the best "action".
        - Do NOT evaluate any state *after* your move.
        - Be concise. Output ONLY the JSON and adhere to the action and position formats strictly.
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
        - Be concise. Output ONLY the JSON and adhere to the format strictly.
        """

    # --- Client initialization ---
    def _init_clients(self):
        """Initialize all provider clients dynamically from config."""
        self.clients = {}

        # Dictionary mapping provider name -> (env var name, client class)
        client_map = {
            "gpt": (OPEN_AI_API_KEY, OpenAI),
            "grok": (
                XAI_API_KEY,
                lambda api_key: OpenAI(api_key=api_key, base_url="https://api.x.ai/v1"),
            ),
            # Other future models can be added here
        }

        for provider_name, (api_key, client_class) in client_map.items():
            if api_key:
                self.clients[provider_name] = client_class(api_key=api_key)

        # Special case for Gemini
        self.gemini_key = GEMINI_API_KEY

    # --- Utilities ---

    def _encode_image_to_base64(self, img: np.ndarray) -> tuple[str, str]:
        success, encoded_jpg = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if not success:
            raise ValueError("Could not encode image")
        base64_data = base64.b64encode(encoded_jpg.tobytes()).decode("utf-8")
        return base64_data, "image/jpeg"

    def _validate_move_dict(self, move_dict: dict) -> dict:
        keys_to_check = EXPECTED_RESPONSE_FORMAT.keys()
        for key in keys_to_check:
            if key not in move_dict:
                print_yellow(f" ⚠️ Missing required key '{key}' in VLM response. Using default.")
                move_dict[key] = DEFAULT_MOVE.get(key, "N/A")
        move_dict["action"] = move_dict["action"].replace("middle", "center")
        return move_dict

    # --- Provider calls and reasoning streaming ---

    def _call_gemini(
        self,
        img: np.ndarray,
        prompt: str,
        turn_id: str,
        stream_reasoning: bool = True,
        reasoning_effort: str = "medium",
    ) -> str:
        """Handles streaming call to the Gemini API."""
        if not self.gemini_key:
            print_yellow(" ⚠️ GEMINI_API_KEY not configured. Using default move.")
            return json.dumps(DEFAULT_MOVE)

        client = genai.Client(api_key=self.gemini_key)
        success, encoded_jpg = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if not success:
            print_yellow(" ⚠️ Failed to encode image. Using default move.")
            return json.dumps(DEFAULT_MOVE)

        thinking_budget = REASONING_MAPPING.get(reasoning_effort, {}).get("gemini", 15000)

        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=prompt),
                    types.Part.from_bytes(data=encoded_jpg.tobytes(), mime_type="image/jpeg"),
                ],
            ),
        ]
        generate_content_config = types.GenerateContentConfig(
            response_mime_type="application/json",
            thinking_config=types.ThinkingConfig(
                include_thoughts=True, thinking_budget=thinking_budget
            ),
        )

        try:
            stream = client.models.generate_content_stream(
                model=self.model_name,
                contents=contents,
                config=generate_content_config,
            )

            full_json_text = ""
            for chunk in stream:
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
                    if is_thought and stream_reasoning:
                        send_reasoning_chunk(text_part, done=False, turn_id=turn_id)
                    elif not is_thought:
                        full_json_text += text_part

            if stream_reasoning:
                send_reasoning_chunk("", done=True, turn_id=turn_id)

            return full_json_text

        except Exception as e:
            print_yellow(f" ⚠️ Gemini Error: {e}")
            return json.dumps(DEFAULT_MOVE)

    def _call_openai(
        self,
        img: np.ndarray,
        prompt: str,
        turn_id: str,
        stream_reasoning: bool = True,
        reasoning_effort: str = "low",
    ) -> str:
        """
        Handles streaming call to OpenAI GPT models that support reasoning and verbosity.
        """
        client = self.clients.get("gpt")
        if not client:
            print_yellow(" ⚠️ OpenAI client not configured. Using default move.")
            return json.dumps(DEFAULT_MOVE)

        # Encode the image to base64
        encoded_img, mime_type = self._encode_image_to_base64(img)

        verbosity = "medium" if "o3" in self.model_name else "low"
        # Stream request
        try:
            stream = client.responses.create(
                model=self.model_name,
                reasoning={"effort": reasoning_effort, "summary": "auto"},
                text={"verbosity": verbosity},
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {
                                "type": "input_image",
                                "image_url": f"data:{mime_type};base64,{encoded_img}",
                            },
                        ],
                    }
                ],
                stream=True,
            )
        except Exception as e:
            print_yellow(f" ⚠️ OpenAI Error: {e}")
            return json.dumps(DEFAULT_MOVE)

        # --- Process stream ---
        full_json_text = ""
        for ev in stream:
            t = getattr(ev, "type", "")

            # Live reasoning summary (Step 1)
            if "response.reasoning_summary_text.delta" in t:
                delta = getattr(ev, "delta", "") or ""
                if delta and stream_reasoning:
                    send_reasoning_chunk(delta, done=False, turn_id=turn_id)
                    time.sleep(0.1)
                continue

            # Final JSON arrives as a single done text event
            if t == "response.output_text.done":
                text = getattr(ev, "text", "") or ""
                full_json_text += text
                continue

            # Close reasoning stream
            if t == "response.completed" and stream_reasoning:
                send_reasoning_chunk("", done=True, turn_id=turn_id)

        return full_json_text

    def _call_grok(
        self,
        img: np.ndarray,
        prompt: str,
        turn_id: str,
        stream_reasoning: bool = True,
        reasoning_effort: str = "medium",
    ) -> str:
        """Handles streaming call to xAI Grok API (OpenAI-compatible)."""
        client = self.clients.get("grok")
        if not client:
            print_yellow(" ⚠️ Grok client not configured. Using default move.")
            return json.dumps(DEFAULT_MOVE)

        # Encode image
        encoded_img, mime_type = self._encode_image_to_base64(img)

        # Get effort mappings (default to medium)
        effort_config = REASONING_MAPPING.get(reasoning_effort, {}).get(
            "grok", {"temperature": 0.4, "max_tokens": 1000}
        )

        # Slightly enhance base prompt for visible reasoning based on effort
        reasoning_instruction = {
            "low": "Think briefly if needed.",
            "medium": "Think step by step.",
            "high": "Think carefully step by step in detail before deciding.",
        }[reasoning_effort]

        full_prompt = f"{prompt}\n\n{reasoning_instruction} Output ONLY the final JSON at the end."

        try:
            stream = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": full_prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime_type};base64,{encoded_img}"},
                            },
                        ],
                    }
                ],
                response_format={"type": "json_object"},
                stream=True,
                temperature=effort_config["temperature"],
                max_tokens=effort_config.get("max_tokens"),
            )

            full_json_text = ""
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    full_json_text += delta
                    if stream_reasoning:
                        send_reasoning_chunk(delta, done=False, turn_id=turn_id)

            if stream_reasoning:
                send_reasoning_chunk("", done=True, turn_id=turn_id)

            return full_json_text

        except Exception as e:
            print_yellow(f" ⚠️ Grok Error: {e}")
            return json.dumps(DEFAULT_MOVE)

    # --- Core unified VLM call ---

    def _call_vlm(
        self,
        img: np.ndarray,
        prompt: str,
        stream_reasoning: bool = True,
        reasoning_effort: str = "medium",
    ) -> dict:
        """Centralized retry, parsing, and validation for any VLM."""
        turn_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        provider_func = self.provider_map.get(self.model_name)

        if not provider_func:
            print_yellow(f" ❌ ERROR: Model '{self.model_name}' not mapped to a provider function.")
            return DEFAULT_MOVE

        @retry(
            stop=stop_after_attempt(MAX_RETRIES),
            wait=wait_exponential(multiplier=INITIAL_DELAY, min=1, max=MAX_WAIT_TIME),
            retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
            reraise=True,
        )
        def robust_provider_call():
            print_yellow(f" 🔄 Attempting call to {self.model_name}...")
            return provider_func(img, prompt, turn_id, stream_reasoning, reasoning_effort)

        try:
            full_json_text = robust_provider_call()
            cleaned = full_json_text.replace("```json", "").replace("```", "").strip()
            move_dict = json.loads(cleaned)
            return self._validate_move_dict(move_dict)
        except RETRYABLE_EXCEPTIONS as e:
            print_yellow(
                f" ❌ CRITICAL: {self.model_name} failed after {MAX_RETRIES} attempts. Error: {e}"
            )
            return DEFAULT_MOVE
        except Exception as e:
            print_yellow(f" ❌ CRITICAL: Unrecoverable error in VLM call. Error: {e}")
            return DEFAULT_MOVE

    # --- Public Methods ---

    def get_move_decision(self, img: np.ndarray, reasoning_effort: str = "low") -> dict:
        """Step 1: Get the strategic decision (action and pre-move state)."""
        print_yellow(" ➡️ Calling VLM: Step 1 (Move Decision)")
        return self._call_vlm(
            img, self.prompt_before_move, stream_reasoning=True, reasoning_effort=reasoning_effort
        )

    def get_post_move_state(self, img: np.ndarray) -> dict:
        """Step 2: Get the post-move game state (win/draw/ongoing)."""
        print_yellow(" ➡️ Calling VLM: Step 2 (Post-Move State Check)")
        return self._call_vlm(img, self.prompt_after_move, stream_reasoning=False)
