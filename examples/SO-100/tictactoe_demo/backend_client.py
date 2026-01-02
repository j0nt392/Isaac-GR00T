# Standard library
import logging
import os
import time

# Third-party libraries
import cv2
import requests
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("backend_client")

BACKEND_URL = os.getenv("BACKEND_URL")
load_dotenv()


# Telemetry endpoints.
def send_telemetry(observation_dict, robot_state_keys):
    motors = [float(observation_dict[k]) for k in robot_state_keys]
    payload = {"t": time.time(), "motors": motors}
    try:
        requests.post(f"{BACKEND_URL}/ingest", json=payload, timeout=0.02)
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to send telemetry: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in send_telemetry: {e}")


# VLM endpoints.
## Final reasoning-decisions.
def send_decisions(move: dict) -> None:
    try:
        payload = {
            "observation": str(move.get("observation", "")),
            "reasoning": str(move.get("reasoning", "")),
            "action": str(move.get("action", "")),
            "game_over": bool(move.get("game_over", False)),
            "game_state": str(move.get("game_state", "")),
            "visible": bool(move.get("visible", True)),
        }
        requests.post(f"{BACKEND_URL}/decisions", json=payload, timeout=0.3)
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to send decision: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in send_reasoning: {e}")


## Live thinking-stream.
def send_reasoning_chunk(chunk: str, done: bool = False, turn_id: str = "") -> None:
    try:
        requests.post(
            f"{BACKEND_URL}/thinking_chunk",
            json={"turn_id": turn_id, "chunk": chunk, "done": done},
            timeout=0.3,
        )
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to send thinking chunk: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in send_reasoning_chunk: {e}")


# Camera endpoints.
def send_frame(camera_key: str, frame_rgb):
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", frame_bgr)
    if not ok:
        return
    try:
        requests.post(
            f"{BACKEND_URL}/ingest_frame/{camera_key}",
            data=buf.tobytes(),
            headers={"content-type": "image/jpeg"},
            timeout=0.001,
        )
    except requests.exceptions.RequestException:
        # High-frequency frames, logging every failure might spam
        pass
    except Exception as e:
        logger.error(f"Unexpected error in send_frame: {e}")


# Game endpoints.
def get_player_turn() -> bool:
    try:
        r = requests.get(f"{BACKEND_URL}/turn_status", timeout=0.2)
        return bool(r.json().get("player_turn", True))
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to get player turn: {e}")
        return True
    except Exception as e:
        logger.error(f"Unexpected error in get_player_turn: {e}")
        return True


def set_player_turn(val: bool) -> None:
    try:
        requests.post(f"{BACKEND_URL}/turn_status", json={"player_turn": val}, timeout=0.2)
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to set player turn: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in set_player_turn: {e}")


def set_board_state(state: list) -> None:
    try:
        requests.post(f"{BACKEND_URL}/board_state", json={"board_state": state}, timeout=0.2)
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to set board state: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in set_board_state: {e}")


def get_board_state() -> dict:
    try:
        r = requests.get(f"{BACKEND_URL}/board_state", timeout=0.2)
        return r.json().get("board_state", {})
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to get board state: {e}")
        return {}
    except Exception as e:
        logger.error(f"Unexpected error in get_board_state: {e}")
        return {}


def set_judge_status(val: bool) -> None:
    try:
        requests.post(f"{BACKEND_URL}/judge_status", json={"is_judging": val}, timeout=0.2)
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to set judge status: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in set_judge_status: {e}")


def stop_ec2() -> None:
    try:
        requests.post(f"{BACKEND_URL}/ec2/stop", timeout=0.2)
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to stop EC2: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in stop_ec2: {e}")
