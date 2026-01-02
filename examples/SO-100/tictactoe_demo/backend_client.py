# Standard library
import os
import time

# Third-party libraries
import cv2
import requests
from dotenv import load_dotenv

BACKEND_URL = os.getenv("BACKEND_URL")

load_dotenv()


# Telemetry endpoints.
def send_telemetry(observation_dict, robot_state_keys):
    motors = [float(observation_dict[k]) for k in robot_state_keys]
    payload = {"t": time.time(), "motors": motors}
    try:
        requests.post(f"{BACKEND_URL}/ingest", json=payload, timeout=0.02)
    except Exception as e:
        print(f"Failed to send telemetry: {e}")


# VLM endpoints.
## Final reasoning-decisions.
def send_reasoning(move: dict) -> None:
    try:
        payload = {
            "observation": str(move.get("observation", "")),
            "reasoning": str(move.get("reasoning", "")),
            "action": str(move.get("action", "")),
            "game_over": bool(move.get("game_over", False)),
            "game_state": str(move.get("game_state", "")),
            "visible": bool(move.get("visible", True)),
        }
        requests.post(f"{BACKEND_URL}/reasoning", json=payload, timeout=0.3)
    except Exception:
        pass


## Live thinking-stream.
def send_reasoning_chunk(chunk: str, done: bool = False, turn_id: str = "") -> None:
    try:
        requests.post(
            f"{BACKEND_URL}/reasoning_chunk",
            json={"turn_id": turn_id, "chunk": chunk, "done": done},
            timeout=0.3,
        )
    except Exception:
        pass


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
    except Exception:
        pass


# Game endpoints.
def get_player_turn() -> bool:
    try:
        r = requests.get(f"{BACKEND_URL}/turn_status", timeout=0.2)
        return bool(r.json().get("player_turn", True))
    except Exception:
        return True


def set_player_turn(val: bool) -> None:
    try:
        requests.post(f"{BACKEND_URL}/turn_status", json={"player_turn": val}, timeout=0.2)
    except Exception:
        pass


def set_board_state(state: list) -> None:
    try:
        requests.post(f"{BACKEND_URL}/board_state", json={"board_state": state}, timeout=0.2)
    except Exception:
        pass


def get_board_state() -> dict:
    try:
        r = requests.get(f"{BACKEND_URL}/board_state", timeout=0.2)
        return r.json().get("board_state", {})
    except Exception:
        return {}


def set_judge_status(val: bool) -> None:
    try:
        requests.post(f"{BACKEND_URL}/judge_status", json={"is_judging": val}, timeout=0.2)
    except Exception:
        pass


def stop_ec2() -> None:
    try:
        requests.post(f"{BACKEND_URL}/ec2/stop", timeout=0.2)
    except Exception:
        pass
