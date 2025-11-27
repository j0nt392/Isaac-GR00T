# Standard library
import threading
import time

# Third-party libraries
from collections.abc import Callable
from queue import Empty, Queue

# Local modules
from backend_client import (
    get_player_turn,
    send_telemetry,
)
from eval_lerobot import Gr00tRobotInferenceClient
from eval_lerobot_async import merge_with_queue
from lerobot.robots.robot import Robot  # noqa: F401


class RTCMotionController:
    """
    Handles real-time robot action execution:
    - inference_loop: fetches action chunks from the policy server
    - control_loop: sends smoothed actions to the robot
    Loops run until the player turn changes.
    """

    def __init__(
        self,
        robot: Robot,
        client: Gr00tRobotInferenceClient,
        get_obs: Callable,
        action_queue: Queue,
        control_dt: float,
        smoothing_factor: float,
    ):
        self.robot = robot
        self.client = client
        self.get_obs = get_obs
        self.action_queue = action_queue
        self.control_dt = control_dt
        self.smoothing_factor = smoothing_factor

    def start(self, language_instruction: str):
        # Launch loops
        inference_thread = threading.Thread(
            target=self._inference_loop, args=(language_instruction,), daemon=True
        )
        control_thread = threading.Thread(target=self._control_loop, daemon=True)
        inference_thread.start()
        control_thread.start()

        # Wait for both loops to exit (e.g., turn switched)
        inference_thread.join()
        control_thread.join()

    def _inference_loop(self, language_instruction: str):
        while not get_player_turn():
            try:
                obs = self.get_obs()
                action_chunk = self.client.get_action(obs, language_instruction)
                if action_chunk:
                    merge_with_queue(
                        self.action_queue, action_chunk, overlap=len(action_chunk) // 2
                    )
                else:
                    time.sleep(0.05)
            except Exception as e:
                print(f"[RTC inference] error: {e}")
                time.sleep(0.1)

    def _control_loop(self):
        last_action = None
        while not get_player_turn():
            try:
                action = self.action_queue.get(timeout=0.1)
            except Empty:
                if last_action is None:
                    time.sleep(self.control_dt)
                    continue
                action = last_action
                print("Queue empty, repeating last action")

            if last_action is not None:
                action = {
                    k: self.smoothing_factor * last_action[k]
                    + (1 - self.smoothing_factor) * action[k]
                    for k in action
                }

            try:
                self.robot.send_action(action)
            except Exception as e:
                print(f"[RTC control] Error sending action {e}")

            last_action = action
            obs = self.get_obs()
            send_telemetry(obs, self.client.robot_state_keys)
            time.sleep(self.control_dt)
