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
        robot_lock: threading.Lock,
        get_obs: Callable,
        action_queue: Queue,
        control_dt: float,
        smoothing_factor: float,
    ):
        self.robot = robot
        self.client = client
        self.robot_lock = robot_lock
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

    def _should_smooth(
        self, last_action: dict[str, float], new_action: dict[str, float], threshold: float = 1e-3
    ) -> bool:
        """Decide if smoothing is needed based on meaningful change."""
        return any(abs(last_action[k] - new_action[k]) > threshold for k in new_action)

    def _inference_loop(self, language_instruction: str):
        """
        Continuously fetches action chunks from the policy server and merges them into the action queue.
        """
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
        """
        Continuously sends smoothed actions to the robot, and updates telemetry on the interface, until the player's turn.
        """
        last_action = None

        while not get_player_turn():
            try:
                # Wait for a new action with a short timeout to remain responsive
                action = self.action_queue.get(timeout=0.05)
            except Empty:
                # No new action yet; loop again
                time.sleep(self.control_dt)
                continue

            # Smooth only if there's a last_action and a meaningful change
            if last_action is not None and self._should_smooth(last_action, action):
                action = {
                    k: self.smoothing_factor * last_action[k]
                    + (1 - self.smoothing_factor) * action[k]
                    for k in action
                }

            # Send the action safely
            try:
                with self.robot_lock:
                    self.robot.send_action(action)
            except Exception as e:
                print(f"[RTC control] Error sending action: {e}")

            last_action = action

            # Telemetry outside the lock
            obs = self.get_obs()
            send_telemetry(obs, self.client.robot_state_keys)

            # Maintain consistent control loop rate
            time.sleep(self.control_dt)
