# Standard library
import threading
import time
from collections.abc import Callable
from queue import Empty, Queue

# Third-party libraries
import numpy as np

# Local modules
from backend_client import (
    get_player_turn,
    send_telemetry,
)
from eval_lerobot import Gr00tRobotInferenceClient
from lerobot.robots.robot import Robot  # noqa: F401


class RTCMotionController:
    """
    Handles real-time execution of robot actions with asynchronous policy inference.

    Loops:
        - _inference_loop: Continuously requests new action chunks from the policy server
                           when enough of the previous actions have been executed.
                           Merges old and new actions safely using Real-Time Chunking (RTC).
        - _control_loop: Consumes actions from the queue, applies smoothing, and sends
                         them to the robot at a fixed rate.

    Parameters:
        robot: The Robot instance to send actions to.
        client: Gr00tRobotInferenceClient instance for policy inference.
        robot_lock: Lock to ensure thread-safe robot action commands.
        get_obs: Callable that returns the current robot observation/state.
        action_queue: Queue to hold upcoming robot actions.
        action_horizon: Number of actions predicted per policy request.
        control_loop_interval: Time (s) between sending consecutive actions to the robot.
        action_smoothing_factor: Weight for smoothing actions (0 = no smoothing, 1 = keep previous action).
        rtc_overlap: Number of actions to blend between old and new chunks.
        action_queue_threshold: Fraction of the action horizon remaining before requesting a new chunk.
    """

    def __init__(
        self,
        robot: Robot,
        client: Gr00tRobotInferenceClient,
        robot_lock: threading.Lock,
        get_obs: Callable,
        action_queue: Queue,
        action_horizon: int,
        control_loop_interval: float,
        action_smoothing_factor: float,
        rtc_overlap: int = 4,
        action_queue_threshold: float = 0.5,
    ):
        # --- Robot and concurrency ---
        self.robot = robot
        self.client = client
        self.robot_lock = robot_lock
        self.get_obs = get_obs

        # --- Action queue and timing ---
        self.action_queue = action_queue
        self.action_horizon = action_horizon
        self.control_loop_interval = control_loop_interval
        self.executed_actions_since_threshold = 0

        # --- RTC / smoothing parameters ---
        self.action_smoothing_factor = action_smoothing_factor
        self.rtc_overlap = rtc_overlap
        self.action_queue_threshold = action_queue_threshold

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

    def _control_loop(self) -> None:
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
                time.sleep(self.control_loop_interval)
                continue

            # Smooth new action only if there's a last_action and a meaningful change
            if last_action is not None and self._should_smooth(last_action, action):
                action = {
                    k: self.action_smoothing_factor * last_action[k]
                    + (1 - self.action_smoothing_factor) * action[k]
                    for k in action
                }

            # Send the action safely
            try:
                with self.robot_lock:
                    self.robot.send_action(action)
                    self.executed_actions_since_threshold += 1
            except Exception as e:
                print(f"[RTC control] Error sending action: {e}")

            last_action = action

            # Telemetry outside the lock
            obs = self.get_obs()
            send_telemetry(obs, self.client.robot_state_keys)

            # Maintain consistent control loop rate
            time.sleep(self.control_loop_interval)

    def _should_smooth(
        self, last_action: dict[str, float], new_action: dict[str, float], threshold: float = 1e-3
    ) -> bool:
        """Decide if smoothing is needed based on meaningful change."""
        return any(abs(last_action[k] - new_action[k]) > threshold for k in new_action)

    def _inference_loop(self, language_instruction: str) -> None:
        """
        Continuously fetches action chunks from the policy server and merges them into the action queue,
        respecting chunk size threshold and temporal alignment.
        """
        while not get_player_turn():
            try:
                # Only fetch new chunk if remaining fraction of the queue is below threshold
                qsize = self.action_queue.qsize()
                if qsize / float(self.action_horizon) > self.action_queue_threshold:
                    time.sleep(0.002)
                    continue  # wait until enough actions are consumed

                # Request new action chunk
                obs = self.get_obs()
                action_chunk = self.client.get_action(obs, language_instruction)

                # Merge new chunk into queue safely
                if action_chunk:
                    self._merge_with_queue(action_chunk)
                else:
                    time.sleep(0.05)

                # Reset executed actions counter after merging
                self.executed_actions_since_threshold = 0

            except Exception as e:
                print(f"[RTC inference] error: {e}")
                time.sleep(0.1)

    def _merge_with_queue(self, new_chunk: list[dict[str, float]]) -> None:
        """
        Merge a new action chunk into the current queue while respecting:
        - temporal alignment (executed actions since threshold)
        - overlap blending
        - thread-safe queue operations
        """
        # Determine overlap size (cannot exceed new_chunk length)
        overlap = min(self.rtc_overlap, len(new_chunk))
        # Drain the current queue safely into a local list
        existing = []
        while not self.action_queue.empty():
            try:
                existing.append(self.action_queue.get_nowait())
            except Empty:
                break
        remaining_old = len(existing)
        executed = self.executed_actions_since_threshold
        # Compute the indices in the new chunk for overlap
        start_new = executed
        end_new = min(start_new + overlap, len(new_chunk))
        actual_overlap = min(overlap, remaining_old, end_new - start_new)

        # Blend overlap region if possible
        if remaining_old > 0 and actual_overlap > 0:
            for i in range(actual_overlap):
                old_action = existing[remaining_old - actual_overlap + i]
                new_action = new_chunk[start_new + i]
                # Apply exponential decay to make merged action smooth
                alpha = np.exp(-i / actual_overlap)
                existing[remaining_old - actual_overlap + i] = {
                    k: alpha * old_action[k] + (1 - alpha) * new_action[k] for k in old_action
                }
        # Append remaining new actions after overlap
        merged = existing + new_chunk[end_new:]

        # Requeue safely (respect maxsize)
        for action in merged:
            if self.action_queue.full():
                break
            self.action_queue.put_nowait(action)
