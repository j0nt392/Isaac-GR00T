import threading
import time
from queue import Empty, Queue

from backend_client import send_frame
from lerobot.robots.robot import Robot


class CameraSystem:
    def __init__(self, robot: Robot, robot_lock: threading.Lock, fps: float):
        self.robot = robot
        self.robot_lock = robot_lock
        self.fps = fps

        self.camera_queue: Queue[dict] = Queue(maxsize=1)
        self.last_observation = None

        self.stop_event = threading.Event()
        self._camera_thread = None
        self._interface_thread = None

    def start(self):
        """Start camera capture and interface display threads."""
        self.stop_event.clear()

        self._camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self._interface_thread = threading.Thread(target=self._interface_display_loop, daemon=True)

        self._camera_thread.start()
        self._interface_thread.start()

    def stop(self):
        """Stop both threads and wait for them to finish."""
        self.stop_event.set()
        self._camera_thread.join()
        self._interface_thread.join()

    def get_latest_obs(self) -> dict:
        """Return the most recent observation, falling back to last frame if queue is empty."""
        try:
            return self.camera_queue.get_nowait()
        except Empty:
            return self.last_observation

    # ---------------------------- Internal Loops ----------------------------

    def _camera_loop(self):
        """
        Continuously captures images and robot state, jointly called 'observation'.
        The observations are stored in a queue, which is accessed by the inference step or when displaying data on the interface.
        """
        while not self.stop_event.is_set():
            try:
                with self.robot_lock:
                    obs = self.robot.get_observation()
            except Exception:
                continue  # skip frame if port busy / packet error
            self.last_observation = obs
            if self.camera_queue.full():
                _ = self.camera_queue.get()  # drop old frame
            self.camera_queue.put(obs)
            time.sleep(1.0 / self.fps)

    def _interface_display_loop(self):
        """
        Continuously sends camera frames to the interface to create a live video feed.
        """
        while not self.stop_event.is_set():
            try:
                obs = self.camera_queue.get(timeout=0.05)  # wait for a frame
                send_frame("front", obs.get("front", None))
                send_frame("wrist", obs.get("wrist", None))
            except Empty:
                # no frame yet, just wait for the next loop
                continue
            time.sleep(1 / 30.0)  # control display rate
