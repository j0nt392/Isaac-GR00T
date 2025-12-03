# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This is the new Gr00T policy eval script with so100, so101 robot arm. Based on:
https://github.com/huggingface/lerobot/pull/777

Example command:

```shell

python eval_gr00t_so100.py \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=lil_guy \
    --robot.cameras="{ wrist: {type: opencv, index_or_path: 9, width: 640, height: 480, fps: 30}, front: {type: opencv, index_or_path: 15, width: 640, height: 480, fps: 30}}" \
    --policy_host=10.112.209.136 \
    --lang_instruction="Grab markers and place into pen holder."
```


First replay to ensure the robot is working:
```shell
python -m lerobot.replay \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=lil_guy \
    --dataset.repo_id=youliangtan/so100-table-cleanup \
    --dataset.episode=2
```
"""

import logging
import os
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pprint import pformat
from queue import Empty, Queue

import draccus
import matplotlib.pyplot as plt
import numpy as np
from lerobot.cameras.opencv.configuration_opencv import (  # noqa: F401
    OpenCVCameraConfig,
)
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    koch_follower,
    make_robot_from_config,
    so100_follower,
    so101_follower,
)
from lerobot.utils.utils import (
    init_logging,
    log_say,
)

# NOTE:
sys.path.append(os.path.expanduser("~/home/wsi-robotics/forks/Isaac-GR00T/gr00t/eval/"))
# from service import ExternalRobotInferenceClient

from gr00t.eval.service import ExternalRobotInferenceClient

#################################################################################


class Gr00tRobotInferenceClient:
    """The exact keys used is defined in modality.json

    This currently only supports so100_follower, so101_follower
    modify this code to support other robots with other keys based on modality.json
    """

    def __init__(
        self,
        host="localhost",
        port=5555,
        camera_keys=[],
        robot_state_keys=[],
        show_images=False,
    ):
        self.policy = ExternalRobotInferenceClient(host=host, port=port)
        self.camera_keys = camera_keys
        self.robot_state_keys = robot_state_keys
        self.show_images = show_images
        assert (
            len(robot_state_keys) == 6
        ), f"robot_state_keys should be size 6, but got {len(robot_state_keys)} "
        self.modality_keys = ["single_arm", "gripper"]

    def get_action(self, observation_dict, lang: str):
        # first add the images
        obs_dict = {f"video.{key}": observation_dict[key] for key in self.camera_keys}

        # show images
        if self.show_images:
            view_img(obs_dict)

        # Make all single float value of dict[str, float] state into a single array
        state = np.array([observation_dict[k] for k in self.robot_state_keys])
        obs_dict["state.single_arm"] = state[:5].astype(np.float64)
        obs_dict["state.gripper"] = state[5:6].astype(np.float64)
        obs_dict["annotation.human.task_description"] = lang

        # then add a dummy dimension of np.array([1, ...]) to all the keys (assume history is 1)
        for k in obs_dict:
            if isinstance(obs_dict[k], np.ndarray):
                obs_dict[k] = obs_dict[k][np.newaxis, ...]
            else:
                obs_dict[k] = [obs_dict[k]]

        # get the action chunk via the policy server
        # Example of obs_dict for single camera task:
        # obs_dict = {
        #     "video.front": np.zeros((1, 480, 640, 3), dtype=np.uint8),
        #     "video.wrist": np.zeros((1, 480, 640, 3), dtype=np.uint8),
        #     "state.single_arm": np.zeros((1, 5)),
        #     "state.gripper": np.zeros((1, 1)),
        #     "annotation.human.action.task_description": [self.language_instruction],
        # }
        action_chunk = self.policy.get_action(obs_dict)

        # convert the action chunk to a list of dict[str, float]
        lerobot_actions = []
        action_horizon = action_chunk[f"action.{self.modality_keys[0]}"].shape[0]
        for i in range(action_horizon):
            action_dict = self._convert_to_lerobot_action(action_chunk, i)
            lerobot_actions.append(action_dict)
        return lerobot_actions

    def _convert_to_lerobot_action(
        self, action_chunk: dict[str, np.array], idx: int
    ) -> dict[str, float]:
        """
        This is a magic function that converts the action chunk to a dict[str, float]
        This is because the action chunk is a dict[str, np.array]
        and we want to convert it to a dict[str, float]
        so that we can send it to the robot
        """
        concat_action = np.concatenate(
            [np.atleast_1d(action_chunk[f"action.{key}"][idx]) for key in self.modality_keys],
            axis=0,
        )
        assert len(concat_action) == len(self.robot_state_keys), "this should be size 6"
        # convert the action to dict[str, float]
        action_dict = {key: concat_action[i] for i, key in enumerate(self.robot_state_keys)}
        return action_dict


#################################################################################


######################### RTC Merge Function ####################################
def merge_with_queue(action_queue: Queue, new_chunk: list[dict[str, float]], overlap: int = 4):
    """
    Thread-safe merge of new actions into the existing queue with exponential RTC blending.
    Rebuilds the queue safely to prevent abrupt jumps caused by concurrent access.
    """

    # 1. Extract all queued actions (safe, drains queue)
    existing = []
    while not action_queue.empty():
        try:
            existing.append(action_queue.get_nowait())
        except Empty:
            break

    remaining = len(existing)
    actual_overlap = min(overlap, remaining, len(new_chunk))

    # 2. Blend overlap region
    if actual_overlap > 0:
        for i in range(actual_overlap):
            # exponential decay, smooth as butter
            alpha = np.exp(-i / actual_overlap)

            old_act = existing[remaining - actual_overlap + i]
            new_act = new_chunk[i]

            blended = {k: alpha * old_act[k] + (1 - alpha) * new_act[k] for k in old_act}
            existing[remaining - actual_overlap + i] = blended

    # 3. Append remaining new actions
    merged = existing + new_chunk[actual_overlap:]

    # 4. Requeue everything safely
    for action in merged:
        if action_queue.full():
            break
        action_queue.put_nowait(action)


#################################################################################
def view_img(img, overlay_img=None):
    """
    This is a matplotlib viewer since cv2.imshow can be flaky in lerobot env
    """
    if isinstance(img, dict):
        # stack the images horizontally
        img = np.concatenate([img[k] for k in img], axis=1)

    plt.imshow(img)
    plt.title("Camera View")
    plt.axis("off")
    plt.pause(0.001)  # Non-blocking show
    plt.clf()  # Clear the figure for the next frame


def print_yellow(text):
    print(f"\033[93m {text}\033[00m")


@dataclass
class EvalConfig:
    robot: RobotConfig  # the robot to use
    policy_host: str = "localhost"  # host of the gr00t server
    policy_port: int = 5555  # port of the gr00t server
    action_horizon: int = 8  # number of actions to execute from the action chunk
    actions_per_chunk: int = 16  # how many actions to enqueue per policy query (server max ~16)
    chunk_size_threshold: float = 0.5  # send new obs when qsize/chunk_size <= threshold
    lang_instruction: str = "Grab pens and place into pen holder."
    play_sounds: bool = False  # whether to play sounds
    timeout: int = 60  # timeout in seconds
    show_images: bool = False  # whether to show images
    control_hz: float = 30.0
    min_queue_size: int = 16
    max_queue_size: int = 16


@draccus.wrap()
def eval(cfg: EvalConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    # Step 1: Initialize the robot
    robot = make_robot_from_config(cfg.robot)
    robot.connect()

    # get camera keys from RobotConfig
    camera_keys = list(cfg.robot.cameras.keys())
    print("camera_keys: ", camera_keys)

    log_say("Initializing robot", cfg.play_sounds, blocking=True)

    language_instruction = cfg.lang_instruction

    # NOTE: for so100/so101, this should be:
    # ['shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos', 'wrist_flex.pos', 'wrist_roll.pos', 'gripper.pos']
    robot_state_keys = list(robot._motors_ft.keys())
    print("robot_state_keys: ", robot_state_keys)

    # Step 2: Initialize the policy
    policy = Gr00tRobotInferenceClient(
        host=cfg.policy_host,
        port=cfg.policy_port,
        camera_keys=camera_keys,
        robot_state_keys=robot_state_keys,
    )
    log_say(
        "Initializing policy client with language instruction: " + language_instruction,
        cfg.play_sounds,
        blocking=True,
    )

    # Runtime clamps to keep queue/chunk sizes stable and avoid jerkiness
    max_queue_size = min(int(cfg.max_queue_size), 16)
    actions_per_chunk = max(1, min(int(cfg.actions_per_chunk), 16))

    action_queue: Queue[dict[str, float]] = Queue(maxsize=max_queue_size)
    stop_event = threading.Event()
    last_action: dict[str, float] | None = None
    control_dt = 1.0 / max(1.0, float(cfg.control_hz))
    # Track the typical chunk size returned by the policy (initialize from config)
    action_chunk_size = actions_per_chunk

    # ---------- BACKGROUND INFERENCE LOOP ----------
    def inference_loop():
        nonlocal last_action
        nonlocal action_chunk_size
        while not stop_event.is_set():
            try:
                # Only send a fresh observation when queue fullness is below threshold
                # i.e. qsize / action_chunk_size <= chunk_size_threshold
                qsize = action_queue.qsize()
                if (qsize / float(action_chunk_size)) > cfg.chunk_size_threshold:
                    time.sleep(0.002)
                    continue

                # get fresh observation from robot
                obs = robot.get_observation()

                # blocking call to remote GR00T server (ZMQ / HTTP)
                action_chunk = policy.get_action(obs, cfg.lang_instruction)
                # action_chunk: List[dict[str, float]] (desired length ~= actions_per_chunk depending on server)

                # Update our notion of chunk size based on what we received
                received_len = len(action_chunk)
                if received_len > 0:
                    action_chunk_size = max(action_chunk_size, received_len)

                # enqueue actions, respecting max_queue_size
                # for idx, action_dict in enumerate(action_chunk):
                #     if idx >= actions_per_chunk:
                #         break
                #     if stop_event.is_set():
                #         break
                #     if action_queue.full():
                #         break
                #     action_queue.put(action_dict)
                merge_with_queue(action_queue, action_chunk, overlap=cfg.actions_per_chunk // 2)

            except Exception as e:
                logging.error(f"[inference_loop] Error: {e}")
                time.sleep(0.05)

    # ---------- CONTROL LOOP (MAIN THREAD) ----------

    def control_loop():
        nonlocal last_action
        smoothing_factor = 0.8  # 80% previous action, 20% new action (tune as needed)

        while not stop_event.is_set():
            t0 = time.time()
            try:
                # Get a fresh action if available
                action = action_queue.get_nowait()
            except Empty:
                # No new action: hold previous
                if last_action is None:
                    time.sleep(0.002)
                    continue
                action = last_action

            # Apply low-pass filter for smoothness (always, if last_action exists)
            if last_action is not None:
                action = {
                    k: smoothing_factor * last_action[k] + (1 - smoothing_factor) * action[k]
                    for k in action
                }

            # Send filtered action
            try:
                robot.send_action(action)
            except Exception as e:
                logging.error(f"[control_loop] Error sending action: {e}")

            last_action = action  # Update for next iteration

            elapsed = time.time() - t0
            time.sleep(max(0.0, control_dt - elapsed))

    # 4) Run it
    inference_thread = threading.Thread(target=inference_loop, daemon=True)
    inference_thread.start()

    try:
        control_loop()  # run in main thread
    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt: stopping...")
    finally:
        stop_event.set()
        inference_thread.join(timeout=1.0)
        robot.disconnect()
        logging.info("Robot disconnected, eval finished.")


if __name__ == "__main__":
    eval()
