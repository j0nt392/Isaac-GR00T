# Standard library
import atexit
import datetime
import io
import os
import sys

# Third-party libraries
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Local modules


def print_green(text):
    print(f"\033[92m{text}\033[0m")


def print_yellow(text):
    print(f"\033[93m{text}\033[0m")


def print_blue(text):
    print(f"\033[94m{text}\033[0m")


def project_image_to_bev(img: np.ndarray) -> np.ndarray:
    # Source points (corners of the board in the original image)
    pts_src = np.float32(
        [
            [277, 62],  # top-left
            [519, 140],  # top-right
            [413, 379],  # bottom-right
            [126, 252],  # bottom-left
        ]
    )
    # Destination points (perfect square board)
    pts_dst = np.float32([[0, 0], [600, 0], [600, 600], [0, 600]])
    # Compute perspective transform matrix
    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
    # Apply warp
    warped = cv2.warpPerspective(img, M, (600, 600))

    return warped


def enhance_image(img: np.ndarray) -> np.ndarray:
    # Convert to grayscale for morphology
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Structuring element size should match grid-line thickness
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    # Top-Hat: emphasizes bright thin lines on darker background
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    # Add result to original to boost bright elements
    enhanced = cv2.add(gray, tophat)

    return enhanced


def prepare_frame_for_vlm(self, obs: dict) -> np.ndarray:
    """
    Preprocesses an image before sending it to VLM for board analysis.
    The image is projected to BEV for a better perspective of the board, and then optionally enhanced by emphasizing contrast.
    The raw, bev and enhanced images are optionally saved.
    """
    img_raw = obs.get("front")
    img_bev = project_image_to_bev(img_raw)
    img = img_bev
    if self.cfg.enhance_images:
        img = enhance_image(img_bev)
    if self.cfg.show_board_images:
        print("Showing board image")
        plt.imshow(img)
        plt.show()
    if self.cfg.save_images:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_images({"raw": img_raw, "bev": img_bev, "enhanced": img}, timestamp)

    return img


def save_images(images: dict, timestamp: str) -> None:
    """Save captured board images to disk."""
    os.makedirs("./board_images", exist_ok=True)
    for img_type, img in images.items():
        if img is not None:
            cv2.imwrite(f"./board_images/board_{img_type}_{timestamp}.jpg", img)


class _Tee(io.TextIOBase):
    def __init__(self, original_stream, shared_file):
        self._original = original_stream
        self._file = shared_file

    def write(self, s):
        self._original.write(s)
        self._file.write(s)
        self._file.flush()  # flush so logs are written even if program crashes

    def flush(self):
        self._original.flush()
        self._file.flush()

    def isatty(self):
        return self._original.isatty()

    @property
    def encoding(self):
        # mimic the original stream’s encoding
        return getattr(self._original, "encoding", "utf-8")


def setup_tee_logging(log_path: str | None = None):
    """
    Mirror everything written to stdout and stderr into a text file.
    If log_path is None, create a timestamped file next to this script.
    """
    if log_path is None:
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        logs_dir = os.path.join(os.path.dirname(__file__), "logs")
        os.makedirs(logs_dir, exist_ok=True)
        log_path = os.path.join(logs_dir, f"output_{ts}.txt")

    # Open a single shared file for both stdout and stderr
    log_file = open(log_path, "a", encoding="utf-8")

    # Replace sys.stdout and sys.stderr with tee'd streams
    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)

    # Ensure file closes on exit
    def _close_log():
        try:
            log_file.close()
        except Exception:
            pass

    atexit.register(_close_log)
    print(f"Logging to: {log_path}")
