# Standard library
import atexit
import datetime
import io
import os
import sys

# Third-party libraries
import cv2
import numpy as np

# Local modules


def print_green(text):
    print(f"\033[92m{text}\033[0m")


def print_yellow(text):
    print(f"\033[93m{text}\033[0m")


def print_blue(text):
    print(f"\033[94m{text}\033[0m")


def project_image_to_bev(img: np.ndarray) -> np.ndarray:
    # Rotate image 180 degrees to align with user point-of-view
    img = np.rot90(img, k=2)

    # Source points (corners of the board in the original image)
    pts_src = np.float32(
        [
            [196, 158],  # top-left
            [432, 149],  # top-right
            [406, 336],  # bottom-right
            [224, 344],  # bottom-left
        ]
    )
    # Destination points (straight square board)
    pts_dst = np.float32([[0, 0], [300, 0], [300, 300], [0, 300]])
    # Compute perspective transform matrix
    M = cv2.getPerspectiveTransform(pts_src, pts_dst)

    # Apply warp
    warped = cv2.warpPerspective(img, M, (300, 300))

    return warped


def enhance_image(img: np.ndarray) -> np.ndarray:
    # Apply smooth (optional)
    smoothed = cv2.edgePreservingFilter(img, flags=1, sigma_s=60, sigma_r=0.4)

    # 2️⃣ Convert to HSV to adjust brightness/contrast more naturally
    hsv = cv2.cvtColor(smoothed, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Increase brightness and contrast on the value channel
    v = cv2.convertScaleAbs(v, alpha=1.5, beta=20)

    # Merge back and convert to RGB
    hsv_enhanced = cv2.merge([h, s, v])
    enhanced = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)

    # Mild sharpening to make Xs/Os stand out
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    enhanced = cv2.filter2D(enhanced, -1, kernel)

    return enhanced


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
