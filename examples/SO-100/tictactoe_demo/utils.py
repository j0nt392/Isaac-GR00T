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
from config import TicTacToeConfig


def print_green(text):
    print(f"\033[92m{text}\033[0m")


def print_yellow(text):
    print(f"\033[93m{text}\033[0m")


def print_blue(text):
    print(f"\033[94m{text}\033[0m")


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]  # top-left
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[2] = pts[np.argmax(s)]  # bottom-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect


def project_image_to_bev(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Canny edge detection
    edges = cv2.Canny(blur, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop over contours and find the largest quadrilateral
    board_contour = None
    max_area = 0
    for cnt in contours:
        # Approximate contour to polygon
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # Check if it's a quadrilateral
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area > max_area:
                max_area = area
                board_contour = approx

    if board_contour is not None:
        pts_src = np.float32(board_contour.reshape(4, 2))
        print("Applying BEV transformation. Detected corners:", pts_src)
    else:
        print("⚠️ WARNING: Could not detect board quadrilateral. Returning original image.")
        return img.copy()

    pts_src = order_points(pts_src)

    pts_dst = np.float32([[0, 0], [600, 0], [600, 600], [0, 600]])
    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
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


def prepare_frame_for_vlm(obs: dict, cfg: TicTacToeConfig) -> np.ndarray:
    """
    Preprocesses an image before sending it to VLM for board analysis.
    The image is projected to BEV for a better perspective of the board, and then optionally enhanced by emphasizing contrast.
    The raw, bev and enhanced images are optionally saved.
    """
    img_raw = obs.get("front")
    img_bev = project_image_to_bev(img_raw)
    img = img_bev
    if cfg.enhance_images:
        img = enhance_image(img_bev)
    if cfg.show_board_images:
        print("Showing board image")
        plt.imshow(img)
        plt.show()
    if cfg.save_images:
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
