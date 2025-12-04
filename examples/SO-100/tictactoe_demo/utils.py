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
from config import TicTacToeConfig
from debug_display import DebugDisplay


def print_green(text):
    print(f"\033[92m{text}\033[0m")


def print_yellow(text):
    print(f"\033[93m{text}\033[0m")


def print_blue(text):
    print(f"\033[94m{text}\033[0m")


def _project_image_to_bev(img: np.ndarray) -> np.ndarray:
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


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect


def edge_length_score(quad):
    """Compute score of quadrilateral based on sum of edge lengths"""
    pts = quad.reshape(4, 2)
    score = 0
    for i in range(4):
        p1 = pts[i]
        p2 = pts[(i + 1) % 4]
        score += np.linalg.norm(p2 - p1)
    return score


def detect_board_contour(img: np.ndarray, debug_display=None):
    """
    Detect the board by choosing the quadrilateral with the longest edges.
    Displays all intermediate steps using debug_display.
    """
    h, w = img.shape[:2]
    pts_src = np.float32([[261, 51], [559, 140], [396, 464], [50, 279]])
    first_warp = project_image_to_bev(img, pts_src)
    # Step 1: Grayscale + blur
    gray = cv2.cvtColor(first_warp, cv2.COLOR_BGR2GRAY)

    # Step 2: Canny edges
    edges = cv2.Canny(gray, 50, 150)
    if debug_display:
        debug_display.show_image(cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB), title="Canny Edges")

    # Step 3: Find all external contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if debug_display:
        contour_img = img.copy()
        cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
        debug_display.show_image(contour_img, title="All Contours")

    # Step 4: Approximate polygons and keep only quadrilaterals
    quads = []
    for cnt in contours:
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) == 4:
            quads.append(approx)

    if debug_display:
        quad_img = img.copy()
        cv2.drawContours(quad_img, quads, -1, (255, 0, 0), 2)
        debug_display.show_image(quad_img, title="All Quadrilaterals")

    if not quads:
        print("⚠️ WARNING: No quadrilaterals found. Returning raw image.")
        return None

    # Step 5: Score quadrilaterals by total edge length
    scored_quads = [(edge_length_score(q), q) for q in quads]
    scored_quads.sort(key=lambda x: x[0], reverse=True)
    board_contour = scored_quads[0][1]

    if debug_display:
        board_img = img.copy()
        cv2.polylines(board_img, [board_contour], True, (0, 0, 255), 3)
        debug_display.show_image(board_img, title="Selected Quadrilateral")

    # Step 6: Order corners
    pts_src = np.float32(board_contour.reshape(4, 2))
    pts_src = order_points(pts_src)

    if debug_display:
        ordered_img = img.copy()
        for i, p in enumerate(pts_src):
            cv2.circle(ordered_img, tuple(p.astype(int)), 10, (0, 255, 0), -1)
            cv2.putText(
                ordered_img,
                str(i),
                tuple(p.astype(int) + np.array([10, -10])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
        debug_display.show_image(ordered_img, title="Ordered Corners (0=TL,1=TR,2=BR,3=BL)")

    return pts_src


def project_image_to_bev(img: np.ndarray, pts_src: np.ndarray, bev_size=(600, 600)):
    pts_dst = np.float32([[0, 0], [bev_size[0], 0], [bev_size[0], bev_size[1]], [0, bev_size[1]]])
    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
    warped = cv2.warpPerspective(img, M, bev_size)
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


def prepare_frame_for_vlm(
    obs: dict, cfg: TicTacToeConfig, debug_display: DebugDisplay
) -> np.ndarray:
    """
    Preprocesses an image before sending it to VLM for board analysis.
    The image is projected to BEV for a better perspective of the board, and then optionally enhanced by emphasizing contrast.
    The raw, bev and enhanced images are optionally saved.
    """
    img_raw = obs.get("front")
    pts_src = np.float32([[261, 51], [559, 140], [396, 464], [50, 279]])
    img_bev = project_image_to_bev(img_raw, pts_src)
    img = img_bev
    if cfg.enhance_images:
        img = enhance_image(img_bev)
    if cfg.show_board_images:
        debug_display.show_image(img)
    if cfg.save_images:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_images({"raw": img_raw, "bev": img_bev, "enhanced": img}, timestamp)

    return img


# Main preprocessing function
def _prepare_frame_for_vlm(obs: dict, cfg, debug_display=None):
    img_raw = obs.get("front")

    # Detect corners
    pts_src = detect_board_contour(img_raw, debug_display=debug_display)
    if pts_src is None:
        warped = img_raw.copy()
        if debug_display:
            debug_display.show_image(warped, title="Fallback: Original Image")
        return warped

    # BEV warp
    img_bev = project_image_to_bev(img_raw, pts_src)
    if debug_display:
        debug_display.show_image(img_bev, title="Warped BEV")

    img_final = img_bev
    if cfg.enhance_images:
        img_final = enhance_image(img_bev)
        if debug_display:
            img_rgb = cv2.cvtColor(img_final, cv2.COLOR_GRAY2RGB)
            debug_display.show_image(img_rgb, title="Enhanced Image")

    if cfg.save_images:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_images({"raw": img_raw, "bev": img_bev, "enhanced": img_final}, timestamp)

    if cfg.show_board_images and debug_display:
        debug_display.show_image(img_final, title="Final Output")

    return img_final


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
