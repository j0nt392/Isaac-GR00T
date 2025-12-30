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

# =================================
# GLOBAL CONSTANTS
# =================================

# Minimum required area for a detected board contour (in pixels).
# If the area is smaller than this, the detection is considered a failure,
# and hard-coded fallback points are used. (2x the largest single cell area)
MIN_BOARD_AREA_THRESHOLD = 15000

# Hard-coded source points (manually selected) for fallback BEV warp
FALLBACK_PTS_SRC = np.float32([[261, 51], [559, 140], [396, 464], [50, 279]])

BOARD_CONTOUR_WARNING_COUNTER = 0  # Number of times contour detection has failed
SELECTED_AREA_WARNING_COUNTER = 0  # Number of times area confidence has failed


# =================================
# I/O and System Utilities
# ================================


def print_green(text):
    print(f"\033[92m{text}\033[0m")


def print_yellow(text):
    print(f"\033[93m{text}\033[0m")


def print_blue(text):
    print(f"\033[94m{text}\033[0m")


def save_images(images: dict, timestamp: str) -> None:
    """Save captured board images to disk."""
    os.makedirs("./board_images", exist_ok=True)
    for img_type, img in images.items():
        if img is not None:
            # Handle grayscale vs color images for saving
            if len(img.shape) == 2:
                cv2.imwrite(f"./board_images/board_{img_type}_{timestamp}.jpg", img)
            else:
                cv2.imwrite(f"./board_images/board_{img_type}_{timestamp}.jpg", img)


class _Tee(io.TextIOBase):
    def __init__(self, original_stream, shared_file):
        self._original = original_stream
        self._file = shared_file

    def write(self, s):
        self._original.write(s)
        self._file.write(s)
        self._file.flush()

    def flush(self):
        self._original.flush()
        self._file.flush()

    def isatty(self):
        return self._original.isatty()

    @property
    def encoding(self):
        return getattr(self._original, "encoding", "utf-8")


def setup_tee_logging(log_path: str | None = None):
    """
    Mirror everything written to stdout and stderr into a text file.
    """
    if log_path is None:
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        logs_dir = os.path.join(os.path.dirname(__file__), "logs")
        os.makedirs(logs_dir, exist_ok=True)
        log_path = os.path.join(logs_dir, f"output_{ts}.txt")

    log_file = open(log_path, "a", encoding="utf-8")

    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)

    def _close_log():
        try:
            log_file.close()
        except Exception:
            pass

    atexit.register(_close_log)
    print(f"Logging to: {log_path}")


# =================================
# Geometric Helpers
# =================================


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Orders a set of 4 points (TL, TR, BR, BL) based on their sum and difference.
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)[..., 0]

    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect


def extrapolate_missing_point(pts: np.ndarray) -> np.ndarray | None:
    """
    Takes 3 points and extrapolates the fourth (assumed TL) using vector addition
    to complete the projected parallelogram.
    """
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)[..., 0]

    # Identify the three known corners
    try:
        P_BR = pts[np.argmax(s)]
        P_TR = pts[np.argmin(diff)]
        P_BL = pts[np.argmax(diff)]
    except ValueError:
        return None

    if len(set([tuple(P_BR), tuple(P_TR), tuple(P_BL)])) < 3:
        return None

    P_TL = P_BL + (P_TR - P_BR)

    return np.array([P_TL, P_TR, P_BR, P_BL], dtype="float32")


def calculate_aspect_ratio(pts_ordered: np.ndarray) -> float:
    """
    Calculates the aspect ratio (AvgWidth / AvgHeight) for a set of 4 ordered points.
    """
    width_top = np.linalg.norm(pts_ordered[0] - pts_ordered[1])
    width_bottom = np.linalg.norm(pts_ordered[3] - pts_ordered[2])
    height_right = np.linalg.norm(pts_ordered[1] - pts_ordered[2])
    height_left = np.linalg.norm(pts_ordered[0] - pts_ordered[3])

    avg_width = (width_top + width_bottom) / 2.0
    avg_height = (height_right + height_left) / 2.0

    return avg_width / avg_height if avg_height > 0 else 0.0


# =================================
# Image Processing Utilities
# =================================


def project_image_to_bev(img: np.ndarray, pts_src: np.ndarray, bev_size=(600, 600)) -> np.ndarray:
    """
    Applies perspective transformation (Bird's Eye View) using given source points.
    """
    pts_dst = np.float32([[0, 0], [bev_size[0], 0], [bev_size[0], bev_size[1]], [0, bev_size[1]]])
    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
    warped = cv2.warpPerspective(img, M, bev_size)
    return warped


def enhance_image(img: np.ndarray) -> np.ndarray:
    """
    Enhances the image using morphological operations (Top-Hat) to boost contrast.
    """
    if len(img.shape) == 3:
        # Assuming input to enhance_image (img_bev) is usually 3-channel (BGR) or 1-channel (Gray)
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except cv2.error:
            gray = img
    else:
        gray = img

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    enhanced = cv2.add(gray, tophat)

    return enhanced


def extract_cell_image(img: np.ndarray, row: int, col: int) -> np.ndarray | None:
    """
    Extracts the image of a specific cell from the 3x3 board image.
    """
    detection_result = detect_board_contour(img)
    if detection_result is None:
        return None
    else:
        # Successful detection
        pts_src, _ = detection_result
    img_bev = project_image_to_bev(img, pts_src)
    img_enhanced = enhance_image(img_bev)
    h, w = img_enhanced.shape[:2]
    cell_h, cell_w = h // 3, w // 3
    y1, y2 = row * cell_h, (row + 1) * cell_h
    y1, y2 = max(0, y1), min(h, y2)
    x1, x2 = col * cell_w, (col + 1) * cell_w
    x1, x2 = max(0, x1), min(w, x2)
    margin = 15
    cell = img_enhanced[y1 + margin : y2 - margin, x1 + margin : x2 - margin]
    return cell


# =================================
# PieceDetection Logic
# =================================


def is_piece_in_cell(cell_img: np.ndarray, min_occ: float = 0.04, max_occ: float = 0.12) -> bool:
    """
    Detects if there is a piece in the cell based on the number of occupied pixels.
    The occuppancy ratio must be within the specified min and max thresholds to decrease number of false positives.
    """
    _, thresh = cv2.threshold(cell_img, 200, 255, cv2.THRESH_BINARY)
    occupied = cv2.countNonZero(thresh)
    total = cell_img.shape[0] * cell_img.shape[1]
    ratio = occupied / total
    return min_occ <= ratio <= max_occ


def detect_piece_type(cell_img: np.ndarray, min_val: float = 0.835, max_val: float = 1.2) -> str:
    """
    Determines whether the piece is 'X or 'O' based on the amount of circularity.
    """
    # Find contours
    _, thresh = cv2.threshold(cell_img, 200, 255, cv2.THRESH_BINARY)
    # Dilate contours to fix broken shapes
    kernel = np.ones((3, 3), np.uint8)
    thresh_dilated = cv2.dilate(thresh, kernel, iterations=1)
    contours_dilated, _ = cv2.findContours(
        thresh_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    # Check for O (circle)
    for cnt in contours_dilated:
        area = cv2.contourArea(cnt)
        if area < 50:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if min_val <= circularity <= max_val:
            return "O"
    return "X"


def get_position_from_action(action: str) -> str:
    """
    Extracts the <position> in which the robot will place the piece.
    The action must be in the form "Place the X in the <position> box"
    """
    s = action.split()
    return s[-2]


# =================================
# Core Board Detection Logic
# =================================


def _create_quad_visualization(
    img_raw: np.ndarray, pts_src: np.ndarray, title: str, debug_display=None
) -> np.ndarray:
    """Helper to create and display the 'Selected Quadrilateral' visualization with indexed points."""

    selected_quad_img = img_raw.copy()

    # 1. Draw the polygon (contour)
    # The default red/orange for the detected contour.
    color = (
        (0, 0, 255) if "Fallback" not in title else (0, 165, 255)
    )  # Red for detected, Orange for fallback
    final_contour_for_viz = pts_src.reshape(-1, 1, 2).astype(int)
    cv2.polylines(selected_quad_img, [final_contour_for_viz], True, color, 3)

    # 2. Draw circles and index labels for corner points
    for i, p in enumerate(pts_src):
        # Draw green circle for points
        point_color = (0, 255, 0)
        center = tuple(p.astype(int))
        cv2.circle(selected_quad_img, center, 10, point_color, -1)
        # Draw index label (0=TL, 1=TR, 2=BR, 3=BL)
        cv2.putText(
            selected_quad_img,
            str(i),
            center + np.array([15, -10]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            point_color,
            2,
        )

    # Display the image only if debug_display is active
    if debug_display:
        debug_display.show_image(selected_quad_img, title=title)

    return selected_quad_img


def detect_board_contour(img: np.ndarray) -> tuple[np.ndarray, float] | None:
    """
    Detects the four corner points (pts_src) of the board using Dilation and max-area selection.
    Returns the ordered corner points and the selected area, or None on failure/low confidence.
    """

    # Step 1: Preprocessing for Canny
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Step 2: Canny Edges
    edges = cv2.Canny(blur, 50, 150)

    # Step 3: Contour Healing (Dilation)
    kernel = np.ones((5, 5), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)

    # Step 4: Find Contours on the HEALED image
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 5: Approximate polygons, filter, and select by Max Area
    board_contour = None
    max_area = 0
    selected_area = 0

    for cnt in contours:
        area_raw = cv2.contourArea(cnt)
        if area_raw < 5000:  # Simple area filter
            continue

        epsilon = 0.04 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        num_vertices = len(approx)

        # Filter 1: Must be a 3- or 4-sided convex shape
        if (num_vertices == 3 or num_vertices == 4) and cv2.isContourConvex(approx):

            # Check Aspect Ratio
            pts_temp = approx.reshape(-1, 2).astype(np.float32)

            if num_vertices == 3:
                pts_temp = extrapolate_missing_point(pts_temp)
                if pts_temp is None:
                    continue

            pts_temp_ordered = order_points(pts_temp)
            aspect_ratio = calculate_aspect_ratio(pts_temp_ordered)

            # Filter 2: Must be square-like
            if 0.5 < aspect_ratio < 2.0:
                area_approx = cv2.contourArea(approx)

                # Selection: Choose the largest area among candidates that passed filters
                if area_approx > max_area:
                    max_area = area_approx
                    board_contour = approx
                    selected_area = area_approx  # Store the area

    if board_contour is None:
        global BOARD_CONTOUR_WARNING_COUNTER
        BOARD_CONTOUR_WARNING_COUNTER += 1
        if BOARD_CONTOUR_WARNING_COUNTER == 1 or BOARD_CONTOUR_WARNING_COUNTER % 1000 == 0:
            print("⚠️ WARNING: No suitable contour found.")
        if BOARD_CONTOUR_WARNING_COUNTER >= 10000:
            BOARD_CONTOUR_WARNING_COUNTER = 0  # prevent overflow
        return None

    # Step 6: Final Extrapolation/Ordering
    pts_src_raw = board_contour.reshape(-1, 2).astype(np.float32)

    if len(board_contour) == 3:
        pts_src_raw = extrapolate_missing_point(pts_src_raw)
        if pts_src_raw is None:
            print("⚠️ WARNING: Extrapolation failed.")
            return None

    pts_src = order_points(pts_src_raw)

    # Step 7: Check Area Confidence
    if selected_area < MIN_BOARD_AREA_THRESHOLD:
        global SELECTED_AREA_WARNING_COUNTER
        SELECTED_AREA_WARNING_COUNTER += 1
        if SELECTED_AREA_WARNING_COUNTER == 1 or SELECTED_AREA_WARNING_COUNTER % 1000 == 0:
            print(
                f"⚠️ WARNING: Detected area ({selected_area:.0f}) is too small. Confidence failed."
            )
        if SELECTED_AREA_WARNING_COUNTER >= 10000:
            SELECTED_AREA_WARNING_COUNTER = 0  # prevent overflow
        return None  # Return None to trigger the hard-coded fallback

    # Step 8: Return points and area. Visualization happens in prepare_frame_for_vlm
    return pts_src, selected_area


def prepare_frame_for_vlm(obs: dict, cfg, debug_display=None):
    """
    Preprocesses an image by detecting the board, projecting it to BEV, and optionally enhancing it.
    Displays only two images in debug mode: Input Selection (Image 1) and Final BEV (Image 2).
    """

    img_raw = obs.get("front")
    selected_area = 0

    # --- 1. Detect corners and handle failure/low confidence ---
    detection_result = detect_board_contour(img_raw)

    if detection_result is None:
        # Use hard-coded fallback points
        pts_src_raw = FALLBACK_PTS_SRC.copy()
        pts_src = order_points(pts_src_raw)
        print_yellow("   -> Using hard-coded fallback points for warp.")
        title_viz = "1. Selected Quadrilateral & Corners (Fallback)"
    else:
        # Successful detection
        pts_src, selected_area = detection_result
        title_viz = f"1. Selected Quadrilateral & Corners (Area: {selected_area:.0f})"

    # --- 2. Display Image 1: Input Selection (Raw + Points) ---
    img_selected_quad = _create_quad_visualization(
        img_raw, pts_src, title=title_viz, debug_display=debug_display
    )

    # --- 3. BEV warp ---
    img_bev = project_image_to_bev(img_raw, pts_src)

    img_final = img_bev
    title_final = "2. Final BEV Image (Raw)"

    # --- 4. Enhancement ---
    if cfg.enhance_images:
        img_final = enhance_image(img_bev)
        title_final = "2. Final BEV Image (Enhanced)"

    # --- 5. Display Image 2: Final BEV ---
    # Only display the final image if the flag is set AND the display object is available.
    if cfg.show_board_images and debug_display:
        if cfg.enhance_images:
            # Handle enhanced grayscale image conversion for consistent display
            if len(img_final.shape) == 2:
                img_display = cv2.cvtColor(img_final, cv2.COLOR_GRAY2BGR)
            else:
                img_display = img_final
            debug_display.show_image(img_display, title=title_final)
        else:
            debug_display.show_image(img_final, title=title_final)

    # --- 6. Save Images ---
    if cfg.save_images:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_images(
            {
                "raw": img_raw,
                "selected_area": img_selected_quad,  # Use the image with points drawn
                "bev": img_bev,
                "enhanced": img_final,
            },
            timestamp,
        )

    return img_final
