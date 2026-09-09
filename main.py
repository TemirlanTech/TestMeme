"""MemeCV: webcam expression and hand-gesture meme viewer.

Uses the current MediaPipe Tasks API (FaceLandmarker and HandLandmarker),
which works with current MediaPipe releases and does not use the removed
``mp.solutions`` namespace.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import urllib.request
from collections import deque
from pathlib import Path
from typing import Deque, Iterable, Optional, Sequence, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

DEFAULT = "default"
SMILE = "smile"
THINKING = "thinking"
PEACE = "peace"
THUMBS_UP = "thumbs_up"
TIMEOUT = "timeout"
STABLE_FRAMES = 7

# Face Landmarker indices.
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
UPPER_LIP = 13
LOWER_LIP = 14
FACE_LEFT = 234
FACE_RIGHT = 454
FOREHEAD = 10
CHIN = 152
NOSE_TIP = 1
LEFT_IRIS = range(468, 473)
RIGHT_IRIS = range(473, 478)

# Hand Landmarker indices.
WRIST = 0
THUMB_TIP = 4
THUMB_IP = 3
THUMB_MCP = 2
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_TIP = 12
RING_MCP = 13
RING_PIP = 14
RING_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_TIP = 20

WHITE = (255, 255, 255)
PINK = (180, 105, 255)
GREEN = (80, 220, 100)
YELLOW = (0, 220, 255)

MEME_FILES = {
    SMILE: "109fb257daabe2f3db63bd7bc1944934.jpg",
    PEACE: "109fb257daabe2f3db63bd7bc1944934.jpg",
    THINKING: "maxresdefault.jpg",
    DEFAULT: "maxresdefault.jpg",
    THUMBS_UP: "7dc6efb0fe7548ae00dd6143e739f630.jpg",
    TIMEOUT: "bc3d38ffc8a2e9a574bb54d3bffa5445.jpg",
}

MEME_TITLES = {
    DEFAULT: "Thinking monkey / default",
    SMILE: "Smiling monkey",
    THINKING: "Thinking monkey",
    PEACE: "Smiling monkey (peace sign)",
    THUMBS_UP: "Thumbs-up monkey",
    TIMEOUT: "Time-out meme",
}

# Official MediaPipe model-host URLs. The .task files are downloaded on first run.
MODEL_URLS = {
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/1/face_landmarker.task"
    ),
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task"
    ),
}

Point = Tuple[float, float]
# Hand skeleton edges, using the 21-point MediaPipe hand landmark topology.
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)


def _point(landmarks: Sequence, index: int) -> Point:
    landmark = landmarks[index]
    return float(landmark.x), float(landmark.y)


def _average(points: Iterable[Point]) -> Point:
    values = list(points)
    return sum(p[0] for p in values) / len(values), sum(p[1] for p in values) / len(values)


def _distance(first: Point, second: Point) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


def _face_scale(face_landmarks: Sequence) -> float:
    return max(_distance(_point(face_landmarks, FOREHEAD), _point(face_landmarks, CHIN)), 1e-6)


def detect_big_smile(face_landmarks: Sequence) -> bool:
    mouth_width = _distance(_point(face_landmarks, MOUTH_LEFT), _point(face_landmarks, MOUTH_RIGHT))
    face_width = max(_distance(_point(face_landmarks, FACE_LEFT), _point(face_landmarks, FACE_RIGHT)), 1e-6)
    mouth_open = _distance(_point(face_landmarks, UPPER_LIP), _point(face_landmarks, LOWER_LIP))
    return mouth_width / face_width >= 0.28 and mouth_open / _face_scale(face_landmarks) >= 0.045


def _iris_is_high(face_landmarks: Sequence, iris_indices: range, upper_lid: int, lower_lid: int) -> bool:
    iris = _average(_point(face_landmarks, index) for index in iris_indices)
    upper = _point(face_landmarks, upper_lid)
    lower = _point(face_landmarks, lower_lid)
    eye_height = lower[1] - upper[1]
    if eye_height <= 1e-6:
        return False
    return (lower[1] - iris[1]) / eye_height >= 0.58


def detect_looking_up_or_thinking(face_landmarks: Sequence) -> bool:
    """Detect an upward-looking/thinking pose using iris and eye landmarks."""
    if len(face_landmarks) >= 478:
        return _iris_is_high(face_landmarks, LEFT_IRIS, 159, 145) and _iris_is_high(
            face_landmarks, RIGHT_IRIS, 386, 374
        )
    eye_mid = _average((_point(face_landmarks, 33), _point(face_landmarks, 263)))
    nose = _point(face_landmarks, NOSE_TIP)
    return nose[1] < eye_mid[1] + 0.18 * _face_scale(face_landmarks)


def _finger_is_up(hand_landmarks: Sequence, tip: int, pip: int, mcp: int) -> bool:
    return _point(hand_landmarks, tip)[1] < _point(hand_landmarks, pip)[1] < _point(hand_landmarks, mcp)[1]


def _finger_is_curled(hand_landmarks: Sequence, tip: int, pip: int, mcp: int) -> bool:
    return not _finger_is_up(hand_landmarks, tip, pip, mcp)


def detect_peace_sign(hand_landmarks: Sequence) -> bool:
    return (
        _finger_is_up(hand_landmarks, INDEX_TIP, INDEX_PIP, INDEX_MCP)
        and _finger_is_up(hand_landmarks, MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP)
        and _finger_is_curled(hand_landmarks, RING_TIP, RING_PIP, RING_MCP)
        and _finger_is_curled(hand_landmarks, PINKY_TIP, PINKY_PIP, PINKY_MCP)
    )


def detect_thumbs_up(hand_landmarks: Sequence) -> bool:
    thumb_up = _point(hand_landmarks, THUMB_TIP)[1] < _point(hand_landmarks, THUMB_IP)[1] < _point(hand_landmarks, THUMB_MCP)[1]
    fingers_curled = all(
        _finger_is_curled(hand_landmarks, tip, pip, mcp)
        for tip, pip, mcp in (
            (INDEX_TIP, INDEX_PIP, INDEX_MCP),
            (MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP),
            (RING_TIP, RING_PIP, RING_MCP),
            (PINKY_TIP, PINKY_PIP, PINKY_MCP),
        )
    )
    return thumb_up and fingers_curled


def _is_vertical_stem(hand_landmarks: Sequence) -> bool:
    wrist = _point(hand_landmarks, WRIST)
    middle_tip = _point(hand_landmarks, MIDDLE_TIP)
    return abs(middle_tip[1] - wrist[1]) > abs(middle_tip[0] - wrist[0]) * 1.35 and middle_tip[1] < wrist[1]


def _is_horizontal_bar(hand_landmarks: Sequence) -> bool:
    wrist = _point(hand_landmarks, WRIST)
    middle_mcp = _point(hand_landmarks, MIDDLE_MCP)
    return abs(middle_mcp[0] - wrist[0]) > abs(middle_mcp[1] - wrist[1]) * 1.35


def _palm_center(hand_landmarks: Sequence) -> Point:
    return _average(_point(hand_landmarks, index) for index in (WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP))


def _stem_fingertip_touches_bar(stem: Sequence, bar: Sequence) -> bool:
    bar_palm = _palm_center(bar)
    nearest_tip = min(_distance(_point(stem, tip), bar_palm) for tip in (MIDDLE_TIP, INDEX_TIP))
    bar_width = _distance(_point(bar, INDEX_MCP), _point(bar, PINKY_MCP))
    return nearest_tip <= max(0.10, bar_width * 1.15)


def detect_timeout_gesture(hand_landmarks_list: Sequence[Sequence]) -> bool:
    if len(hand_landmarks_list) != 2:
        return False
    first, second = hand_landmarks_list
    return (
        (_is_vertical_stem(first) and _is_horizontal_bar(second) and _stem_fingertip_touches_bar(first, second))
        or (_is_vertical_stem(second) and _is_horizontal_bar(first) and _stem_fingertip_touches_bar(second, first))
    )


def detect_hand_gesture(hand_landmarks_list: Sequence[Sequence]) -> Optional[str]:
    if detect_timeout_gesture(hand_landmarks_list):
        return TIMEOUT
    for hand in hand_landmarks_list:
        if detect_thumbs_up(hand):
            return THUMBS_UP
        if detect_peace_sign(hand):
            return PEACE
    return None


def detect_current_state(face_result, hand_result) -> str:
    hands = list(getattr(hand_result, "hand_landmarks", []) or [])
    hand_state = detect_hand_gesture(hands)
    if hand_state is not None:
        return hand_state
    faces = list(getattr(face_result, "face_landmarks", []) or [])
    if faces:
        face_landmarks = faces[0]
        if detect_big_smile(face_landmarks):
            return SMILE
        if detect_looking_up_or_thinking(face_landmarks):
            return THINKING
    return DEFAULT


def update_stable_state(history: Deque[str], candidate: str, current_state: str, required_frames: int = STABLE_FRAMES) -> str:
    history.append(candidate)
    if len(history) == required_frames and len(set(history)) == 1:
        return candidate
    return current_state


def draw_face_landmarks(frame: np.ndarray, face_landmarks: Sequence) -> None:
    height, width = frame.shape[:2]
    for landmark in face_landmarks:
        x = min(max(int(landmark.x * width), 0), width - 1)
        y = min(max(int(landmark.y * height), 0), height - 1)
        cv2.circle(frame, (x, y), 1, WHITE, -1, cv2.LINE_AA)


def draw_hand_landmarks(frame: np.ndarray, hand_landmarks: Sequence) -> None:
    height, width = frame.shape[:2]
    points = [
        (min(max(int(p.x * width), 0), width - 1), min(max(int(p.y * height), 0), height - 1))
        for p in hand_landmarks
    ]
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], PINK, 2, cv2.LINE_AA)
    for point in points:
        cv2.circle(frame, point, 4, PINK, -1, cv2.LINE_AA)


def make_placeholder(title: str, filename: str) -> np.ndarray:
    canvas = np.zeros((600, 800, 3), dtype=np.uint8)
    canvas[:] = (35, 35, 35)
    cv2.putText(canvas, "Meme image not found", (145, 245), cv2.FONT_HERSHEY_SIMPLEX, 1.15, WHITE, 2, cv2.LINE_AA)
    cv2.putText(canvas, title, (150, 305), cv2.FONT_HERSHEY_SIMPLEX, 0.85, PINK, 2, cv2.LINE_AA)
    cv2.putText(canvas, filename, (60, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.58, YELLOW, 1, cv2.LINE_AA)
    cv2.putText(canvas, "Add the image under assets/new/", (160, 445), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 1, cv2.LINE_AA)
    return canvas


def load_meme_images(asset_dir: Path) -> dict[str, np.ndarray]:
    images: dict[str, np.ndarray] = {}
    for state, filename in MEME_FILES.items():
        image = cv2.imread(str(asset_dir / filename))
        images[state] = image if image is not None else make_placeholder(MEME_TITLES[state], filename)
    return images


def render_meme_window(image: np.ndarray, state: str, window_size: Tuple[int, int] = (800, 650)) -> np.ndarray:
    window_width, window_height = window_size
    canvas = np.zeros((window_height, window_width, 3), dtype=np.uint8)
    canvas[:] = (18, 18, 18)
    image_height, image_width = image.shape[:2]
    max_width, max_height = window_width - 24, window_height - 70
    scale = min(max_width / image_width, max_height / image_height)
    resized = cv2.resize(image, (max(1, int(image_width * scale)), max(1, int(image_height * scale))), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    top = 48 + (max_height - resized.shape[0]) // 2
    left = (window_width - resized.shape[1]) // 2
    canvas[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
    cv2.putText(canvas, f"MemeCV: {MEME_TITLES[state]}", (16, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.72, WHITE, 2, cv2.LINE_AA)
    return canvas


def ensure_models(model_dir: Path) -> None:
    """Download official .task models once, so setup works with new MediaPipe."""
    model_dir.mkdir(parents=True, exist_ok=True)
    for filename, url in MODEL_URLS.items():
        path = model_dir / filename
        if path.exists() and path.stat().st_size > 100_000:
            continue
        print(f"Downloading MediaPipe model: {filename}")
        try:
            urllib.request.urlretrieve(url, path)
        except Exception as error:
            path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Could not download {filename}. Check internet access or download it manually "
                f"from {url} and place it in {model_dir}."
            ) from error


def open_webcam(camera_index: int) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened() and sys.platform.startswith("win"):
        capture.release()
        capture = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    return capture


def run(camera_index: int = 0) -> int:
    capture = open_webcam(camera_index)
    if not capture.isOpened():
        print("ERROR: MemeCV could not open the webcam. Please run from a Windows terminal, not WSL2, and check the camera.")
        capture.release()
        return 1

    root_dir = Path(__file__).resolve().parent
    ensure_models(root_dir / "models")
    meme_images = load_meme_images(root_dir / "assets" / "new")
    current_state = DEFAULT
    history: Deque[str] = deque(maxlen=STABLE_FRAMES)
    timestamp_ms = 0

    face_options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(root_dir / "models" / "face_landmarker.task")),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
    )
    hand_options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(root_dir / "models" / "hand_landmarker.task")),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    try:
        with vision.FaceLandmarker.create_from_options(face_options) as face_landmarker, vision.HandLandmarker.create_from_options(hand_options) as hand_landmarker:
            while True:
                success, frame = capture.read()
                if not success:
                    print("WARNING: Could not read a frame from the webcam.")
                    break
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                timestamp_ms = max(timestamp_ms + 1, time.monotonic_ns() // 1_000_000)
                face_result = face_landmarker.detect_for_video(mp_image, timestamp_ms)
                hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

                for face in getattr(face_result, "face_landmarks", []) or []:
                    draw_face_landmarks(frame, face)
                for hand in getattr(hand_result, "hand_landmarks", []) or []:
                    draw_hand_landmarks(frame, hand)

                candidate_state = detect_current_state(face_result, hand_result)
                previous_state = current_state
                current_state = update_stable_state(history, candidate_state, current_state)
                streak = 0
                for item in reversed(history):
                    if item != candidate_state:
                        break
                    streak += 1
                cv2.putText(frame, f"Detected: {candidate_state} | Stable: {current_state} | Hold: {streak}/{STABLE_FRAMES}", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, GREEN if current_state == candidate_state else YELLOW, 2, cv2.LINE_AA)
                if current_state != previous_state:
                    print(f"MemeCV switched to: {MEME_TITLES[current_state]}")

                cv2.imshow("MemeCV - Camera", frame)
                cv2.imshow("MemeCV - Meme", render_meme_window(meme_images[current_state], current_state))
                if cv2.waitKey(1) & 0xFF == 27:
                    break
    finally:
        capture.release()
        cv2.destroyAllWindows()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MemeCV webcam meme detector")
    parser.add_argument("--camera", type=int, default=0, help="camera device index (default: 0)")
    return parser.parse_args()


def main() -> int:
    return run(camera_index=parse_args().camera)


if __name__ == "__main__":
    raise SystemExit(main())
