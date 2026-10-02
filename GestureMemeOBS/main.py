import cv2
import json
import time
import threading
from pathlib import Path

import numpy as np

from flask import Flask, Response, render_template, jsonify

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from gestures import GestureManager


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

with open(
    BASE_DIR / "config.json",
    "r",
    encoding="utf-8"
) as file:

    config = json.load(file)


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)


# ============================================================
# GLOBAL STATE
# ============================================================

latest_frame = None

state = {
    "detection_enabled": True,
    "current_meme": None,
    "meme_until": 0,
    "meme_started": 0,
    "fps": 0
}

frame_lock = threading.Lock()


# ============================================================
# POSE LANDMARKER
# ============================================================

pose_model_path = (
    BASE_DIR
    / "models"
    / "pose_landmarker_lite.task"
)

if not pose_model_path.exists():

    raise FileNotFoundError(
        f"\nPose model not found:\n"
        f"{pose_model_path}\n"
    )


pose_base_options = python.BaseOptions(
    model_asset_path=str(
        pose_model_path
    )
)

pose_options = vision.PoseLandmarkerOptions(
    base_options=pose_base_options,

    running_mode=vision.RunningMode.VIDEO,

    num_poses=1,

    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

pose_detector = (
    vision.PoseLandmarker
    .create_from_options(
        pose_options
    )
)


# ============================================================
# FACE LANDMARKER
# ============================================================

face_model_path = (
    BASE_DIR
    / "models"
    / "face_landmarker.task"
)

if not face_model_path.exists():

    raise FileNotFoundError(
        f"\nFace model not found:\n"
        f"{face_model_path}\n\n"
        f"Download face_landmarker.task and put it "
        f"inside the models folder."
    )


face_base_options = python.BaseOptions(
    model_asset_path=str(
        face_model_path
    )
)

face_options = vision.FaceLandmarkerOptions(
    base_options=face_base_options,

    running_mode=vision.RunningMode.VIDEO,

    num_faces=1,

    output_face_blendshapes=True,

    output_facial_transformation_matrixes=False
)

face_detector = (
    vision.FaceLandmarker
    .create_from_options(
        face_options
    )
)


# ============================================================
# LOAD FAHH PNG
# ============================================================

fahh_path = (
    BASE_DIR
    / "assets"
    / "fahh.png"
)

fahh_meme = cv2.imread(
    str(fahh_path),
    cv2.IMREAD_UNCHANGED
)

if fahh_meme is None:

    raise FileNotFoundError(
        f"Could not load FAHH image:\n"
        f"{fahh_path}"
    )

if fahh_meme.shape[2] != 4:

    raise ValueError(
        "fahh.png must contain transparency (RGBA)."
    )


# ============================================================
# LOAD ROCK PNG
#
# Automatically removes white background.
# ============================================================

rock_path = (
    BASE_DIR
    / "assets"
    / "rock.png"
)

rock_meme = cv2.imread(
    str(rock_path),
    cv2.IMREAD_UNCHANGED
)

if rock_meme is None:

    raise FileNotFoundError(
        f"Could not load ROCK image:\n"
        f"{rock_path}"
    )


# ------------------------------------------------------------
# Convert ROCK image to BGRA
# ------------------------------------------------------------

if len(rock_meme.shape) == 2:

    rock_meme = cv2.cvtColor(
        rock_meme,
        cv2.COLOR_GRAY2BGRA
    )

elif rock_meme.shape[2] == 3:

    rock_meme = cv2.cvtColor(
        rock_meme,
        cv2.COLOR_BGR2BGRA
    )


# ------------------------------------------------------------
# Remove white background
# ------------------------------------------------------------

bgr = rock_meme[:, :, :3]

white_mask = (
    (bgr[:, :, 0] > 235)
    &
    (bgr[:, :, 1] > 235)
    &
    (bgr[:, :, 2] > 235)
)

rock_meme[white_mask, 3] = 0


# ============================================================
# GESTURE MANAGER
# ============================================================

gesture_manager = GestureManager(
    config
)


# ============================================================
# PNG OVERLAY
# ============================================================

def overlay_png(
    frame,
    png,
    center_x,
    center_y,
    width
):

    original_height, original_width = (
        png.shape[:2]
    )

    height = int(
        original_height
        * width
        / original_width
    )

    width = int(width)
    height = int(height)

    resized = cv2.resize(
        png,
        (width, height),
        interpolation=cv2.INTER_AREA
    )

    x = int(
        center_x
        - width / 2
    )

    y = int(
        center_y
        - height / 2
    )

    frame_h, frame_w = (
        frame.shape[:2]
    )

    x1 = max(
        0,
        x
    )

    y1 = max(
        0,
        y
    )

    x2 = min(
        frame_w,
        x + width
    )

    y2 = min(
        frame_h,
        y + height
    )

    if (
        x1 >= x2
        or
        y1 >= y2
    ):

        return frame

    overlay_x1 = x1 - x
    overlay_y1 = y1 - y

    overlay_x2 = (
        overlay_x1
        + (x2 - x1)
    )

    overlay_y2 = (
        overlay_y1
        + (y2 - y1)
    )

    cropped = resized[
        overlay_y1:overlay_y2,
        overlay_x1:overlay_x2
    ]

    rgb = cropped[:, :, :3]

    alpha = (
        cropped[:, :, 3:4]
        .astype(float)
        / 255.0
    )

    background = frame[
        y1:y2,
        x1:x2
    ].astype(float)

    result = (
        background
        * (1 - alpha)
        +
        rgb.astype(float)
        * alpha
    )

    frame[
        y1:y2,
        x1:x2
    ] = result.astype(
        "uint8"
    )

    return frame


# ============================================================
# CAMERA LOOP
# ============================================================

def camera_loop():

    global latest_frame

    camera = cv2.VideoCapture(
        config["camera_index"],
        cv2.CAP_DSHOW
    )

    if not camera.isOpened():

        print(
            "ERROR: Could not open webcam."
        )

        return

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        config["camera_width"]
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        config["camera_height"]
    )

    previous_time = time.monotonic()

    pose_timestamp_ms = 0
    face_timestamp_ms = 0

    while True:

        success, frame = camera.read()

        if not success:

            print(
                "ERROR: Could not read camera frame."
            )

            break

        # ====================================================
        # MIRROR CAMERA
        # ====================================================

        frame = cv2.flip(
            frame,
            1
        )

        # ====================================================
        # RGB
        # ====================================================

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        # ====================================================
        # MEDIAPIPE POSE
        # ====================================================

        pose_timestamp_ms += 33

        pose_result = (
            pose_detector
            .detect_for_video(
                mp_image,
                pose_timestamp_ms
            )
        )

        # ====================================================
        # MEDIAPIPE FACE
        # ====================================================

        face_timestamp_ms += 33

        face_result = (
            face_detector
            .detect_for_video(
                mp_image,
                face_timestamp_ms
            )
        )

        # ====================================================
        # GESTURE DETECTION
        # ====================================================

        if state["detection_enabled"]:

            fahh_triggered = False
            rock_triggered = False

            # ------------------------------------------------
            # FAHH
            # ------------------------------------------------

            if pose_result.pose_landmarks:

                landmarks = (
                    pose_result.pose_landmarks[0]
                )

                nose = landmarks[0]

                left_wrist = landmarks[15]
                right_wrist = landmarks[16]

                fahh_triggered = (
                    gesture_manager
                    .update_fahh(
                        left_wrist.y,
                        right_wrist.y,
                        nose.y
                    )
                )

            # ------------------------------------------------
            # ROCK
            # ------------------------------------------------

            if face_result.face_landmarks:

                face_landmarks = (
                    face_result.face_landmarks[0]
                )

                face_blendshapes = []

                if (
                    face_result.face_blendshapes
                ):

                    face_blendshapes = (
                        face_result
                        .face_blendshapes[0]
                    )

                rock_triggered = (
                    gesture_manager
                    .update_rock(
                        face_landmarks,
                        face_blendshapes
                    )
                )

            # ------------------------------------------------
            # TRIGGER FAHH
            # ------------------------------------------------

            if fahh_triggered:

                state["current_meme"] = "fahh"

                state["meme_started"] = (
                    time.monotonic()
                )

                state["meme_until"] = (
                    time.monotonic()
                    +
                    config[
                        "meme_duration_seconds"
                    ]
                )

                print(
                    "FAHH TRIGGERED!"
                )

            # ------------------------------------------------
            # TRIGGER ROCK
            # ------------------------------------------------

            elif rock_triggered:

                state["current_meme"] = "rock"

                state["meme_started"] = (
                    time.monotonic()
                )

                state["meme_until"] = (
                    time.monotonic()
                    +
                    config["gestures"]["rock"]["duration"]
                )

                print(
                    "ROCK TRIGGERED!"
                )

        # ====================================================
        # DISPLAY MEME
        # ====================================================

        if (
            state["current_meme"] is not None
            and
            time.monotonic()
            <
            state["meme_until"]
        ):

            # ------------------------------------------------
            # FAHH
            # ------------------------------------------------

            if (
                state["current_meme"]
                == "fahh"
                and
                pose_result.pose_landmarks
            ):

                landmarks = (
                    pose_result.pose_landmarks[0]
                )

                nose = landmarks[0]

                center_x = (
                    nose.x
                    *
                    frame.shape[1]
                )

                center_y = (
                    nose.y
                    *
                    frame.shape[0]
                    -
                    80
                )

                frame = overlay_png(
                    frame,
                    fahh_meme,
                    center_x,
                    center_y,
                    250
                )

            # ------------------------------------------------
            # ROCK
            # ------------------------------------------------

            elif (
                state["current_meme"]
                == "rock"
                and
                face_result.face_landmarks
            ):

                face_landmarks = (
                    face_result.face_landmarks[0]
                )

                nose = face_landmarks[1]

                center_x = (
                    nose.x
                    *
                    frame.shape[1]
                )

                center_y = (
                    nose.y
                    *
                    frame.shape[0]
                    -
                    100
                )

                frame = overlay_png(
                    frame,
                    rock_meme,
                    center_x,
                    center_y,
                    250
                )

        else:

            state["current_meme"] = None

        # ====================================================
        # FPS
        # ====================================================

        now = time.monotonic()

        delta = (
            now
            -
            previous_time
        )

        if delta > 0:

            state["fps"] = round(
                1 / delta,
                1
            )

        previous_time = now

        # ====================================================
        # SAVE FRAME
        # ====================================================

        with frame_lock:

            latest_frame = (
                frame.copy()
            )


    camera.release()


# ============================================================
# MJPEG STREAM
# ============================================================

def generate_frames():

    global latest_frame

    while True:

        with frame_lock:

            if latest_frame is None:

                continue

            frame = (
                latest_frame.copy()
            )

        success, encoded = cv2.imencode(
            ".jpg",
            frame,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                75
            ]
        )

        if not success:

            continue

        jpeg = encoded.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            +
            jpeg
            +
            b"\r\n"
        )


# ============================================================
# WEB ROUTES
# ============================================================

@app.route("/")
def index():

    return render_template(
        "obs.html"
    )


@app.route("/video")
def video():

    return Response(
        generate_frames(),
        mimetype=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        )
    )


@app.route("/state")
def get_state():

    return jsonify(
        state
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print(
        "======================================"
    )

    print(
        "       Gesture Meme OBS"
    )

    print(
        "======================================"
    )

    print()

    print(
        "Camera: ON"
    )

    print(
        "FAHH: Both hands above head"
    )

    print(
        "ROCK: Face turned + screen-left eyebrow raised"
    )

    print()

    print(
        "OBS URL:"
    )

    print(
        "http://127.0.0.1:5000/"
    )

    print()

    print(
        "Press Ctrl+C to stop."
    )

    print()

    thread = threading.Thread(
        target=camera_loop,
        daemon=True
    )

    thread.start()

    app.run(
        host=config["server"]["host"],
        port=config["server"]["port"],
        threaded=True,
        debug=False
    )