import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# --------------------------------------------------
# 1. MediaPipe Pose Landmarker setup
# --------------------------------------------------
# Load meme image with transparency.
meme = cv2.imread(
    "assets/fahh.png",
    cv2.IMREAD_UNCHANGED
)

if meme is None:
    print("ERROR: Could not load assets/fahh.png")
    exit()

model_path = "models/pose_landmarker_lite.task"

base_options = python.BaseOptions(
    model_asset_path=model_path
)

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

pose_detector = vision.PoseLandmarker.create_from_options(options)


# --------------------------------------------------
# 2. Open webcam
# --------------------------------------------------

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Could not open webcam.")
    exit()


# MediaPipe VIDEO mode needs timestamps.
timestamp_ms = 0

def overlay_png(background, overlay, x, y):
    """
    Places a transparent PNG on top of an OpenCV frame.
    """

    overlay_h, overlay_w = overlay.shape[:2]

    # Make sure the overlay fits inside the frame.
    if x < 0 or y < 0:
        return background

    if x + overlay_w > background.shape[1]:
        return background

    if y + overlay_h > background.shape[0]:
        return background

    # Separate RGB/BGR and alpha.
    overlay_rgb = overlay[:, :, :3]
    alpha = overlay[:, :, 3] / 255.0

    # Alpha has shape:
    # height × width
    #
    # We need:
    # height × width × 3
    alpha = np.expand_dims(alpha, axis=2)

    # Area of the camera where the meme goes.
    background_area = background[
        y:y + overlay_h,
        x:x + overlay_w
    ]

    # Blend the two images.
    background[
        y:y + overlay_h,
        x:x + overlay_w
    ] = (
        background_area * (1 - alpha)
        + overlay_rgb * alpha
    ).astype(np.uint8)

    return background
# --------------------------------------------------
# 3. Main loop
# --------------------------------------------------

while True:

    success, frame = camera.read()

    if not success:
        print("ERROR: Could not read webcam frame.")
        break

    # Mirror the camera like a selfie camera.
    frame = cv2.flip(frame, 1)

    # OpenCV uses BGR.
    # MediaPipe uses RGB.
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert OpenCV image into a MediaPipe image.
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Increase timestamp for every frame.
    timestamp_ms += 33

    # Run pose detection.
    result = pose_detector.detect_for_video(
        mp_image,
        timestamp_ms
    )

    # --------------------------------------------------
    # 4. Check whether a person was detected
    # --------------------------------------------------

    if result.pose_landmarks:

        landmarks = result.pose_landmarks[0]

        # MediaPipe Pose landmark indexes:
        # 0  = nose
        # 15 = left wrist
        # 16 = right wrist

        nose = landmarks[0]
        left_wrist = landmarks[15]
        right_wrist = landmarks[16]

        # Draw the landmarks.
        for landmark in landmarks:

            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])

            cv2.circle(
                frame,
                (x, y),
                4,
                (0, 255, 0),
                -1
            )

        # Draw the important points in a different color.

        for landmark in [nose, left_wrist, right_wrist]:

            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])

            cv2.circle(
                frame,
                (x, y),
                8,
                (0, 0, 255),
                -1
            )

        # --------------------------------------------------
        # 5. First gesture:
        #    Both hands above head
        # --------------------------------------------------

        hands_above_head = (
            left_wrist.y < nose.y - 0.04
            and
            right_wrist.y < nose.y - 0.04
        )

        if hands_above_head:

         print("FAHH GESTURE DETECTED!")

    # Use the PNG at its original size.
    meme_height, meme_width = meme.shape[:2]

    # Center the meme horizontally.
    x = (frame.shape[1] - meme_width) // 2
    y = 40

    # Make sure the meme fits inside the camera frame.
    if (
        x >= 0
        and y >= 0
        and x + meme_width <= frame.shape[1]
        and y + meme_height <= frame.shape[0]
    ):
        frame = overlay_png(
            frame,
            meme,
            x,
            y
        )
        
    # --------------------------------------------------
    # 6. Display
    # --------------------------------------------------

    cv2.imshow(
        "Gesture Meme - MediaPipe Test",
        frame
    )

    # Press Q to quit.
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# --------------------------------------------------
# 7. Cleanup
# --------------------------------------------------

camera.release()
cv2.destroyAllWindows()
pose_detector.close()