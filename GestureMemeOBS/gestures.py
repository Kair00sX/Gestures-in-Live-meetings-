import time


class GestureManager:

    def __init__(self, config):

        self.config = config

        # =====================================================
        # FAHH SETTINGS
        # =====================================================

        self.hold_frames = config.get(
            "hold_frames",
            5
        )

        self.cooldown = config.get(
            "cooldown_seconds",
            2.5
        )

        self.hold_counter = 0
        self.last_trigger_time = 0

        # =====================================================
        # ROCK SETTINGS
        # =====================================================

        self.rock_hold_frames = config.get(
            "rock_hold_frames",
            4
        )

        self.rock_cooldown = config.get(
            "rock_cooldown_seconds",
            2.5
        )

        self.rock_face_turn_threshold = config.get(
            "rock_face_turn_threshold",
            0.12
        )

        self.rock_eyebrow_threshold = config.get(
            "rock_eyebrow_threshold",
            0.30
        )

        self.rock_hold_counter = 0
        self.last_rock_trigger_time = 0

    # =========================================================
    # FAHH COOLDOWN
    # =========================================================

    def can_trigger_fahh(self):

        return (
            time.monotonic()
            - self.last_trigger_time
            >= self.cooldown
        )

    # =========================================================
    # FAHH
    #
    # BOTH HANDS ABOVE HEAD
    # =========================================================

    def update_fahh(
        self,
        left_wrist_y,
        right_wrist_y,
        nose_y
    ):

        threshold = (
            nose_y
            - self.config.get(
                "head_margin",
                0.05
            )
        )

        hands_up = (
            left_wrist_y < threshold
            and
            right_wrist_y < threshold
        )

        if hands_up:
            self.hold_counter += 1
        else:
            self.hold_counter = 0

        if (
            self.hold_counter >= self.hold_frames
            and
            self.can_trigger_fahh()
        ):

            self.hold_counter = 0
            self.last_trigger_time = time.monotonic()

            return True

        return False

    # =========================================================
    # ROCK
    #
    # SCREEN-LEFT EYEBROW RAISED
    # +
    # FACE TURNED
    #
    # Because the webcam image is mirrored, we deliberately
    # use the opposite MediaPipe eyebrow landmark.
    # =========================================================

    def update_rock(
        self,
        face_landmarks,
        face_blendshapes
    ):

        if not face_landmarks:
            self.rock_hold_counter = 0
            return False

        # =====================================================
        # FACE TURN
        # =====================================================

        nose = face_landmarks[1]

        left_face = face_landmarks[234]
        right_face = face_landmarks[454]

        face_center_x = (
            left_face.x +
            right_face.x
        ) / 2

        face_width = abs(
            right_face.x -
            left_face.x
        )

        if face_width <= 0:
            self.rock_hold_counter = 0
            return False

        nose_offset = (
            nose.x -
            face_center_x
        ) / face_width

        face_turned = (
            abs(nose_offset)
            >= self.rock_face_turn_threshold
        )

        # =====================================================
        # EYEBROW
        #
        # We intentionally use the MediaPipe LEFT eyebrow
        # blendshape for the screen-right side because the
        # camera preview is mirrored.
        # =====================================================

        eyebrow_raised = False

        for category in face_blendshapes:

            name = category.category_name
            score = category.score

            if name == "browOuterUpLeft":

                if score >= self.rock_eyebrow_threshold:
                    eyebrow_raised = True

        # =====================================================
        # ROCK GESTURE
        # =====================================================

        rock_detected = (
            face_turned
            and
            eyebrow_raised
        )

        # =====================================================
        # HOLD
        # =====================================================

        if rock_detected:

            self.rock_hold_counter += 1

        else:

            self.rock_hold_counter = 0

        # =====================================================
        # COOLDOWN
        # =====================================================

        rock_can_trigger = (
            time.monotonic()
            -
            self.last_rock_trigger_time
            >= self.rock_cooldown
        )

        # =====================================================
        # TRIGGER
        # =====================================================

        if (
            self.rock_hold_counter
            >= self.rock_hold_frames
            and
            rock_can_trigger
        ):

            self.rock_hold_counter = 0

            self.last_rock_trigger_time = (
                time.monotonic()
            )

            print("ROCK GESTURE DETECTED!")

            return True

        return False