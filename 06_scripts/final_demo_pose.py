import cv2
import math
from collections import defaultdict, deque
from ultralytics import YOLO

# ============================================================
# PATHS
# ============================================================

VIDEO = r"C:\Users\rsani\OneDrive\Documents\FlowSense\08_demo_video\cctv_demo.mp4"

TRACKER = r"C:\Users\rsani\OneDrive\Documents\FlowSense\05_configs\botsort_reid.yaml"

OUTPUT = r"C:\Users\rsani\OneDrive\Documents\FlowSense\07_results\final_demo_pose.mp4"


# ============================================================
# SETTINGS
# ============================================================

LOITER_RADIUS_METERS = 5.0
PERSON_HEIGHT_METERS = 1.7

LOITERING_TIME_THRESHOLD = 10

# Pose activity settings
POSE_WINDOW = 15

# Walking thresholds are relative to person's body size
ANKLE_MOVEMENT_RATIO = 0.35
ANKLE_NET_MOVEMENT_RATIO = 0.08

# Critical alert
CRITICAL_RISK_THRESHOLD = 80


# ============================================================
# LOAD POSE MODEL
# ============================================================

model = YOLO("yolo26s-pose.pt")


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO)

fps = cap.get(cv2.CAP_PROP_FPS)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    OUTPUT,
    fourcc,
    fps,
    (width, height)
)


# ============================================================
# DATA STORAGE
# ============================================================

positions = defaultdict(
    lambda: deque(maxlen=30)
)

pose_history = defaultdict(
    lambda: deque(maxlen=POSE_WINDOW)
)

area_anchor = {}
area_start_time = {}

person_behaviour = {}
person_risk = {}
person_activity = {}
person_duration = {}


# ============================================================
# TRACKING + POSE
# ============================================================

results = model.track(
    source=VIDEO,
    tracker=TRACKER,
    classes=[0],
    conf=0.10,
    imgsz=960,
    stream=True
)


# ============================================================
# MAIN LOOP
# ============================================================

frame_no = 0

for result in results:

    frame = result.orig_img.copy()

    current_time = frame_no / fps

    total_persons = 0
    abnormal_count = 0
    critical_count = 0

    # ========================================================
    # PERSON DETECTION
    # ========================================================

    if result.boxes.id is not None:

        ids = result.boxes.id.cpu().tolist()

        boxes = result.boxes.xyxy.cpu().tolist()

        total_persons = len(ids)

        # ----------------------------------------------------
        # POSE KEYPOINTS
        # ----------------------------------------------------

        if result.keypoints is not None:

            keypoints = result.keypoints.xy.cpu().tolist()

        else:

            keypoints = [None] * len(ids)


        # ====================================================
        # EACH PERSON
        # ====================================================

        for index, (person_id, box) in enumerate(
            zip(ids, boxes)
        ):

            person_id = int(person_id)

            x1, y1, x2, y2 = map(int, box)

            # ------------------------------------------------
            # PERSON CENTER
            # ------------------------------------------------

            cx = int((x1 + x2) / 2)

            cy = int((y1 + y2) / 2)

            positions[person_id].append(
                (cx, cy)
            )


            # =================================================
            # LOITERING AREA
            # =================================================

            person_height_pixels = max(
                20,
                y2 - y1
            )

            radius_pixels = (
                person_height_pixels
                * LOITER_RADIUS_METERS
                / PERSON_HEIGHT_METERS
            )

            radius_pixels = max(
                100,
                min(radius_pixels, 600)
            )


            if person_id not in area_anchor:

                area_anchor[person_id] = (
                    cx,
                    cy
                )

                area_start_time[person_id] = (
                    current_time
                )


            anchor_x, anchor_y = (
                area_anchor[person_id]
            )


            distance_from_anchor = math.sqrt(

                (cx - anchor_x) ** 2

                +

                (cy - anchor_y) ** 2

            )


            # Person moved outside local area
            if distance_from_anchor > radius_pixels:

                area_anchor[person_id] = (
                    cx,
                    cy
                )

                area_start_time[person_id] = (
                    current_time
                )


            area_duration = (
                current_time
                -
                area_start_time[person_id]
            )


            # =================================================
            # POSE ACTIVITY DETECTION
            # =================================================

            activity = "STANDING"

            current_pose = keypoints[index]


            if current_pose is not None:

                # COCO pose keypoints:
                # Left knee  = 13
                # Right knee = 14
                # Left ankle = 15
                # Right ankle = 16

                if len(current_pose) >= 17:

                    left_knee = current_pose[13]
                    right_knee = current_pose[14]

                    left_ankle = current_pose[15]
                    right_ankle = current_pose[16]


                    # -----------------------------------------
                    # Check keypoint visibility by coordinates
                    # -----------------------------------------

                    if (
                        left_ankle[0] > 0
                        and left_ankle[1] > 0
                        and right_ankle[0] > 0
                        and right_ankle[1] > 0
                    ):

                        pose_history[person_id].append(
                            (
                                left_knee,
                                right_knee,
                                left_ankle,
                                right_ankle
                            )
                        )


            # =================================================
            # ANALYSE RECENT POSE HISTORY
            # =================================================

            if len(pose_history[person_id]) >= POSE_WINDOW:

                history = list(
                    pose_history[person_id]
                )

                total_ankle_movement = 0

                total_knee_movement = 0

                first_left_ankle = history[0][2]
                first_right_ankle = history[0][3]

                last_left_ankle = history[-1][2]
                last_right_ankle = history[-1][3]


                # ---------------------------------------------
                # Calculate ankle movement
                # ---------------------------------------------

                for i in range(1, len(history)):

                    previous = history[i - 1]
                    current = history[i]


                    # Left ankle
                    total_ankle_movement += math.sqrt(

                        (current[2][0] - previous[2][0]) ** 2

                        +

                        (current[2][1] - previous[2][1]) ** 2

                    )


                    # Right ankle
                    total_ankle_movement += math.sqrt(

                        (current[3][0] - previous[3][0]) ** 2

                        +

                        (current[3][1] - previous[3][1]) ** 2

                    )


                    # Left knee
                    total_knee_movement += math.sqrt(

                        (current[0][0] - previous[0][0]) ** 2

                        +

                        (current[0][1] - previous[0][1]) ** 2

                    )


                    # Right knee
                    total_knee_movement += math.sqrt(

                        (current[1][0] - previous[1][0]) ** 2

                        +

                        (current[1][1] - previous[1][1]) ** 2

                    )


                # ---------------------------------------------
                # Net ankle movement
                # ---------------------------------------------

                left_net = math.sqrt(

                    (last_left_ankle[0] - first_left_ankle[0]) ** 2

                    +

                    (last_left_ankle[1] - first_left_ankle[1]) ** 2

                )


                right_net = math.sqrt(

                    (last_right_ankle[0] - first_right_ankle[0]) ** 2

                    +

                    (last_right_ankle[1] - first_right_ankle[1]) ** 2

                )


                net_ankle_movement = (
                    left_net + right_net
                )


                # ---------------------------------------------
                # Normalize using person's height
                # ---------------------------------------------

                body_height = max(
                    50,
                    y2 - y1
                )


                movement_ratio = (
                    total_ankle_movement
                    /
                    body_height
                )


                net_ratio = (
                    net_ankle_movement
                    /
                    body_height
                )


                # ---------------------------------------------
                # WALKING DECISION
                # ---------------------------------------------

                if (
                    movement_ratio
                    >
                    ANKLE_MOVEMENT_RATIO
                    and
                    net_ratio
                    >
                    ANKLE_NET_MOVEMENT_RATIO
                ):

                    activity = "WALKING"

                else:

                    activity = "STANDING"


            # =================================================
            # LOITERING / NORMAL
            # =================================================

            if (
                area_duration
                >=
                LOITERING_TIME_THRESHOLD
            ):

                behaviour = "LOITERING"

                status = "ABNORMAL"

                abnormal_count += 1

                extra_time = (
                    area_duration
                    -
                    LOITERING_TIME_THRESHOLD
                )

                risk = int(
                    min(
                        100,
                        60 + extra_time * 4
                    )
                )

            else:

                behaviour = activity

                status = "NORMAL"

                risk = 10


            # =================================================
            # CRITICAL ALERT
            # =================================================

            critical_alert = (
                risk
                >
                CRITICAL_RISK_THRESHOLD
            )


            if critical_alert:

                critical_count += 1


            # =================================================
            # STORE INFORMATION
            # =================================================

            person_behaviour[person_id] = behaviour

            person_risk[person_id] = risk

            person_activity[person_id] = activity

            person_duration[person_id] = area_duration


            # =================================================
            # BOX COLOR
            # =================================================

            if critical_alert:

                box_color = (0, 0, 255)

            elif behaviour == "LOITERING":

                box_color = (0, 0, 255)

            elif activity == "WALKING":

                box_color = (255, 180, 0)

            else:

                box_color = (0, 255, 0)


            # =================================================
            # DRAW BOX
            # =================================================

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                box_color,
                2
            )


            # =================================================
            # ID + BEHAVIOUR
            # =================================================

            label = (
                f"ID {person_id} | "
                f"{behaviour}"
            )


            cv2.putText(

                frame,

                label,

                (
                    x1,
                    max(25, y1 - 8)
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.55,

                (255, 255, 255),

                2

            )


            # =================================================
            # CRITICAL ALERT
            # =================================================

            if critical_alert:

                alert_top = max(
                    0,
                    y1 - 58
                )

                alert_bottom = max(
                    0,
                    y1 - 5
                )


                cv2.rectangle(

                    frame,

                    (
                        x1,
                        alert_top
                    ),

                    (
                        x2,
                        alert_bottom
                    ),

                    (0, 0, 255),

                    -1

                )


                cv2.putText(

                    frame,

                    "CRITICAL ALERT",

                    (
                        x1 + 5,
                        max(20, y1 - 25)
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.55,

                    (255, 255, 255),

                    2

                )


            # =================================================
            # ACTIVITY / RISK
            # =================================================

            if behaviour == "LOITERING":

                cv2.putText(

                    frame,

                    f"Activity: {activity}",

                    (
                        x1,
                        y2 + 18
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.42,

                    (0, 0, 255),

                    1

                )


                cv2.putText(

                    frame,

                    f"Risk: {risk}/100",

                    (
                        x1,
                        y2 + 35
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.42,

                    (0, 0, 255),

                    1

                )

            else:

                cv2.putText(

                    frame,

                    f"Activity: {activity}",

                    (
                        x1,
                        y2 + 18
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.42,

                    (255, 255, 255),

                    1

                )


    # ========================================================
    # COUNTS
    # ========================================================

    normal_count = max(
        0,
        total_persons - abnormal_count
    )


    # ========================================================
    # TOP DASHBOARD
    # ========================================================

    cv2.rectangle(

        frame,

        (0, 0),

        (width, 105),

        (20, 20, 20),

        -1

    )


    cv2.putText(

        frame,

        "FLOWSENSE - AI BEHAVIOUR INTELLIGENCE",

        (20, 30),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.8,

        (0, 255, 255),

        2

    )


    cv2.putText(

        frame,

        f"TOTAL PERSONS: {total_persons}",

        (20, 65),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.55,

        (255, 255, 255),

        2

    )


    cv2.putText(

        frame,

        f"NORMAL: {normal_count}",

        (260, 65),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.55,

        (0, 255, 0),

        2

    )


    cv2.putText(

        frame,

        f"ABNORMAL: {abnormal_count}",

        (430, 65),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.55,

        (0, 0, 255),

        2

    )


    cv2.putText(

        frame,

        f"CRITICAL: {critical_count}",

        (610, 65),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.55,

        (0, 0, 255),

        2

    )


    # ========================================================
    # TIME
    # ========================================================

    minutes = int(
        current_time // 60
    )

    seconds = int(
        current_time % 60
    )

    timestamp = (
        f"{minutes:02d}:{seconds:02d}"
    )


    cv2.putText(

        frame,

        f"TIME: {timestamp}",

        (800, 65),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.55,

        (255, 255, 255),

        2

    )


    # ========================================================
    # ABNORMAL PANEL
    # ========================================================

    panel_x = width - 390


    cv2.rectangle(

        frame,

        (panel_x, 110),

        (width - 10, 470),

        (15, 15, 15),

        -1

    )


    cv2.putText(

        frame,

        "ABNORMAL BEHAVIOUR ALERTS",

        (
            panel_x + 15,
            140
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.55,

        (0, 0, 255),

        2

    )


    y = 175

    abnormal_people = []


    for person_id, behaviour in (
        person_behaviour.items()
    ):

        if behaviour == "LOITERING":

            abnormal_people.append(
                person_id
            )


    if abnormal_people:

        for person_id in abnormal_people[:7]:

            risk = person_risk.get(
                person_id,
                10
            )

            duration = person_duration.get(
                person_id,
                0
            )

            activity = person_activity.get(
                person_id,
                "UNKNOWN"
            )


            if risk > CRITICAL_RISK_THRESHOLD:

                cv2.putText(

                    frame,

                    f"ID {person_id} | CRITICAL",

                    (
                        panel_x + 15,
                        y
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.48,

                    (0, 0, 255),

                    2

                )

                y += 22


                cv2.putText(

                    frame,

                    f"LOITERING | Activity: {activity}",

                    (
                        panel_x + 15,
                        y
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.40,

                    (255, 255, 255),

                    1

                )

                y += 22


                cv2.putText(

                    frame,

                    f"Duration: {duration:.1f}s  Risk: {risk}/100",

                    (
                        panel_x + 15,
                        y
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.40,

                    (0, 0, 255),

                    1

                )

                y += 30


            else:

                cv2.putText(

                    frame,

                    f"ID {person_id} | LOITERING",

                    (
                        panel_x + 15,
                        y
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.48,

                    (0, 0, 255),

                    1

                )

                y += 22


                cv2.putText(

                    frame,

                    f"Activity: {activity}",

                    (
                        panel_x + 15,
                        y
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.40,

                    (255, 255, 255),

                    1

                )

                y += 22


                cv2.putText(

                    frame,

                    f"Duration: {duration:.1f}s  Risk: {risk}/100",

                    (
                        panel_x + 15,
                        y
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.40,

                    (255, 255, 255),

                    1

                )

                y += 30


    else:

        cv2.putText(

            frame,

            "No abnormal behaviour",

            (
                panel_x + 15,
                y
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.48,

            (0, 255, 0),

            1

        )


    # ========================================================
    # CRITICAL HIGH-RISK SIGNAL
    # ========================================================

    if critical_count > 0:

        cv2.rectangle(

            frame,

            (
                panel_x,
                480
            ),

            (
                width - 10,
                530
            ),

            (0, 0, 255),

            -1

        )


        cv2.putText(

            frame,

            "!!! CRITICAL HIGH-RISK SIGNAL !!!",

            (
                panel_x + 15,
                512
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.50,

            (255, 255, 255),

            2

        )


    # ========================================================
    # LEGEND
    # ========================================================

    cv2.putText(

        frame,

        "Activity: WALKING / STANDING",

        (
            20,
            height - 20
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.5,

        (0, 255, 255),

        1

    )


    # ========================================================
    # WRITE FRAME
    # ========================================================

    out.write(frame)

    frame_no += 1


# ============================================================
# RELEASE
# ============================================================

cap.release()

out.release()


# ============================================================
# RESULT
# ============================================================

print()
print("======================================")
print("FlowSense Pose Activity Demo created")
print("======================================")
print()

print("Output:")
print(OUTPUT)

print()

print("YOLO Pose detection: ACTIVE")
print("Tracking: BoT-SORT + ReID")
print("Activity: POSE KEYPOINT ANALYSIS")
print("Activity window:", POSE_WINDOW, "frames")
print("Walking detection: ANKLES + KNEES")
print("Loitering: ACTIVE")
print("Critical signal: ENABLED")
print("Trajectory lines: DISABLED")
print()