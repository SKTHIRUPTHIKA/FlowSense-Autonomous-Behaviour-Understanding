import cv2
import math
from collections import defaultdict, deque
from ultralytics import YOLO

VIDEO = r"C:\Users\rsani\OneDrive\Documents\FlowSense\08_demo_video\cctv_demo.mp4"
TRACKER = r"C:\Users\rsani\OneDrive\Documents\FlowSense\05_configs\botsort_reid.yaml"

OUTPUT = r"C:\Users\rsani\OneDrive\Documents\FlowSense\07_results\final_demo_critical.mp4"

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

LOITER_RADIUS_METERS = 5.0

PERSON_HEIGHT_METERS = 1.7

LOITERING_TIME_THRESHOLD = 10

MOVEMENT_THRESHOLD = 3.0

ACTIVITY_WINDOW = 30

# Critical alert threshold
CRITICAL_RISK_THRESHOLD = 80


# --------------------------------------------------
# MODEL
# --------------------------------------------------

model = YOLO("yolo26s.pt")


# --------------------------------------------------
# VIDEO
# --------------------------------------------------

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


# --------------------------------------------------
# PERSON DATA
# --------------------------------------------------

positions = defaultdict(
    lambda: deque(maxlen=ACTIVITY_WINDOW)
)

area_anchor = {}

area_start_time = {}

person_behaviour = {}

person_risk = {}

person_activity = {}

person_duration = {}


# --------------------------------------------------
# TRACKING
# --------------------------------------------------

results = model.track(
    source=VIDEO,
    tracker=TRACKER,
    classes=[0],
    conf=0.10,
    imgsz=960,
    stream=True
)


frame_no = 0


for result in results:

    frame = result.orig_img.copy()

    current_time = frame_no / fps


    # --------------------------------------------------
    # COUNTERS
    # --------------------------------------------------

    total_persons = 0

    abnormal_count = 0

    critical_count = 0


    # --------------------------------------------------
    # CURRENT IDs
    # --------------------------------------------------

    current_ids = set()


    # --------------------------------------------------
    # PROCESS PEOPLE
    # --------------------------------------------------

    if result.boxes.id is not None:

        ids = result.boxes.id.cpu().tolist()

        boxes = result.boxes.xyxy.cpu().tolist()

        total_persons = len(ids)


        for person_id, box in zip(ids, boxes):

            person_id = int(person_id)

            current_ids.add(person_id)

            x1, y1, x2, y2 = map(
                int,
                box
            )


            # --------------------------------------------------
            # CENTRE POINT
            # --------------------------------------------------

            cx = int(
                (x1 + x2) / 2
            )

            cy = int(
                (y1 + y2) / 2
            )


            positions[person_id].append(
                (cx, cy)
            )


            # --------------------------------------------------
            # PERSON HEIGHT
            # --------------------------------------------------

            person_height_pixels = max(
                20,
                y2 - y1
            )


            # --------------------------------------------------
            # APPROXIMATE 5-METRE AREA
            # --------------------------------------------------

            radius_pixels = (
                person_height_pixels
                * LOITER_RADIUS_METERS
                / PERSON_HEIGHT_METERS
            )


            radius_pixels = max(
                100,
                min(
                    radius_pixels,
                    600
                )
            )


            # --------------------------------------------------
            # CREATE LOCAL AREA
            # --------------------------------------------------

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


            # --------------------------------------------------
            # DISTANCE FROM LOCAL AREA
            # --------------------------------------------------

            distance_from_anchor = math.sqrt(
                (cx - anchor_x) ** 2
                +
                (cy - anchor_y) ** 2
            )


            # --------------------------------------------------
            # PERSON LEFT LOCAL AREA
            # --------------------------------------------------

            if distance_from_anchor > radius_pixels:

                area_anchor[person_id] = (
                    cx,
                    cy
                )

                area_start_time[person_id] = (
                    current_time
                )

                anchor_x = cx

                anchor_y = cy


            # --------------------------------------------------
            # TIME INSIDE LOCAL AREA
            # --------------------------------------------------

            area_duration = (
                current_time
                -
                area_start_time[person_id]
            )


            # --------------------------------------------------
            # CURRENT MOVEMENT
            # --------------------------------------------------

            movement = 0

            points = positions[person_id]


            if len(points) >= 2:

                total_distance = 0


                for i in range(
                    1,
                    len(points)
                ):

                    px, py = points[i - 1]

                    nx, ny = points[i]


                    distance = math.sqrt(
                        (nx - px) ** 2
                        +
                        (ny - py) ** 2
                    )


                    total_distance += distance


                movement = (
                    total_distance
                    /
                    (len(points) - 1)
                )


            # --------------------------------------------------
            # CURRENT ACTIVITY
            # --------------------------------------------------

            if movement > MOVEMENT_THRESHOLD:

                activity = "WALKING"

            else:

                activity = "STANDING"


            # --------------------------------------------------
            # BEHAVIOUR
            # --------------------------------------------------

            if area_duration >= LOITERING_TIME_THRESHOLD:

                behaviour = "LOITERING"

                status = "ABNORMAL"

                abnormal_count += 1


                # ----------------------------------------------
                # DETERMINISTIC RISK
                # ----------------------------------------------

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


            # --------------------------------------------------
            # CRITICAL ALERT
            # --------------------------------------------------

            critical_alert = (
                risk > CRITICAL_RISK_THRESHOLD
            )


            if critical_alert:

                critical_count += 1


            # --------------------------------------------------
            # SAVE PERSON INFORMATION
            # --------------------------------------------------

            person_behaviour[person_id] = (
                behaviour
            )

            person_risk[person_id] = (
                risk
            )

            person_activity[person_id] = (
                activity
            )

            person_duration[person_id] = (
                area_duration
            )


            # --------------------------------------------------
            # BOX COLOUR
            # --------------------------------------------------

            if critical_alert:

                # CRITICAL = RED
                box_color = (
                    0,
                    0,
                    255
                )

            elif behaviour == "LOITERING":

                # ABNORMAL = RED
                box_color = (
                    0,
                    0,
                    255
                )

            elif behaviour == "WALKING":

                # WALKING = ORANGE
                box_color = (
                    255,
                    180,
                    0
                )

            else:

                # NORMAL = GREEN
                box_color = (
                    0,
                    255,
                    0
                )


            # --------------------------------------------------
            # DRAW PERSON BOX
            # --------------------------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                box_color,
                2
            )


            # --------------------------------------------------
            # PERSON LABEL
            # --------------------------------------------------

            label = (
                f"ID {person_id} | "
                f"{behaviour}"
            )


            cv2.putText(
                frame,
                label,
                (
                    x1,
                    max(
                        25,
                        y1 - 8
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )


            # --------------------------------------------------
            # CRITICAL ALERT ABOVE PERSON
            # --------------------------------------------------

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
                        max(
                            20,
                            y1 - 25
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2
                )


            # --------------------------------------------------
            # ACTIVITY TEXT
            # --------------------------------------------------

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


    # --------------------------------------------------
    # NORMAL COUNT
    # --------------------------------------------------

    normal_count = max(
        0,
        total_persons - abnormal_count
    )


    # --------------------------------------------------
    # TOP DASHBOARD
    # --------------------------------------------------

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


    # --------------------------------------------------
    # CRITICAL COUNT
    # --------------------------------------------------

    cv2.putText(
        frame,
        f"CRITICAL: {critical_count}",
        (610, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 255),
        2
    )


    # --------------------------------------------------
    # TIME
    # --------------------------------------------------

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


    # --------------------------------------------------
    # ABNORMAL BEHAVIOUR PANEL
    # --------------------------------------------------

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


    # --------------------------------------------------
    # ALERT PANEL
    # --------------------------------------------------

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


            # ----------------------------------------------
            # CRITICAL
            # ----------------------------------------------

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


            # ----------------------------------------------
            # NORMAL ABNORMAL
            # ----------------------------------------------

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


    # --------------------------------------------------
    # CRITICAL SYSTEM SIGNAL
    # --------------------------------------------------

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


    # --------------------------------------------------
    # FOOTER
    # --------------------------------------------------

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


    # --------------------------------------------------
    # WRITE FRAME
    # --------------------------------------------------

    out.write(frame)

    frame_no += 1


# --------------------------------------------------
# RELEASE
# --------------------------------------------------

cap.release()

out.release()


# --------------------------------------------------
# FINAL MESSAGE
# --------------------------------------------------

print()
print("======================================")
print("FlowSense Critical Alert Demo created")
print("======================================")
print()
print("Output:")
print(OUTPUT)
print()
print("YOLO detection: ACTIVE")
print("Tracking: BoT-SORT + ReID")
print("Behaviour: LOCAL-AREA TEMPORAL ANALYSIS")
print("Loitering area: approximately 5 metre equivalent")
print("Loitering threshold:", LOITERING_TIME_THRESHOLD, "seconds")
print("Movement threshold:", MOVEMENT_THRESHOLD)
print("Critical risk threshold:", CRITICAL_RISK_THRESHOLD)
print("Critical signal: ENABLED")
print("Trajectory lines: DISABLED")
print()