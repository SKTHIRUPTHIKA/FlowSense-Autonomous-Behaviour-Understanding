import cv2
import math
from collections import defaultdict, deque
from ultralytics import YOLO

VIDEO = r"C:\Users\rsani\OneDrive\Documents\FlowSense\08_demo_video\cctv_demo.mp4"
TRACKER = r"C:\Users\rsani\OneDrive\Documents\FlowSense\05_configs\botsort_reid.yaml"

OUTPUT = r"C:\Users\rsani\OneDrive\Documents\FlowSense\07_results\final_demo_v4.mp4"

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

# Person must remain in approximately this area
LOITER_RADIUS_METERS = 5.0

# Approximate human height used to convert meters -> pixels
PERSON_HEIGHT_METERS = 1.7

# Time required inside the area
LOITERING_TIME_THRESHOLD = 10

# Movement threshold
MOVEMENT_THRESHOLD = 3.0

# Recent frames used for activity detection
ACTIVITY_WINDOW = 30


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

positions = defaultdict(lambda: deque(maxlen=ACTIVITY_WINDOW))

# Starting point of local area
area_anchor = {}

# When person entered current local area
area_start_time = {}

# Current behaviour
person_behaviour = {}

# Last seen time
last_seen = {}


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
    # CURRENT PERSON COUNT
    # --------------------------------------------------

    total_persons = 0
    abnormal_count = 0


    # IDs currently visible
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

            x1, y1, x2, y2 = map(int, box)


            # --------------------------------------------------
            # CENTRE
            # --------------------------------------------------

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            positions[person_id].append((cx, cy))

            last_seen[person_id] = current_time


            # --------------------------------------------------
            # PERSON HEIGHT IN PIXELS
            # --------------------------------------------------

            person_height_pixels = max(
                20,
                y2 - y1
            )


            # --------------------------------------------------
            # APPROXIMATE 5 METRE RADIUS
            #
            # 1.7m person ~= bounding-box height
            #
            # 5m radius ~= 5 / 1.7 times
            # person's height
            # --------------------------------------------------

            radius_pixels = (
                person_height_pixels *
                LOITER_RADIUS_METERS /
                PERSON_HEIGHT_METERS
            )


            # Prevent extreme radius
            radius_pixels = max(
                100,
                min(radius_pixels, 600)
            )


            # --------------------------------------------------
            # CREATE AREA ANCHOR
            # --------------------------------------------------

            if person_id not in area_anchor:

                area_anchor[person_id] = (
                    cx,
                    cy
                )

                area_start_time[person_id] = current_time


            anchor_x, anchor_y = area_anchor[person_id]


            # --------------------------------------------------
            # DISTANCE FROM LOCAL AREA
            # --------------------------------------------------

            distance_from_anchor = math.sqrt(
                (cx - anchor_x) ** 2 +
                (cy - anchor_y) ** 2
            )


            # --------------------------------------------------
            # IF PERSON LEAVES LOCAL AREA
            # --------------------------------------------------

            if distance_from_anchor > radius_pixels:

                # Start a new local area
                area_anchor[person_id] = (
                    cx,
                    cy
                )

                area_start_time[person_id] = current_time

                anchor_x = cx
                anchor_y = cy


            # --------------------------------------------------
            # TIME INSIDE LOCAL AREA
            # --------------------------------------------------

            area_duration = (
                current_time -
                area_start_time[person_id]
            )


            # --------------------------------------------------
            # CURRENT MOVEMENT
            # --------------------------------------------------

            movement = 0

            points = positions[person_id]

            if len(points) >= 2:

                total_distance = 0

                for i in range(1, len(points)):

                    px, py = points[i - 1]
                    nx, ny = points[i]

                    distance = math.sqrt(
                        (nx - px) ** 2 +
                        (ny - py) ** 2
                    )

                    total_distance += distance

                movement = (
                    total_distance /
                    (len(points) - 1)
                )


            # --------------------------------------------------
            # ACTIVITY
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

                # Deterministic risk
                extra_time = (
                    area_duration -
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


            person_behaviour[person_id] = behaviour


            # --------------------------------------------------
            # BOX COLOUR
            # --------------------------------------------------

            if behaviour == "LOITERING":

                box_color = (0, 0, 255)

            elif behaviour == "WALKING":

                box_color = (255, 180, 0)

            else:

                box_color = (0, 255, 0)


            # --------------------------------------------------
            # DRAW BOX
            # --------------------------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                box_color,
                2
            )


            # --------------------------------------------------
            # LABEL
            # --------------------------------------------------

            label = (
                f"ID {person_id} | "
                f"{behaviour}"
            )


            cv2.putText(
                frame,
                label,
                (x1, max(25, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )


            # --------------------------------------------------
            # SHOW ACTIVITY
            # --------------------------------------------------

            if behaviour == "LOITERING":

                activity_text = (
                    f"Activity: {activity}"
                )

                cv2.putText(
                    frame,
                    activity_text,
                    (x1, y2 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (0, 0, 255),
                    1
                )

                cv2.putText(
                    frame,
                    f"Risk: {risk}/100",
                    (x1, y2 + 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (0, 0, 255),
                    1
                )

            else:

                cv2.putText(
                    frame,
                    f"Activity: {activity}",
                    (x1, y2 + 18),
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
    # TIME
    # --------------------------------------------------

    minutes = int(current_time // 60)
    seconds = int(current_time % 60)

    timestamp = (
        f"{minutes:02d}:{seconds:02d}"
    )


    cv2.putText(
        frame,
        f"TIME: {timestamp}",
        (650, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


    # --------------------------------------------------
    # ABNORMAL ALERT PANEL
    # --------------------------------------------------

    panel_x = width - 390

    cv2.rectangle(
        frame,
        (panel_x, 110),
        (width - 10, 450),
        (15, 15, 15),
        -1
    )


    cv2.putText(
        frame,
        "ABNORMAL BEHAVIOUR ALERTS",
        (panel_x + 15, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 255),
        2
    )


    y = 175


    abnormal_people = []

    for person_id, behaviour in person_behaviour.items():

        if behaviour == "LOITERING":

            abnormal_people.append(
                person_id
            )


    if abnormal_people:

        for person_id in abnormal_people[:7]:

            duration = (
                current_time -
                area_start_time[person_id]
            )

            extra_time = (
                duration -
                LOITERING_TIME_THRESHOLD
            )

            risk = int(
                min(
                    100,
                    60 + extra_time * 4
                )
            )


            cv2.putText(
                frame,
                f"ID {person_id} | LOITERING",
                (panel_x + 15, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 0, 255),
                1
            )

            y += 23


            cv2.putText(
                frame,
                f"Duration: {duration:.1f}s  Risk: {risk}/100",
                (panel_x + 15, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                1
            )

            y += 30


    else:

        cv2.putText(
            frame,
            "No abnormal behaviour",
            (panel_x + 15, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 255, 0),
            1
        )


    # --------------------------------------------------
    # FOOTER
    # --------------------------------------------------

    cv2.putText(
        frame,
        "Activity: WALKING / STANDING",
        (20, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        1
    )


    # --------------------------------------------------
    # WRITE
    # --------------------------------------------------

    out.write(frame)

    frame_no += 1


# --------------------------------------------------
# RELEASE
# --------------------------------------------------

cap.release()
out.release()


print()
print("======================================")
print("FlowSense final demo v4 created")
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
print("Trajectory lines: DISABLED")
print()