import csv
from ultralytics import YOLO
from collections import defaultdict
import math

VIDEO = r"C:\Users\rsani\OneDrive\Documents\FlowSense\08_demo_video\cctv_demo.mp4"
TRACKER = r"C:\Users\rsani\OneDrive\Documents\FlowSense\05_configs\botsort_reid.yaml"
CSV_FILE = r"C:\Users\rsani\OneDrive\Documents\FlowSense\07_results\behavior_events.csv"

model = YOLO("yolo26s.pt")

tracks = defaultdict(list)

results = model.track(
    source=VIDEO,
    tracker=TRACKER,
    classes=[0],
    conf=0.10,
    imgsz=960,
    stream=True
)

for frame_no, result in enumerate(results):

    if result.boxes.id is None:
        continue

    ids = result.boxes.id.cpu().tolist()
    boxes = result.boxes.xyxy.cpu().tolist()

    for person_id, box in zip(ids, boxes):

        x1, y1, x2, y2 = box

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        tracks[int(person_id)].append(
            (frame_no, cx, cy)
        )


FPS = 30

# Person must stay in approximately the same area
LOITER_TIME = 10

# Maximum allowed area movement in pixels
AREA_THRESHOLD = 120

events = []

for person_id, points in tracks.items():

    window_size = FPS * LOITER_TIME

    if len(points) < window_size:
        continue

    for i in range(window_size, len(points)):

        window = points[i - window_size:i]

        start_frame = window[0][0]
        end_frame = window[-1][0]

        start_x = window[0][1]
        start_y = window[0][2]

        max_distance = 0

        for point in window:

            x = point[1]
            y = point[2]

            distance = math.sqrt(
                (x - start_x) ** 2 +
                (y - start_y) ** 2
            )

            if distance > max_distance:
                max_distance = distance

        if max_distance <= AREA_THRESHOLD:

            start_seconds = start_frame / FPS
            end_seconds = end_frame / FPS

            start_minutes = int(start_seconds // 60)
            start_secs = int(start_seconds % 60)

            end_minutes = int(end_seconds // 60)
            end_secs = int(end_seconds % 60)

            start_time = f"{start_minutes:02d}:{start_secs:02d}"
            end_time = f"{end_minutes:02d}:{end_secs:02d}"

            events.append([
                person_id,
                "Loitering",
                start_time,
                end_time,
                round(end_seconds - start_seconds, 1),
                "ABNORMAL"
            ])

            break


with open(CSV_FILE, "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "Person_ID",
        "Behaviour",
        "Start_Time",
        "End_Time",
        "Duration_seconds",
        "Status"
    ])

    writer.writerows(events)


print()
print("Improved loitering analysis completed")
print("Persons observed:", len(tracks))
print("Loitering events:", len(events))
print()
print("CSV saved to:")
print(CSV_FILE)