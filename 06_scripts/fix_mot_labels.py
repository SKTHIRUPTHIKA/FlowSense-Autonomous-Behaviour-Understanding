import os
from PIL import Image

ROOT = r"C:\Users\rsani\OneDrive\Documents\FlowSense"

SOURCES = [
    os.path.join(ROOT, "02_processed_data", "MOT17"),
    os.path.join(ROOT, "02_processed_data", "MOT20")
]

fixed = 0
checked = 0

for source in SOURCES:

    image_folder = os.path.join(source, "images")
    label_folder = os.path.join(source, "labels")

    if not os.path.exists(image_folder):
        continue

    print("\nChecking:", source)

    for label_name in os.listdir(label_folder):

        if not label_name.endswith(".txt"):
            continue

        label_path = os.path.join(label_folder, label_name)

        image_name = os.path.splitext(label_name)[0] + ".jpg"
        image_path = os.path.join(image_folder, image_name)

        if not os.path.exists(image_path):
            continue

        with Image.open(image_path) as im:
            image_width, image_height = im.size

        new_lines = []
        changed = False

        with open(label_path, "r") as file:

            for line in file:

                parts = line.strip().split()

                if len(parts) != 5:
                    continue

                class_id = parts[0]

                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

                # YOLO normalized -> pixel coordinates
                x = (x_center - width / 2) * image_width
                y = (y_center - height / 2) * image_height

                w = width * image_width
                h = height * image_height

                # Clip bounding box to image boundaries
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(image_width, x + w)
                y2 = min(image_height, y + h)

                new_w = x2 - x1
                new_h = y2 - y1

                # Remove boxes that become invalid
                if new_w <= 0 or new_h <= 0:
                    continue

                # Convert back to YOLO format
                new_x_center = (x1 + new_w / 2) / image_width
                new_y_center = (y1 + new_h / 2) / image_height
                new_width = new_w / image_width
                new_height = new_h / image_height

                new_line = (
                    f"{class_id} "
                    f"{new_x_center:.6f} "
                    f"{new_y_center:.6f} "
                    f"{new_width:.6f} "
                    f"{new_height:.6f}\n"
                )

                new_lines.append(new_line)

                if (
                    abs(new_x_center - x_center) > 0.000001
                    or abs(new_y_center - y_center) > 0.000001
                    or abs(new_width - width) > 0.000001
                    or abs(new_height - height) > 0.000001
                ):
                    changed = True

        if changed:

            with open(label_path, "w") as file:
                file.writelines(new_lines)

            fixed += 1

        checked += 1

        if checked % 2000 == 0:
            print("Checked:", checked)

print("\n================================")
print("MOT LABEL FIX COMPLETE")
print("================================")
print("Labels checked:", checked)
print("Labels corrected:", fixed)
print("Now all MOT boxes are clipped to image boundaries.")