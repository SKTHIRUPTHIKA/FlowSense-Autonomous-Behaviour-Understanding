import os
import json
import shutil

base_source = r"C:\Users\rsani\OneDrive\Documents\FlowSense\01_raw_data\CrowdHuman"

output = r"C:\Users\rsani\OneDrive\Documents\FlowSense\02_processed_data\CrowdHuman"

output_images = os.path.join(output, "images")
output_labels = os.path.join(output, "labels")

os.makedirs(output_images, exist_ok=True)
os.makedirs(output_labels, exist_ok=True)

annotation_file = os.path.join(
    base_source,
    "annotation_train.odgt"
)

image_folders = [
    os.path.join(base_source, "train01", "Images"),
    os.path.join(base_source, "train02", "Images"),
    os.path.join(base_source, "train03", "Images")
]

image_width = 1920
image_height = 1080

total_images = 0
total_labels = 0
total_persons = 0

with open(annotation_file, "r") as file:

    for line in file:

        data = json.loads(line)

        image_id = data["ID"]

        image_name = image_id + ".jpg"

        source_image = None

        for folder in image_folders:

            possible_image = os.path.join(
                folder,
                image_name
            )

            if os.path.exists(possible_image):
                source_image = possible_image
                break

        if source_image is None:
            continue

        labels = []

        for box in data["gtboxes"]:

            if box.get("tag") != "person":
                continue

            if box.get("head_attr", {}).get("ignore", 0) == 1:
                continue

            fbox = box.get("fbox")

            if fbox is None:
                continue

            x = float(fbox[0])
            y = float(fbox[1])
            width = float(fbox[2])
            height = float(fbox[3])

            if width <= 0 or height <= 0:
                continue

            # Clip box to image boundaries

            if x < 0:
                width = width + x
                x = 0

            if y < 0:
                height = height + y
                y = 0

            if x + width > image_width:
                width = image_width - x

            if y + height > image_height:
                height = image_height - y

            if width <= 0 or height <= 0:
                continue

            x_center = (x + width / 2) / image_width
            y_center = (y + height / 2) / image_height

            width_normalized = width / image_width
            height_normalized = height / image_height

            labels.append(
                f"0 {x_center:.6f} {y_center:.6f} "
                f"{width_normalized:.6f} {height_normalized:.6f}"
            )

        if len(labels) == 0:
            continue

        destination_image = os.path.join(
            output_images,
            image_name
        )

        destination_label = os.path.join(
            output_labels,
            image_id + ".txt"
        )

        shutil.copy2(
            source_image,
            destination_image
        )

        with open(destination_label, "w") as label_file:

            for label in labels:
                label_file.write(label + "\n")

        total_images += 1
        total_labels += 1
        total_persons += len(labels)

print()
print("--------------------------------")
print("CROWDHUMAN PROCESSING COMPLETE")
print("--------------------------------")
print("Images processed:", total_images)
print("Labels created:", total_labels)
print("Person boxes:", total_persons)
print("Images folder:", output_images)
print("Labels folder:", output_labels)