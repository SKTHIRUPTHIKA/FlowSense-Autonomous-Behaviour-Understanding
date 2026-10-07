import os
import shutil

# ==============================
# PATHS
# ==============================

base_source = r"C:\Users\rsani\OneDrive\Documents\FlowSense\01_raw_data\MOT17\MOT17\MOT17\train"

output = r"C:\Users\rsani\OneDrive\Documents\FlowSense\02_processed_data\MOT17"

output_images = os.path.join(output, "images")
output_labels = os.path.join(output, "labels")

os.makedirs(output_images, exist_ok=True)
os.makedirs(output_labels, exist_ok=True)


# ==============================
# PROCESS ALL SEQUENCES
# ==============================

sequence_folders = os.listdir(base_source)

total_images = 0
total_labels = 0
sequence_count = 0


for sequence in sequence_folders:

    source = os.path.join(base_source, sequence)

    if not os.path.isdir(source):
        continue

    images_folder = os.path.join(source, "img1")
    gt_file = os.path.join(source, "gt", "gt.txt")
    seqinfo_file = os.path.join(source, "seqinfo.ini")

    if not os.path.exists(images_folder):
        continue

    if not os.path.exists(gt_file):
        continue

    if not os.path.exists(seqinfo_file):
        continue

    # ==============================
    # READ IMAGE SIZE
    # ==============================

    image_width = 1920
    image_height = 1080

    with open(seqinfo_file, "r") as file:

        for line in file:

            line = line.strip()

            if line.startswith("imWidth="):
                image_width = int(line.split("=")[1])

            elif line.startswith("imHeight="):
                image_height = int(line.split("=")[1])


    # ==============================
    # READ GROUND TRUTH
    # ==============================

    annotations = {}

    with open(gt_file, "r") as file:

        for line in file:

            data = line.strip().split(",")

            if len(data) < 8:
                continue

            frame = int(data[0])
            class_id = int(data[7])

            # Keep only person
            if class_id != 1:
                continue

            x = float(data[2])
            y = float(data[3])
            width = float(data[4])
            height = float(data[5])

            # Ignore invalid boxes
            if width <= 0 or height <= 0:
                continue

            # Convert MOT format to YOLO format

            x_center = x + width / 2
            y_center = y + height / 2

            x_center = x_center / image_width
            y_center = y_center / image_height

            width = width / image_width
            height = height / image_height

            if frame not in annotations:
                annotations[frame] = []

            annotations[frame].append(
                f"0 {x_center:.6f} {y_center:.6f} "
                f"{width:.6f} {height:.6f}"
            )


    # ==============================
    # COPY IMAGES + LABELS
    # ==============================

    for frame, labels in annotations.items():

        image_name = f"{frame:06d}.jpg"

        source_image = os.path.join(images_folder, image_name)

        if not os.path.exists(source_image):
            continue

        # Add sequence name to prevent duplicate filenames
        output_name = f"{sequence}_{frame:06d}"

        destination_image = os.path.join(
            output_images,
            output_name + ".jpg"
        )

        destination_label = os.path.join(
            output_labels,
            output_name + ".txt"
        )

        shutil.copy2(
            source_image,
            destination_image
        )

        with open(destination_label, "w") as file:

            for label in labels:
                file.write(label + "\n")

        total_images += 1
        total_labels += 1


    sequence_count += 1

    print(
        "Processed:",
        sequence,
        "| Frames:",
        len(annotations)
    )


# ==============================
# FINAL RESULT
# ==============================

print()
print("--------------------------------")
print("MOT17 PROCESSING COMPLETE")
print("--------------------------------")
print("Sequences processed:", sequence_count)
print("Images processed:", total_images)
print("Labels created:", total_labels)
print("Images folder:", output_images)
print("Labels folder:", output_labels)