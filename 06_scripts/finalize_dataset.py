import os
import shutil
import random
from PIL import Image

ROOT = r"C:\Users\rsani\OneDrive\Documents\FlowSense"

ALL = os.path.join(ROOT, "03_split_data", "all")
FINAL = os.path.join(ROOT, "04_final_dataset")
CFG = os.path.join(ROOT, "05_configs")
RESULT = os.path.join(ROOT, "07_results")

IMG = os.path.join(ALL, "images")
LBL = os.path.join(ALL, "labels")

print("=== CHECKING COMBINED DATASET ===")

images = [
    f for f in os.listdir(IMG)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

labels = [
    f for f in os.listdir(LBL)
    if f.lower().endswith(".txt")
]

print("Images:", len(images))
print("Labels:", len(labels))

image_stems = {os.path.splitext(f)[0] for f in images}
label_stems = {os.path.splitext(f)[0] for f in labels}

missing_labels = image_stems - label_stems
missing_images = label_stems - image_stems

print("Missing labels:", len(missing_labels))
print("Missing images:", len(missing_images))

if missing_labels or missing_images:
    print("ERROR: IMAGE/LABEL MISMATCH")
    exit()

print("\n=== CREATING TRAIN / VAL / TEST ===")

for split in ["train", "val", "test"]:
    os.makedirs(
        os.path.join(FINAL, split, "images"),
        exist_ok=True
    )

    os.makedirs(
        os.path.join(FINAL, split, "labels"),
        exist_ok=True
    )

stems = list(image_stems)

random.seed(42)
random.shuffle(stems)

total = len(stems)

train_end = int(total * 0.70)
val_end = int(total * 0.90)

train = stems[:train_end]
val = stems[train_end:val_end]
test = stems[val_end:]

splits = {
    "train": train,
    "val": val,
    "test": test
}

for split, items in splits.items():

    print("\nCopying", split, ":", len(items))

    for count, stem in enumerate(items, 1):

        image_file = None

        for ext in [".jpg", ".jpeg", ".png"]:

            candidate = os.path.join(
                IMG,
                stem + ext
            )

            if os.path.exists(candidate):
                image_file = candidate
                break

        if image_file is None:
            continue

        shutil.copy2(
            image_file,
            os.path.join(
                FINAL,
                split,
                "images",
                os.path.basename(image_file)
            )
        )

        shutil.copy2(
            os.path.join(LBL, stem + ".txt"),
            os.path.join(
                FINAL,
                split,
                "labels",
                stem + ".txt"
            )
        )

        if count % 1000 == 0:
            print("  copied:", count)

print("\n=== FINAL QA ===")

image_errors = 0
label_errors = 0
invalid_boxes = 0

for split in ["train", "val", "test"]:

    image_folder = os.path.join(
        FINAL,
        split,
        "images"
    )

    label_folder = os.path.join(
        FINAL,
        split,
        "labels"
    )

    split_images = [
        f for f in os.listdir(image_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    split_labels = [
        f for f in os.listdir(label_folder)
        if f.lower().endswith(".txt")
    ]

    image_names = {
        os.path.splitext(f)[0]
        for f in split_images
    }

    label_names = {
        os.path.splitext(f)[0]
        for f in split_labels
    }

    missing_l = image_names - label_names
    missing_i = label_names - image_names

    image_errors += len(missing_l)
    label_errors += len(missing_i)

    corrupt = 0

    for filename in split_images:

        try:

            path = os.path.join(
                image_folder,
                filename
            )

            with Image.open(path) as image:
                image.verify()

        except:

            corrupt += 1

    image_errors += corrupt

    invalid = 0

    for filename in split_labels:

        path = os.path.join(
            label_folder,
            filename
        )

        with open(path, "r") as file:

            for line in file:

                parts = line.strip().split()

                if len(parts) != 5:
                    invalid += 1
                    continue

                if parts[0] != "0":
                    invalid += 1
                    continue

                try:

                    x = float(parts[1])
                    y = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])

                    if not (
                        0 <= x <= 1
                        and 0 <= y <= 1
                        and 0 < w <= 1
                        and 0 < h <= 1
                    ):
                        invalid += 1

                except:

                    invalid += 1

    invalid_boxes += invalid

    print("\n", split)
    print("Images:", len(split_images))
    print("Labels:", len(split_labels))
    print("Missing labels:", len(missing_l))
    print("Missing images:", len(missing_i))
    print("Corrupt images:", corrupt)
    print("Invalid labels:", invalid)

print("\n=== CREATING DATA.YAML ===")

os.makedirs(CFG, exist_ok=True)

yaml_file = os.path.join(
    CFG,
    "data.yaml"
)

yaml_content = f"""path: {FINAL.replace(chr(92), "/")}
train: train/images
val: val/images
test: test/images

names:
  0: person
"""

with open(
    yaml_file,
    "w",
    encoding="utf-8"
) as file:

    file.write(yaml_content)

print("Created:", yaml_file)

print("\n=== CREATING REPORT ===")

os.makedirs(RESULT, exist_ok=True)

report_file = os.path.join(
    RESULT,
    "FINAL_DATASET_REPORT.md"
)

with open(
    report_file,
    "w",
    encoding="utf-8"
) as file:

    file.write("# FlowSense Final Dataset Report\n\n")

    file.write(
        f"Total images: {total}\n\n"
    )

    file.write(
        f"Total labels: {len(labels)}\n\n"
    )

    file.write("Class: person (0)\n\n")

    file.write("Random seed: 42\n\n")

    file.write(
        f"Train: {len(train)}\n\n"
    )

    file.write(
        f"Validation: {len(val)}\n\n"
    )

    file.write(
        f"Test: {len(test)}\n\n"
    )

    file.write(
        f"Image issues: {image_errors}\n\n"
    )

    file.write(
        f"Label issues: {label_errors}\n\n"
    )

    file.write(
        f"Invalid boxes: {invalid_boxes}\n\n"
    )

    if (
        image_errors == 0
        and label_errors == 0
        and invalid_boxes == 0
    ):

        file.write("DATASET QA PASSED\n")

    else:

        file.write("DATASET QA FAILED\n")

print("\n===================================")
print("FLOWSENSE FINAL DATASET COMPLETE")
print("===================================")

print("Total images:", total)
print("Total labels:", len(labels))
print("Train:", len(train))
print("Validation:", len(val))
print("Test:", len(test))

print("Image issues:", image_errors)
print("Label issues:", label_errors)
print("Invalid boxes:", invalid_boxes)

if (
    image_errors == 0
    and label_errors == 0
    and invalid_boxes == 0
):

    print("\nDATASET QA PASSED")

else:

    print("\nDATASET QA FAILED")

print("\ndata.yaml:")
print(yaml_file)

print("\nReport:")
print(report_file)