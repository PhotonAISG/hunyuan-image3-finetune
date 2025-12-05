from datasets import load_dataset
import os
import json
from PIL import Image
from tqdm import tqdm
import csv

# Load the dataset
ds = load_dataset("lambdalabs/naruto-blip-captions", split="train")

# Create output directories
output_image_dir = "naruto_images"
os.makedirs(output_image_dir, exist_ok=True)
captions_csv = "captions.csv"

# Prepare JSONL file and captions.csv
with open(captions_csv, "w", encoding="utf-8", newline="") as csv_f:
    csv_writer = csv.writer(csv_f)
    csv_writer.writerow(["file_name", "text"])
    for i, item in tqdm(enumerate(ds), total=len(ds)):
        # Save image
        img = item["image"]
        img_filename = f"{i:06d}.png"
        img_path = os.path.join(output_image_dir, img_filename)
        img.save(img_path)

        # Write to captions.csv
        csv_writer.writerow([img_filename, item["text"].strip()])
