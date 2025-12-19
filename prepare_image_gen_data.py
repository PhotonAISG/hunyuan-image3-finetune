#!/usr/bin/env python3

# Copyright 2025 Pixo. All Rights Reserved.
#
# This file is licensed under the GNU Affero General Public License, Version 3 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.gnu.org/licenses/agpl-3.0.html
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Dataset preparation script for HunyuanImage-3.0 image generation training.

This script helps convert various image dataset formats into the required JSONL format
for training HunyuanImage-3.0 on image generation tasks.

Usage:
    python prepare_image_gen_data.py \
        --image_dir ./images \
        --output_file train.jsonl \
        --style instruct \
        --captions_file captions.csv

Caption File Format:
    The script expects a CSV file with two columns: 'file_name' and 'text'
    
    Example captions.csv:
        file_name,text
        image001.jpg,A beautiful sunset over the ocean
        image002.png,A cat sitting on a windowsill
        image003.jpg,Modern architecture with glass facade
    
    If no captions file is specified, the script will look for 'captions.csv'
    in the image directory. If not found, it will generate captions from filenames.
"""

import json
import os
import argparse
from pathlib import Path
from typing import List, Optional, Dict
import random
import csv


def load_captions_from_csv(csv_file: str) -> Dict[str, str]:
    """
    Load captions from CSV file with columns: file_name, text
    Returns a dictionary mapping filename to caption
    """
    captions_dict = {}
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        # Validate columns
        if 'file_name' not in reader.fieldnames or 'text' not in reader.fieldnames:
            raise ValueError(
                f"CSV file must have 'file_name' and 'text' columns. "
                f"Found columns: {reader.fieldnames}"
            )
        
        for row in reader:
            filename = row['file_name'].strip()
            caption = row['text'].strip()
            if filename and caption:
                captions_dict[filename] = caption
    
    return captions_dict



def generate_caption_from_filename(filename: str) -> str:
    """Generate a basic caption from filename"""
    # Remove extension and replace underscores/hyphens with spaces
    caption = Path(filename).stem
    caption = caption.replace('_', ' ').replace('-', ' ')
    # Capitalize first letter
    caption = caption.strip().capitalize()
    return caption


def create_instruct_messages(caption: str, system_prompt: Optional[str] = None) -> List[Dict]:
    """Create instruct-style messages"""
    if system_prompt is None:
        system_prompt = "You are an image generation assistant that creates images based on user descriptions."
    # Diverse user prompts, including direct caption use and various phrasings
    user_templates = [
        f"{caption}",
        f"Generate an image: {caption}",
        f"Create an image of {caption}",
        f"Please generate: {caption}",
        f"I need an image of {caption}",
        f"Can you make an image showing {caption}?",
        f"Draw a picture of {caption}.",
        f"Produce an image that depicts {caption}.",
        f"Show me what {caption} looks like.",
        f"Could you create a visual representation of {caption}?",
        f"Make an illustration of {caption}.",
        f"Paint a scene of {caption}.",
        f"Give me an image based on: {caption}",
        f"Imagine and generate: {caption}",
        f"Please provide an image for: {caption}",
        f"Depict the following: {caption}",
        f"Visualize this: {caption}",
        f"Picture this: {caption}",
        f"Render: {caption}",
        f"Show: {caption}",
        f"Turn this into an image: {caption}",
        f"How would {caption} look as an image?",
        f"Bring to life: {caption}",
        f"Illustrate: {caption}",
        f"Generate artwork for: {caption}",
        f"Transform this description into an image: {caption}",
        f"Imagine a scene: {caption}",
        f"Draw: {caption}",
        f"Show me: {caption}",
        f"Artistic rendering of: {caption}",
        f"Picture: {caption}",
    ]

    # Diverse assistant responses, mixing "image complete" style and "here is the image" style
    assistant_templates = [
        # "Image complete" style
        "Image generation complete.",
        "The requested artwork has been created.",
        "Your visual description has been brought to life.",
        "The scene is now illustrated.",
        "I've visualized your prompt.",
        "The depiction is ready.",
        "The image matching your description is finished.",
        "Your idea has been transformed into an image.",
        "The creative rendering is done.",
        "The picture is complete.",
        "The concept has been illustrated.",
        "The requested scene has been generated.",
        "Your prompt has been interpreted visually.",
        "The artwork is finished.",
        "The visual output is ready.",
        "I've completed the image.",
        "The artistic rendering is available.",
        "The scene you described has been painted.",
        "The illustration is done.",
        "The image is now ready for you.",
        "A new image has been generated.",
        "The requested visual is complete.",
        "The prompt has been turned into an image.",
        "The description has been brought to canvas.",
        "Your request has been fulfilled with this image.",
        "The visual representation is finished.",
        "The image based on your prompt is done.",
        "The scene has been created as described.",
        "The artwork inspired by your prompt is ready.",
        "The image is complete.",
        # "Here is the image" style
        "Sure, here is the image you requested.",
        "Here is the image based on your description.",
        "Certainly! Here is the generated image.",
        "Of course, here is the illustration.",
        "Here is the artwork you asked for.",
        "Here is your image.",
        "Here is the visual representation.",
        "Of course, here is the picture you described.",
        "Here is the scene you wanted.",
        "Here is the creative rendering.",
        "Here is the result of your prompt.",
        "Here is the depiction you requested.",
        "Here is the visual output.",
        "Here is the generated artwork.",
        "Sure, here is the completed image.",
        "Here is the illustration based on your prompt.",
        "Here is the image as described.",
        "Here is the artistic rendering.",
        "Here is the finished image.",
        "Here is the image for your request.",
    ]
    return [
        {"role": "system", "content": system_prompt, "type": "text"},
        {"role": "user", "content": random.choice(user_templates), "type": "text"},
        {"role": "assistant", "content": random.choice(assistant_templates), "type": "text"}
    ]


def create_pretrain_messages(caption: str) -> List[Dict]:
    """Create pretrain-style messages (simple user prompt)"""
    return [
        {"role": "user", "content": caption, "type": "text"}
    ]


def create_image_gen_dataset(
    image_dir: str,
    output_file: str,
    style: str = "instruct",
    captions_file: Optional[str] = None,
    system_prompt: Optional[str] = None,
    recursive: bool = False,
    csv_driven: bool = False,
):
    """
    Create image generation training dataset in JSONL format.
    
    Args:
        image_dir: Directory containing images
        output_file: Output JSONL file path
        style: Training style ("instruct" or "pretrain")
        captions_file: Optional CSV file with captions (columns: file_name, text).
                      If not provided, looks for 'captions.csv' in image_dir.
                      Falls back to filename-based captions if CSV not found.
        system_prompt: Custom system prompt for instruct style
        recursive: Recursively search subdirectories
        csv_driven: If True, generate JSONL strictly following CSV rows (allows duplicates)
    
    Note:
        All images in the dataset will use the same aspect ratio, which should be
        specified during training with the --image_ratio argument.
    """
    
    # Supported image formats
    image_exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    
    # CSV-driven mode: generate based on CSV rows
    if csv_driven:
        csv_path = captions_file if captions_file else os.path.join(image_dir, 'captions.csv')
        
        if not os.path.isfile(csv_path):
            raise ValueError(f"CSV-driven mode requires a captions CSV file. Not found: {csv_path}")
        
        print(f"CSV-driven mode: generating dataset from {csv_path}")
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        
        samples_created = 0
        missing_images = []
        skipped_rows = 0
        unique_images = set()
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            # Validate columns
            if 'file_name' not in reader.fieldnames or 'text' not in reader.fieldnames:
                raise ValueError(
                    f"CSV file must have 'file_name' and 'text' columns. "
                    f"Found columns: {reader.fieldnames}"
                )
            
            with open(output_file, 'w', encoding='utf-8') as out:
                for row_num, row in enumerate(reader, start=2):  # Start at 2 (header is row 1)
                    filename = row['file_name'].strip()
                    caption = row['text'].strip()
                    
                    if not filename or not caption:
                        skipped_rows += 1
                        continue
                    
                    # Check if image exists
                    img_path = os.path.join(image_dir, filename)
                    if not os.path.isfile(img_path):
                        if filename not in missing_images:  # Avoid duplicate warnings
                            missing_images.append(filename)
                        continue
                    
                    # Track unique images for statistics
                    unique_images.add(filename)
                    
                    # Build messages based on style
                    if style == "instruct":
                        messages = create_instruct_messages(caption, system_prompt)
                    elif style == "pretrain":
                        messages = create_pretrain_messages(caption)
                    else:
                        raise ValueError(f"Unknown style: {style}. Use 'instruct' or 'pretrain'")
                    
                    # Create sample
                    sample = {
                        "type": "image_generation",
                        "messages": messages,
                        "target_image": filename,
                    }
                    
                    # Write to file
                    out.write(json.dumps(sample, ensure_ascii=False) + '\n')
                    samples_created += 1
        
        print(f"\n✓ Created {samples_created} samples in {output_file}")
        print(f"  - {len(unique_images)} unique images")
        print(f"  - {samples_created - len(unique_images)} duplicate entries (same image, different captions)")
        if skipped_rows > 0:
            print(f"  - Skipped {skipped_rows} empty rows")
        if missing_images:
            print(f"  ⚠ Warning: {len(missing_images)} images referenced in CSV not found:")
            for img in missing_images[:10]:  # Show first 10
                print(f"      - {img}")
            if len(missing_images) > 10:
                print(f"      ... and {len(missing_images) - 10} more")
        print(f"  Style: {style}")
        print(f"  CSV file: {csv_path}")
        print(f"  Image directory: {image_dir}")
        print(f"\nYou can now train with:")
        print(f"  --train_data_file {output_file}")
        print(f"  --image_root_dir {image_dir}")
        print(f"  --sequence_template {style}")
        return
    
    # Original mode: generate based on image files
    # Find all images
    image_files = []
    if recursive:
        for root, dirs, files in os.walk(image_dir):
            for f in files:
                if Path(f).suffix.lower() in image_exts:
                    rel_path = os.path.relpath(os.path.join(root, f), image_dir)
                    image_files.append(rel_path)
    else:
        image_files = [
            f for f in os.listdir(image_dir)
            if Path(f).suffix.lower() in image_exts
        ]
    
    image_files.sort()  # Ensure consistent ordering
    
    print(f"Found {len(image_files)} images in {image_dir}")
    
    # Load captions from CSV if available
    captions_dict = {}
    csv_path = captions_file if captions_file else os.path.join(image_dir, 'captions.csv')
    
    if os.path.isfile(csv_path):
        try:
            captions_dict = load_captions_from_csv(csv_path)
            print(f"Loaded {len(captions_dict)} captions from {csv_path}")
            
            # Check coverage
            matched = sum(1 for img in image_files if img in captions_dict)
            if matched < len(image_files):
                print(f"  Warning: Only {matched}/{len(image_files)} images have captions in CSV")
                print(f"  Missing images will use filename-based captions")
        except Exception as e:
            print(f"  Warning: Failed to load captions CSV: {e}")
            print(f"  Falling back to filename-based captions")
            captions_dict = {}
    else:
        if captions_file:
            print(f"  Warning: Captions file not found: {csv_path}")
        else:
            print(f"  No captions.csv found in {image_dir}")
        print(f"  Using filename-based captions")
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    
    # Generate dataset
    samples_created = 0
    samples_with_csv_captions = 0
    samples_with_filename_captions = 0
    
    with open(output_file, 'w', encoding='utf-8') as out:
        for img_file in image_files:
            # Get caption from CSV or generate from filename
            if img_file in captions_dict:
                caption = captions_dict[img_file]
                samples_with_csv_captions += 1
            else:
                caption = generate_caption_from_filename(img_file)
                samples_with_filename_captions += 1
            
            # Build messages based on style
            if style == "instruct":
                messages = create_instruct_messages(caption, system_prompt)
            elif style == "pretrain":
                messages = create_pretrain_messages(caption)
            else:
                raise ValueError(f"Unknown style: {style}. Use 'instruct' or 'pretrain'")
            
            # Create sample
            sample = {
                "type": "image_generation",
                "messages": messages,
                "target_image": img_file,
            }
            
            # Note: image_ratio is set at dataset level (--image_ratio training argument)
            # All images in a dataset must use the same ratio
            
            # Write to file
            out.write(json.dumps(sample, ensure_ascii=False) + '\n')
            samples_created += 1
    
    print(f"\n✓ Created {samples_created} samples in {output_file}")
    print(f"  - {samples_with_csv_captions} samples with CSV captions")
    print(f"  - {samples_with_filename_captions} samples with filename-based captions")
    print(f"  Style: {style}")
    print(f"  Image directory: {image_dir}")
    print(f"\nYou can now train with:")
    print(f"  --train_data_file {output_file}")
    print(f"  --image_root_dir {image_dir}")
    print(f"  --sequence_template {style}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare image generation training data for HunyuanImage-3.0"
    )
    
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Directory containing training images"
    )
    
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Output JSONL file path"
    )
    
    parser.add_argument(
        "--style",
        type=str,
        default="instruct",
        choices=["instruct", "pretrain"],
        help="Training style (default: instruct)"
    )
    
    parser.add_argument(
        "--captions_file",
        type=str,
        default=None,
        help="CSV file with captions (columns: file_name, text). "
             "If not provided, looks for 'captions.csv' in image_dir. "
             "Falls back to filename-based captions if not found."
    )
    
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=None,
        help="Custom system prompt for instruct style"
    )
    
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search subdirectories"
    )
    
    parser.add_argument(
        "--csv_driven",
        action="store_true",
        help="Generate JSONL strictly following CSV rows (allows duplicates). "
             "Requires --captions_file or captions.csv in image_dir."
    )
    
    args = parser.parse_args()
    
    # Validate image directory
    if not os.path.isdir(args.image_dir):
        print(f"Error: Image directory not found: {args.image_dir}")
        return 1
    
    # Validate captions file if provided
    if args.captions_file and not os.path.isfile(args.captions_file):
        print(f"Error: Captions file not found: {args.captions_file}")
        return 1
    
    # Validate csv_driven mode requirements
    if args.csv_driven:
        csv_path = args.captions_file if args.captions_file else os.path.join(args.image_dir, 'captions.csv')
        if not os.path.isfile(csv_path):
            print(f"Error: CSV-driven mode requires a captions CSV file.")
            print(f"  Either provide --captions_file or ensure captions.csv exists in {args.image_dir}")
            return 1
    
    # Create dataset
    create_image_gen_dataset(
        image_dir=args.image_dir,
        output_file=args.output_file,
        style=args.style,
        captions_file=args.captions_file,
        system_prompt=args.system_prompt,
        recursive=args.recursive,
        csv_driven=args.csv_driven,
    )
    
    return 0


if __name__ == "__main__":
    exit(main())
