# Copyright 2025 [Your Name or Organization]. All Rights Reserved.
#
# Licensed under the Tencent Hunyuan Community License Agreement (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanImage-3.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Pre-process images for HunyuanImage-3.0 training.

This script loads images, encodes them with the VAE, and saves the latents to disk.
This allows you to avoid the "weight should have at least three dimensions" error
that occurs when trying to cache images during dataset initialization with DeepSpeed.

Usage:
    python train/preprocess_images.py \
        --model_name_or_path Tencent-Hunyuan/HunyuanImage-3.0 \
        --train_data_file example_data/t2i/train.jsonl \
        --output_dir example_data/t2i/cached_latents \
        --image_ratio "1:1" \
        --image_base_size 1024 \
        --image_root_dir example_data/t2i/images

Then in your training command, use:
    --train_data_file example_data/t2i/train_with_cached.jsonl

The script will create a new JSONL file with paths updated to point to the cached latents.
"""

import os
import sys
import json
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hunyuan_image_3.hunyuan import HunyuanImage3ForCausalMM
from hunyuan_image_3.tokenizer_wrapper import ImageInfo
from hunyuan_image_3.image_processor import resize_and_crop


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-process images for training")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Path to the HunyuanImage-3.0 model"
    )
    parser.add_argument(
        "--train_data_file",
        type=str,
        required=True,
        help="Path to training data JSONL file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save cached latents"
    )
    parser.add_argument(
        "--image_ratio",
        type=str,
        default="1:1",
        help="Image ratio (e.g., '1:1', '16:9', '4:3')"
    )
    parser.add_argument(
        "--image_base_size",
        type=int,
        default=1024,
        help="Base image size for generation."
    )
    parser.add_argument(
        "--image_root_dir",
        type=str,
        default="",
        help="Root directory for resolving relative image paths"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use for encoding (cuda:0, cuda:1, cpu)"
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use bfloat16 precision"
    )
    
    return parser.parse_args()


def resolve_image_path(image_path: str, data_file: str, image_root_dir: str) -> str:
    """Resolve image path (handle both absolute and relative paths)"""
    if os.path.isabs(image_path):
        return image_path
    
    # Try relative to image_root_dir
    if image_root_dir:
        full_path = os.path.join(image_root_dir, image_path)
        if os.path.exists(full_path):
            return full_path
    
    # Try relative to data file directory
    data_dir = os.path.dirname(os.path.abspath(data_file))
    full_path = os.path.join(data_dir, image_path)
    if os.path.exists(full_path):
        return full_path
    
    # Return as-is and let it fail if not found
    return image_path


def raw_image_preprocess(image: Image.Image, image_processor, image_ratio: str):
    """Preprocess raw image for VAE encoding"""
    if isinstance(image_ratio, str):
        if image_ratio.startswith("<img_ratio_"):
            ratio_index = int(image_ratio.split("_")[-1].rstrip(">"))
            reso = image_processor.reso_group[ratio_index]
            image_ratio = reso.height, reso.width
        elif 'x' in image_ratio:
            image_ratio = [int(s) for s in image_ratio.split('x')]
        elif ':' in image_ratio:
            image_ratio = [int(s) for s in image_ratio.split(':')]
        else:
            raise ValueError(f"Invalid image_ratio: {image_ratio}")
        assert len(image_ratio) == 2
        image_width, image_height = image_processor.reso_group.get_target_size(image_ratio[1], image_ratio[0])
    elif isinstance(image_ratio, (list, tuple)):
        assert len(image_ratio) == 2 and all(isinstance(s, int) for s in image_ratio)
        image_width, image_height = image_processor.reso_group.get_target_size(image_ratio[1], image_ratio[0])
    else:
        image_width, image_height = image_processor.reso_group.get_target_size(image.width, image.height)
    
    resized_image = resize_and_crop(image, (image_width, image_height))
    image_tensor = image_processor.vae_processor(resized_image)
    token_height = image_height // (image_processor.config.vae_downsample_factor[0] * image_processor.config.patch_size)
    token_width = image_width // (image_processor.config.vae_downsample_factor[1] * image_processor.config.patch_size)
    base_size, ratio_index = image_processor.reso_group.get_base_size_and_ratio_index(width=image_width, height=image_height)
    
    return ImageInfo(
        image_type="vae",
        image_tensor=image_tensor.unsqueeze(0),
        image_width=image_width, image_height=image_height,
        token_width=token_width, token_height=token_height,
        base_size=base_size, ratio_index=ratio_index,
    )


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading model from {args.model_name_or_path}...")
    
    # Load model
    init_kwargs = {}
    if args.bf16:
        init_kwargs["torch_dtype"] = torch.bfloat16
    init_kwargs["image_base_size"] = args.image_base_size
    
    model = HunyuanImage3ForCausalMM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        **init_kwargs
    )
    
    device = torch.device(args.device)
    model.vae = model.vae.to(device)
    model.vae.eval()
    
    print(f"VAE loaded on {device}")
    
    # Load data
    print(f"Loading data from {args.train_data_file}...")
    with open(args.train_data_file, 'r', encoding='utf-8') as f:
        data_list = [json.loads(line) for line in f]
    
    # Find all unique images
    unique_images = {}  # path -> [indices]
    for idx, item in enumerate(data_list):
        if item.get('type') == 'image_generation' and 'target_image' in item:
            image_path = resolve_image_path(
                item['target_image'],
                args.train_data_file,
                args.image_root_dir
            )
            if image_path not in unique_images:
                unique_images[image_path] = []
            unique_images[image_path].append(idx)
    
    print(f"Found {len(unique_images)} unique images to process")
    
    # Process images
    latent_map = {}  # original_path -> cached_path
    failed_images = []
    
    for image_path in tqdm(unique_images.keys(), desc="Encoding images"):
        try:
            # Load and preprocess
            raw_image = Image.open(image_path).convert('RGB')
            vae_image_info = raw_image_preprocess(raw_image, model.image_processor, args.image_ratio)
            
            # Encode
            with torch.no_grad():
                _, vae_latent = model.vae_encode(
                    vae_image_info.image_tensor.to(device)
                )
            
            # Save latent
            # Create unique filename based on original path
            relative_path = os.path.relpath(image_path, os.path.dirname(args.train_data_file))
            safe_name = relative_path.replace(os.sep, '_').replace('/', '_').replace('\\', '_')
            latent_filename = f"{Path(safe_name).stem}.pt"
            latent_path = os.path.join(args.output_dir, latent_filename)
            
            # Save as torch tensor
            torch.save(vae_latent.cpu(), latent_path)
            
            latent_map[image_path] = latent_path
            
        except Exception as e:
            print(f"\nFailed to process {image_path}: {e}")
            failed_images.append(image_path)
    
    # Create new data file with cached paths
    output_data_file = args.train_data_file.replace('.jsonl', '_with_cached.jsonl')
    
    print(f"\nCreating updated data file: {output_data_file}")
    with open(output_data_file, 'w', encoding='utf-8') as f:
        for item in data_list:
            if item.get('type') == 'image_generation' and 'target_image' in item:
                original_path = resolve_image_path(
                    item['target_image'],
                    args.train_data_file,
                    args.image_root_dir
                )
                if original_path in latent_map:
                    # Update to point to cached latent
                    item['target_image'] = latent_map[original_path]
                    
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print("\n" + "=" * 80)
    print("Pre-processing complete!")
    print(f"Successfully processed: {len(latent_map)} images")
    print(f"Failed: {len(failed_images)} images")
    print(f"Cached latents saved to: {args.output_dir}")
    print(f"Updated data file: {output_data_file}")
    print("\nUse this data file in your training command:")
    print(f"  --train_data_file {output_data_file}")
    print("=" * 80)
    
    if failed_images:
        print("\nFailed images:")
        for img in failed_images:
            print(f"  - {img}")


if __name__ == "__main__":
    main()
