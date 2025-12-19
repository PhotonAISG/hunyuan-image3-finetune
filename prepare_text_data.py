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
Utility script to prepare text training data for HunyuanImage-3.0 multimodal training.

This script handles text-only conversation data and converts various formats
(Alpaca, ShareGPT, etc.) to the required JSONL format.

For image generation data preparation, use prepare_image_gen_data.py instead.

Supported formats:
1. Text-only: Standard conversation format for language modeling
2. Text+Image: Conversations with image inputs for VL understanding (placeholder)

Message types for HunyuanImage-3.0:
- "text": Standard text messages (user prompts, assistant responses)
- "joint_image": Messages that include conditional image inputs (for understanding)

Output format: JSONL where each line is a JSON object with messages containing
role, content, and type fields.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any


def create_message(role: str, content: str, message_type: str = "text") -> Dict[str, str]:
    """
    Create a message dict with role, content, and type.
    
    Args:
        role: Message role ("system", "user", or "assistant")
        content: Message content
        message_type: Message type ("text", "joint_image", or "gen_image")
    
    Returns:
        Message dict with role, content, and type
    """
    valid_types = ["text", "joint_image", "gen_image"]
    if message_type not in valid_types:
        raise ValueError(f"message_type must be one of {valid_types}, got {message_type}")
    
    return {
        "role": role,
        "content": content,
        "type": message_type
    }


def create_text_only_sample(
    messages: List[Dict[str, str]],
) -> Dict[str, Any]:
    """
    Create a text-only training sample.
    
    Args:
        messages: List of message dicts with 'role', 'content', and 'type'
                 Example: [
                     {"role": "system", "content": "You are a helpful assistant.", "type": "text"},
                     {"role": "user", "content": "What is Python?", "type": "text"},
                     {"role": "assistant", "content": "Python is...", "type": "text"}
                 ]
    
    Returns:
        Sample dict ready for training
    """
    # Ensure all messages have the 'type' field set to 'text'
    for message in messages:
        if 'type' not in message:
            message['type'] = 'text'
    
    return {
        "type": "text_only",
        "messages": messages
    }


def create_image_understanding_sample(
    messages: List[Dict[str, str]],
    image_paths: List[str],
) -> Dict[str, Any]:
    """
    Create a text+image sample for vision-language understanding.
    
    Note: This is a placeholder function. The image understanding pipeline
    is not yet fully implemented. For image generation, use prepare_image_gen_data.py.
    
    Args:
        messages: List of message dicts (can include image placeholders with 'type' field)
        image_paths: List of paths to input images
    
    Returns:
        Sample dict ready for training
    """
    # Ensure messages have appropriate 'type' field (text or joint_image)
    for message in messages:
        if 'type' not in message:
            message['type'] = 'text'
    
    return {
        "type": "image_understanding",
        "messages": messages,
        "images": image_paths
    }


def convert_alpaca_to_text_only(alpaca_data: List[Dict], system_prompt: str = None) -> List[Dict]:
    """
    Convert Alpaca-style data to text-only format.
    
    Alpaca format:
    {
        "instruction": "...",
        "input": "...",  (optional)
        "output": "..."
    }
    """
    samples = []
    for item in alpaca_data:
        instruction = item["instruction"]
        input_text = item.get("input", "")
        output = item["output"]
        
        # Combine instruction and input
        if input_text:
            user_message = f"{instruction}\n\n{input_text}"
        else:
            user_message = instruction
        
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt, "type": "text"})

        messages.extend([
            {"role": "user", "content": user_message, "type": "text"},
            {"role": "assistant", "content": output, "type": "text"}
        ])
        
        samples.append(create_text_only_sample(messages))
    
    return samples


def convert_sharegpt_to_text_only(sharegpt_data: List[Dict], system_prompt: str = None) -> List[Dict]:
    """
    Convert ShareGPT-style data to text-only format.
    
    ShareGPT format:
    {
        "conversations": [
            {"from": "human", "value": "..."},
            {"from": "gpt", "value": "..."},
            ...
        ]
    }
    """
    samples = []
    for item in sharegpt_data:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt, "type": "text"})

        for conv in item["conversations"]:
            role = "user" if conv["from"] in ["human", "user"] else "assistant"
            messages.append({
                "role": role,
                "content": conv["value"],
                "type": "text"
            })
        
        samples.append(create_text_only_sample(messages))
    
    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Prepare training data for HunyuanImage-3.0"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Input data file (JSON or JSONL)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Output JSONL file"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["text_only", "alpaca", "sharegpt", "custom"],
        default="text_only",
        help="Input data format"
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=None,
        help="The system prompt"
    )
    
    args = parser.parse_args()
    
    # Load input data
    input_path = Path(args.input_file)
    if input_path.suffix == ".jsonl":
        with open(input_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
    else:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    
    # Convert based on format
    if args.format == "alpaca":
        samples = convert_alpaca_to_text_only(data, args.system_prompt)
    elif args.format == "sharegpt":
        samples = convert_sharegpt_to_text_only(data, args.system_prompt)
    elif args.format == "text_only":
        # Assume already in correct format
        samples = data
    else:
        raise ValueError(f"Unknown format: {args.format}")
    
    # Write output
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"Converted {len(samples)} samples to {output_path}")


def create_example_data():
    """Create example text-only training data"""
    examples = []
    
    # Example 1: Text-only conversation with system prompt
    examples.append(create_text_only_sample([
        {"role": "system", "content": "You are a helpful AI assistant.", "type": "text"},
        {"role": "user", "content": "Explain quantum computing in simple terms.", "type": "text"},
        {"role": "assistant", "content": "Quantum computing is a type of computing that uses quantum mechanics...", "type": "text"}
    ]))
    
    # Example 2: Text-only without system prompt
    examples.append(create_text_only_sample([
        {"role": "user", "content": "Write a Python function to calculate factorial.", "type": "text"},
        {"role": "assistant", "content": "Here's a Python function:\n\n```python\ndef factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)\n```", "type": "text"}
    ]))
    
    # Save examples
    with open("example_train_data.jsonl", 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    print("Created example_train_data.jsonl with text-only sample data")
    print("\nNote: For image generation data, use prepare_image_gen_data.py")


if __name__ == "__main__":
    # Uncomment to create example data
    # create_example_data()
    
    # Run main conversion
    main()
