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
Dataset classes for HunyuanImage-3.0 multimodal training.

This module contains:
- MultiModalDataset: Main dataset class for text, image understanding, and image generation
- MultiModalDataCollator: Custom collator for batching with proper padding and attention masks
- Helper functions for data loading and tokenizer patching

Image Loading Modes:
--------------------
The dataset supports two ways to load images for training:

1. On-the-fly encoding (default):
   - Images are loaded and encoded with VAE during training
   - Works reliably with DeepSpeed but slower (VAE encoding every epoch)
   - Use when: Quick setup, small datasets, or debugging

2. Pre-cached latents (Optional):
   - Run preprocess_images.py to encode images ahead of time
   - Saves encoded latents as .pt files on disk
   - Dataset automatically detects .pt files and loads them directly
   - Use when: Large datasets, DeepSpeed training, maximum speed
   
Example workflow with pre-caching:
    # Step 1: Pre-process images (run once)
    python train/preprocess_images.py --model_name_or_path MODEL_PATH \
        --train_data_file data.jsonl --output_dir cached_latents
    
    # Step 2: Train with cached latents (use generated data_with_cached.jsonl)
    python train/train_multimodal.py --train_data_file data_with_cached.jsonl ...
"""

import os
import sys
import json
import types
import logging
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Union, Optional, Dict, Any, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image

from hunyuan_image_3.hunyuan import HunyuanImage3ForCausalMM, build_batch_2d_rope
from hunyuan_image_3.tokenizer_wrapper import TokenizerWrapper, ImageInfo
from hunyuan_image_3.image_processor import resize_and_crop
    
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
IGNORE_INDEX = -100

# Type aliases
BatchRaggedImages = Union[torch.Tensor, List[Union[torch.Tensor, List[torch.Tensor]]]]
BatchRaggedTensor = Union[torch.Tensor, List[torch.Tensor]]


def patch_tokenizer_wrapper_for_training(tokenizer_wrapper):
    """
    Monkey-patch TokenizerWrapper.apply_general_template to support add_eos parameter.
    
    The original implementation calls encode_general with add_eos=False hardcoded.
    For training, we need the ability to control this parameter.
    
    This patches the method to accept and pass through an add_eos parameter.
    """
    
    # Save reference to original method
    original_apply_general_template = tokenizer_wrapper.apply_general_template
    
    # Create patched version
    def patched_apply_general_template(
            self,
            message_list,
            max_length=None,
            add_assistant_prefix=False,
            answer="auto",
            bot_task="auto",
            sequence_template="instruct",
            uncond_p=0.0,
            cfg_factor=1,
            batchify=False,
            image_base_size=1024,
            drop_think=False,
            add_eos='auto',  # <-- ADDED PARAMETER (default 'auto' to match encode_general)
    ):
        """
        Patched version of apply_general_template with add_eos parameter support.
        
        Args:
            add_eos: 'auto', True, or False
                - 'auto': Add EOS only if max_length not reached and last token is not EOS
                - True: Always add EOS token
                - False: Never add EOS token (original behavior)
        """
        # If using batchify, handle recursively
        if batchify:
            assert isinstance(message_list[0], list), \
                f"When batchify is True, message_list should be a list of list, but got [{type(message_list[0])}, ...]."
            return self.batch_gen_infer(
                infer_fn=self.apply_general_template,
                prompt_list=[[]],
                infer_fn_kwargs_list=[dict(
                    message_list=message_list_i,
                    max_length=max_length,
                    add_assistant_prefix=add_assistant_prefix,
                    answer=answer,
                    bot_task=bot_task,
                    sequence_template=sequence_template,
                    image_base_size=image_base_size,
                    drop_think=drop_think,
                    add_eos=add_eos,
                ) for message_list_i in message_list],
                do_classifier_free_guidance=cfg_factor > 1,
                condition_repeat_times=1,
                uncondition_repeat_times=cfg_factor - 1,
            )
        
        # Call original implementation to process messages and build sections
        # We'll do this by temporarily calling the original with add_eos=False,
        # then manually calling encode_general with our desired add_eos value
        
        # Import dependencies from tokenizer_wrapper module
        from hunyuan_image_3.tokenizer_wrapper import Conversation
        from copy import deepcopy
        
        conv = Conversation()
        uncond_kwargs = dict(uncond_enabled=uncond_p == 1.0, uncond_p=uncond_p)

        def process_successive_message(_message_list, _cur_message_idx, role, prefix, suffix,
                                       answer_prefix="", answer_suffix=""):
            _sub_sections = []
            while _cur_message_idx < len(_message_list) and _message_list[_cur_message_idx]['role'] == role:
                message = _message_list[_cur_message_idx]
                if message['type'] == 'text':
                    text = message['content']
                    if role == "system":
                        _sub_sections.append(dict(type="text", text=text))
                    elif role == "assistant":
                        if ("<recaption>" in text and "</recaption>" in text) or (
                                "<think>" in text and "</think>" in text):
                            _sub_sections.extend(self.get_cot_sections(text, uncond_kwargs, drop_think=drop_think))
                        else:
                            _sub_sections.append(dict(type="text", text=f"{answer_prefix}{text}{answer_suffix}", **uncond_kwargs))
                    else:
                        _sub_sections.append(dict(
                            type="text", text=text, **uncond_kwargs))
                elif message['type'] == 'gen_image':
                    from hunyuan_image_3.tokenizer_wrapper import ImageInfo
                    info = message['content']
                    assert isinstance(info, ImageInfo), f"Expected ImageInfo, but got {type(info)}"
                    if role == "assistant":
                        _sub_sections.append(dict(type="text", text=answer_prefix))
                    _sub_sections.append(dict(type=message['type'], **info.meta_info))
                    if role == "assistant":
                        _sub_sections.append(dict(type="text", text=answer_suffix))
                elif message['type'] == 'joint_image':
                    from hunyuan_image_3.tokenizer_wrapper import JointImageInfo
                    info = message['content']
                    assert isinstance(info, JointImageInfo), f"Expected JointImageInfo, but got {type(info)}"
                    _sub_sections.append(dict(type=message['type'], **info.meta_info))
                else:
                    raise ValueError(f"Unknown message type: {message['type']}")
                _cur_message_idx += 1
            if len(_sub_sections) > 0:
                # Add role prefix and suffix
                _sub_sections.insert(0, dict(type='text', text=prefix))
                _sub_sections.append(dict(type='text', text=suffix))
            return _sub_sections, _cur_message_idx

        # Define assistant prefix and suffix
        if (answer == "auto" and sequence_template == "instruct") or answer is True:
            answer_prefix, answer_suffix = "<answer>", "</answer>"
        else:
            answer_prefix, answer_suffix = "", ""
        if sequence_template == "pretrain":
            system_suffix = ""
            user_prefix = ""
            user_suffix = ""
            bot_prefix = ""
            bot_suffix = ""
        else:
            system_suffix = f"{conv.sep}"
            user_prefix = f"{conv.roles[0]}: "
            user_suffix = f"{conv.sep}"
            bot_prefix = f"{conv.roles[1]}: "
            bot_suffix = f"{conv.sep}"

        # Process successive user and assistant messages
        sections = []
        cur_message_idx = 0
        final_role = None
        while cur_message_idx < len(message_list):
            # Process successive system messages
            sub_sections, cur_message_idx = process_successive_message(
                message_list, cur_message_idx, role="system", prefix="", suffix=system_suffix)
            sections.extend(sub_sections)
            if len(sub_sections) > 0:
                final_role = "system"

            # Process successive user messages
            sub_sections, cur_message_idx = process_successive_message(
                message_list, cur_message_idx, role="user", prefix=user_prefix, suffix=user_suffix)
            sections.extend(sub_sections)
            if len(sub_sections) > 0:
                final_role = "user"

            # Process successive assistant messages
            sub_sections, cur_message_idx = process_successive_message(
                message_list, cur_message_idx, role="assistant", prefix=bot_prefix, suffix=bot_suffix,
                answer_prefix=answer_prefix, answer_suffix=answer_suffix,
            )
            sections.extend(sub_sections)
            if len(sub_sections) > 0:
                final_role = "assistant"

        if add_assistant_prefix:
            if final_role == "assistant":
                _bot_prefix = ""
                if len(sections) > 0 and sections[-1]['type'] == 'text' and sections[-1]['text'] == bot_suffix:
                    sections = sections[:-1]
            else:
                _bot_prefix = bot_prefix
            bot_response_prefix = dict(
                auto=_bot_prefix,
                image="",
                think=f"{_bot_prefix}<think>",
                recaption=f"{_bot_prefix}<recaption>",
                img_ratio=f"{_bot_prefix}{answer_prefix}<boi><img_size_{image_base_size}>",
            )[bot_task]
            sections.append(dict(type='text', text=bot_response_prefix))

        # Call encode_general with our add_eos parameter
        output = self.encode_general(
            sections=sections,
            use_text_mask=False,
            add_eos=add_eos,  # <-- Use the parameter we received
            add_pad=False,
        )

        if max_length is not None:
            if output.tokens.shape[-1] > max_length:
                raise ValueError(
                    f"Encoded token length {output.tokens.shape[-1]} exceeds max_length {max_length}.\n"
                    f"Please set a larger max_length or check the input messages:\n{message_list}"
                )

        return output, sections
    
    # Bind the patched method to the instance
    tokenizer_wrapper.apply_general_template = types.MethodType(
        patched_apply_general_template, 
        tokenizer_wrapper
    )
    
    logger.info("TokenizerWrapper.apply_general_template patched to support add_eos parameter")



class MultiModalDataset(Dataset):
    """
    Multimodal dataset supporting:
    1. Text-only: {"type": "text_only", "messages": [...]} for LM
    2. Image Generation: {"type": "image_generation", "messages": [...], "target_image": "path/to/image.jpg"} for T2I
    
    Data Format Examples:
    
    1. Text-only (Language Modeling):
        {
            "type": "text_only",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is AI?"},
                {"role": "assistant", "content": "AI stands for Artificial Intelligence..."}
            ]
        }
    
    2. Image Generation - Instruct Style:
        {
            "type": "image_generation",
            "messages": [
                {"role": "system", "content": "You are an image generation assistant."},
                {"role": "user", "content": "Generate a photo of a red sports car"},
                {"role": "assistant", "content": "Here is the image:"}
            ],
            "target_image": "datasets/images/red_car.jpg"
        }
        
        Note: The gen_image tokens will be automatically appended after assistant's text response.
        The model learns to first generate the text "Here is the image:" then generate the image tokens.
    
    3. Image Generation - Pretrain Style:
        {
            "type": "image_generation",
            "messages": [
                {"role": "user", "content": "A photo of a red sports car"}
            ],
            "target_image": "datasets/images/red_car.jpg"
        }
        
        Note: No system/assistant roles. Gen_image tokens appended directly after user prompt.
        Suitable for continued pretraining on text-to-image pairs.
    """
    
    def __init__(
        self,
        data_file: str,
        tokenizer_wrapper: TokenizerWrapper,
        model: HunyuanImage3ForCausalMM,
        max_seq_length: int = 8192,
        train_text_only: bool = True,
        train_image_understanding: bool = False,
        train_image_generation: bool = False,
        image_base_size: int = 1024,
        image_ratio: str = "1:1",
        sequence_template: str = "instruct",  # "instruct" or "pretrain"
        image_root_dir: str = "",
    ):
        self.tokenizer_wrapper = tokenizer_wrapper
        self.model = model
        self.image_processor = model.image_processor
        self.max_seq_length = max_seq_length
        self.train_text_only = train_text_only
        self.train_image_understanding = train_image_understanding
        self.train_image_generation = train_image_generation
        self.image_base_size = image_base_size
        self.image_ratio = image_ratio
        self.sequence_template = sequence_template
        self.image_root_dir = image_root_dir
        
        # Patch tokenizer_wrapper to support add_eos parameter
        patch_tokenizer_wrapper_for_training(self.tokenizer_wrapper)
        
        # Load data
        self.data_file = data_file  # Store for path resolution
        self.data_list = self._load_data(data_file)
        
        logger.info(f"Loaded {len(self.data_list)} samples from {data_file}")
        logger.info(f"Training modes: text_only={train_text_only}, "
                   f"image_understanding={train_image_understanding}, "
                   f"image_generation={train_image_generation}")
        logger.info(f"Sequence template: {sequence_template}")
    
    def _load_data(self, data_file):
        """Load and filter data based on training modes"""
        with open(data_file, 'r', encoding='utf-8') as f:
            data_list = [json.loads(line) for line in f]
        
        # Filter based on training modes
        filtered_data = []
        for item in data_list:
            sample_type = item.get('type', 'text_only')
            
            if sample_type == 'text_only' and self.train_text_only:
                filtered_data.append(item)
            elif sample_type == 'image_understanding' and self.train_image_understanding:
                filtered_data.append(item)
            elif sample_type == 'image_generation' and self.train_image_generation:
                filtered_data.append(item)
        
        if not filtered_data:
            warnings.warn("No data matches the enabled training modes! Using all data.")
            filtered_data = data_list
        
        return filtered_data
    
    def __len__(self):
        return len(self.data_list)
    
    def _encode_text_only(self, item):
        """Encode text-only sample for language modeling
        
        Supports two training styles:
        1. 'instruct': Only compute loss on assistant responses (SFT style)
        2. 'pretrain': Compute loss on all tokens (standard LM style)
        
        Instruct mode: "Assistant: <answer>response</answer> ... <answer>response</answer><|endoftext|>" 
        Pretrain mode: No "Assistant: " prefix, no <answer> tags
        """
        messages = item['messages']
        
        # Use patched apply_general_template with add_eos=True for proper training
        output, sections = self.tokenizer_wrapper.apply_general_template(
            message_list=[messages],
            max_length=self.max_seq_length,
            add_eos=True,  # Ensure EOS token for text generation training
            sequence_template=self.sequence_template,
            add_assistant_prefix=True,
            answer="auto",
            bot_task="auto",
            uncond_p=0.0,
            cfg_factor=1,
            batchify=True,
        )
        
        tokens = output.tokens[0]
        
        # Truncate if needed
        if len(tokens) > self.max_seq_length:
            tokens = tokens[:self.max_seq_length]
        
        # Create labels based on training style
        if self.sequence_template == "instruct":
            # Instruction tuning: Only compute loss on assistant responses
            # HunyuanImage-3.0 format: "Assistant: <answer>response</answer><|endoftext|>"
            # We need to find <answer>...</answer><|endoftext|> regions
            # Note: We include <|endoftext|> so the model learns to stop generating
            tokenizer = self.tokenizer_wrapper.tokenizer
            
            # Get special token IDs
            answer_start_id = tokenizer.convert_tokens_to_ids('<answer>')
            answer_end_id = tokenizer.convert_tokens_to_ids('</answer>')
            eos_token_id = tokenizer.eos_token_id  # <|endoftext|>
            
            # Initialize all labels as IGNORE_INDEX
            labels = torch.full_like(tokens, IGNORE_INDEX)
            
            # Find all <answer> tags
            answer_starts = (tokens == answer_start_id).nonzero(as_tuple=True)[0].tolist()
            answer_ends = (tokens == answer_end_id).nonzero(as_tuple=True)[0].tolist()
            
            # For the last <answer>...</answer> pair, extend to include the following <|endoftext|>
            for i, (start_idx, end_idx) in enumerate(zip(answer_starts, answer_ends)):
                is_last = (i == len(answer_starts) - 1)
                
                if is_last:
                    # Find the next EOS token after the last </answer>
                    # Look for EOS in the range [end_idx+1, min(end_idx+10, len(tokens))]
                    # (EOS should be immediately after </answer>, but we search a small window)
                    eos_search_end = min(end_idx + 10, len(tokens))
                    next_eos_idx = None
                    
                    for idx in range(end_idx + 1, eos_search_end):
                        if tokens[idx] == eos_token_id:
                            next_eos_idx = idx
                            break

                    if next_eos_idx is None:
                        labels[start_idx:end_idx + 1] = tokens[start_idx:end_idx + 1]
                        logger.warning(f"No EOS token found after </answer> at position {end_idx}")
                    else:
                        labels[start_idx:next_eos_idx + 1] = tokens[start_idx:next_eos_idx + 1]
                else:
                    labels[start_idx:end_idx + 1] = tokens[start_idx:end_idx + 1]

            if len(answer_starts) == 0:
                # If no <answer> tags found, this is a fallback - ideally all instruct data should have answer tags
                logger.warning(f"No <answer> tags found in instruct mode. "
                             f"Consider using answer tags or sequence_template='pretrain'.")
                # Fall back to computing loss on all tokens
                labels = tokens.clone()
        else:
            # Pretrain style: Compute loss on all tokens
            labels = tokens.clone()
        
        # Create attention mask
        attention_mask = torch.ones_like(tokens)
        
        # Compute RoPE embeddings
        # For text-only, use 1D RoPE
        seq_len = len(tokens)
        cos, sin = build_batch_2d_rope(
            seq_len=seq_len,
            n_elem=self.model.config.attention_head_dim,
            image_infos=None,
            device=tokens.device,
            base=self.model.config.rope_theta,
            base_rescale_factor=1.0,
        )

        # Build position ids (required for indexing into RoPE embeddings)
        # Shape: [1, seq_len] - will be expanded in collator for batching
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=tokens.device).unsqueeze(0)
        
        return {
            'input_ids': tokens,
            'labels': labels,
            'attention_mask': attention_mask,
            'mode': 'gen_text',
            'custom_pos_emb': (cos, sin),
            'position_ids': position_ids,
            'tokenizer_output': output,
        }
    
    def _encode_image_understanding(self, item):
        """Encode text+image sample for VL understanding"""
        # TODO: Implement
        # For now, fall back to text-only
        warnings.warn("Image understanding training not implemented yet. "
                     "Falling back to text-only mode.")
        return self._encode_text_only(item)
    
    def _encode_image_generation(self, item):
        """Encode image-generation sample for t2i training
        
        Supports two training styles:
        
        1. 'instruct': Assistant responds with text then image (SFT style)
           Example messages:
           [
               {"role": "system", "content": "You are an image generation assistant."},
               {"role": "user", "content": "Generate a red car"},
               {"role": "assistant", "content": "Here is the image:"}
           ]
           
           The gen_image tokens are appended automatically after assistant's text.
           Loss is computed only on assistant's response (text + image tokens).
        
        2. 'pretrain': Direct user prompt to image (continued pretraining style)
           Example messages:
           [
               {"role": "user", "content": "A red sports car"}
           ]
           
           The gen_image tokens are appended directly after user prompt.
           Loss is computed on all tokens (user prompt + image tokens).
        """
        # Create a copy of messages to avoid modifying the original data
        messages = item['messages'].copy()
        
        gen_img_info = self.image_processor.build_image_info(self.image_ratio)
        messages.append({"role": "assistant", "content": gen_img_info, "type": "gen_image"})

        # Currently only allow at most one gen_image for each message_list
        batch_gen_image_info = [gen_img_info]
        
        # Use patched apply_general_template with add_eos=True for proper training
        output, sections = self.tokenizer_wrapper.apply_general_template(
            message_list=[messages],
            max_length=self.max_seq_length,
            sequence_template=self.sequence_template,
            add_assistant_prefix=True,
            answer="auto",
            bot_task="auto",
            uncond_p=0.0,
            cfg_factor=1,
            batchify=True,
            image_base_size=self.image_base_size
        )
        
        model_device = self.model.device
        tokens = output.tokens[0]
        
        # Create labels based on training style
        if self.sequence_template == "instruct":
            # Instruction tuning: Only compute loss on assistant responses
            # HunyuanImage-3.0 format: "Assistant: <answer>response</answer><|endoftext|>"
            # We need to find <answer>...</answer><|endoftext|> regions
            # Note: We include <|endoftext|> so the model learns to stop generating
            tokenizer = self.tokenizer_wrapper.tokenizer
            
            # Get special token IDs
            answer_start_id = tokenizer.convert_tokens_to_ids('<answer>')
            answer_end_id = tokenizer.convert_tokens_to_ids('</answer>')
            eos_token_id = tokenizer.eos_token_id  # <|endoftext|>
            
            # Initialize all labels as IGNORE_INDEX
            labels = torch.full_like(tokens, IGNORE_INDEX)
            
            # Find all <answer> tags
            answer_starts = (tokens == answer_start_id).nonzero(as_tuple=True)[0].tolist()
            answer_ends = (tokens == answer_end_id).nonzero(as_tuple=True)[0].tolist()
            
            # For the last <answer>...</answer> pair, extend to include the following <|endoftext|>
            for i, (start_idx, end_idx) in enumerate(zip(answer_starts, answer_ends)):
                is_last = (i == len(answer_starts) - 1)
                
                if is_last:
                    # Find the next EOS token after the last </answer>
                    # Look for EOS in the range [end_idx+1, min(end_idx+10, len(tokens))]
                    # (EOS should be immediately after </answer>, but we search a small window)
                    eos_search_end = min(end_idx + 10, len(tokens))
                    next_eos_idx = None
                    
                    for idx in range(end_idx + 1, eos_search_end):
                        if tokens[idx] == eos_token_id:
                            next_eos_idx = idx
                            break
        
                    if next_eos_idx is None:
                        labels[start_idx:end_idx + 1] = tokens[start_idx:end_idx + 1]
                        print(f"No EOS token found after </answer> at position {end_idx}")
                    else:
                        labels[start_idx:next_eos_idx + 1] = tokens[start_idx:next_eos_idx + 1]
                else:
                    labels[start_idx:end_idx + 1] = tokens[start_idx:end_idx + 1]
        
            if len(answer_starts) == 0:
                # If no <answer> tags found, this is a fallback - ideally all instruct data should have answer tags
                print(f"No <answer> tags found in instruct mode. "
                            f"Consider using answer tags or sequence_template='pretrain'.")
                # Fall back to computing loss on all tokens
                labels = tokens.clone()
        else:
            # Pretrain style: Compute loss on all tokens
            labels = tokens.clone()

        # Ignore the gen timestep and gen image tokens
        labels[tuple(output['gen_image_slices'][0])] = IGNORE_INDEX
        labels[tuple(output['gen_timestep_scatter_index'])] = IGNORE_INDEX
        
        # Create attention mask
        attention_mask = torch.ones_like(tokens)
        
        # Compute RoPE embeddings   
        rope_image_info = self.model.build_batch_rope_image_info(output, sections)
        seq_len = len(tokens)
        cos, sin = build_batch_2d_rope(
            image_infos=rope_image_info,
            device=tokens.device,
            seq_len=seq_len,
            n_elem=self.model.config.attention_head_dim,
            base=self.model.config.rope_theta,
        )
        
        # Build position ids (required for indexing into RoPE embeddings)
        # Shape: [1, seq_len] - will be expanded in collator for batching
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=tokens.device).unsqueeze(0)

        # Load target image for training
        vae_image = None
        if 'target_image' in item:
            image_path = self._resolve_image_path(item['target_image'])
            vae_image = self._load_image_for_training(image_path).cpu()  # Trainer will do the memory pinning on cpu
            
        return {
            'input_ids': tokens,
            'labels': labels,
            'attention_mask': attention_mask,
            'mode': 'gen_image',
            'custom_pos_emb': (cos, sin),
            'position_ids': position_ids,
            
            'batch_gen_image_info': batch_gen_image_info,
            'tokenizer_output': output,
            
            'gen_timestep_scatter_index': output.gen_timestep_scatter_index,
            'image_mask': output.gen_image_mask,

            'vae_image': vae_image,
        }

    def _resolve_image_path(self, image_path: str) -> str:
        """Resolve image path (handle both absolute and relative paths)"""
        if os.path.isabs(image_path):
            return image_path
        
        # Try relative to image_root_dir
        if self.image_root_dir:
            full_path = os.path.join(self.image_root_dir, image_path)
            if os.path.exists(full_path):
                return full_path
        
        # Try relative to data file directory
        data_dir = os.path.dirname(os.path.abspath(self.data_file))
        full_path = os.path.join(data_dir, image_path)
        if os.path.exists(full_path):
            return full_path
        
        # Return as-is and let it fail if not found
        return image_path
    
    @torch.no_grad()
    def _load_image_for_training(self, image_path: str):
        """
        Load and encode image for training.
        
        Supports two modes:
        1. Pre-cached latent file (if image_path ends with .pt) - loads cached tensor from disk
        2. On-the-fly encoding - loads image and encodes with VAE
        
        Returns:
            vae_latent: Encoded latent tensor (on model device)
        """
        # Check if this is a pre-cached latent file
        if image_path.endswith('.pt'):
            try:
                vae_latent = torch.load(image_path)
                return vae_latent
            except Exception as e:
                logger.error(f"Failed to load pre-cached latent {image_path}: {e}")
                raise
        
        # Load and encode on-the-fly
        raw_image = Image.open(image_path).convert('RGB')
        vae_image_info = self._raw_image_preprocess(raw_image, self.image_ratio)
        
        _, vae_latent = self.model.vae_encode(
            vae_image_info.image_tensor.to(self.model.device),
        )
        
        return vae_latent

    def _raw_image_preprocess(self, image: Image.Image, image_ratio=None):
        if isinstance(image_ratio, str):
            if image_ratio.startswith("<img_ratio_"):
                ratio_index = int(image_ratio.split("_")[-1].rstrip(">"))
                reso = self.image_processor.reso_group[ratio_index]
                image_ratio = reso.height, reso.width
            elif 'x' in image_ratio:
                image_ratio = [int(s) for s in image_ratio.split('x')]
            elif ':' in image_ratio:
                image_ratio = [int(s) for s in image_ratio.split(':')]
            else:
                raise ValueError(
                    f"`image_size` should be in the format of 'HxW', 'H:W' or <img_ratio_i>, got {image_ratio}.")
            assert len(image_ratio) == 2, f"`image_size` should be in the format of 'HxW', got {image_ratio}."
            image_width, image_height = self.image_processor.reso_group.get_target_size(image_ratio[1], image_ratio[0])
        elif isinstance(image_ratio, (list, tuple)):
            assert len(image_ratio) == 2 and all(isinstance(s, int) for s in image_ratio), \
                f"`image_size` should be a tuple of two integers or a string in the format of 'HxW', got {image_ratio}."
            image_width, image_height = self.image_processor.reso_group.get_target_size(image_ratio[1], image_ratio[0])
        else:
            image_width, image_height = self.image_processor.reso_group.get_target_size(image.width, image.height)
        
        resized_image = resize_and_crop(image, (image_width, image_height))
        image_tensor = self.image_processor.vae_processor(resized_image)
        token_height = image_height // (self.image_processor.config.vae_downsample_factor[0] * self.image_processor.config.patch_size)
        token_width = image_width // (self.image_processor.config.vae_downsample_factor[1] * self.image_processor.config.patch_size)
        base_size, ratio_index = self.image_processor.reso_group.get_base_size_and_ratio_index(width=image_width, height=image_height)
        vae_image_info = ImageInfo(
            image_type="vae",
            image_tensor=image_tensor.unsqueeze(0),     
            image_width=image_width, image_height=image_height,
            token_width=token_width, token_height=token_height,
            base_size=base_size, ratio_index=ratio_index,
        )
        return vae_image_info
    

    def __getitem__(self, index):
        item = self.data_list[index]
        sample_type = item.get('type', 'text_only')
        
        try:
            if sample_type == 'text_only':
                return self._encode_text_only(item)
            elif sample_type == 'image_understanding':
                return self._encode_image_understanding(item)
            elif sample_type == 'image_generation':
                return self._encode_image_generation(item)
            else:
                # Default to text-only
                return self._encode_text_only(item)
        except Exception as e:
            logger.error(f"Error processing sample {index}: {e}")
            raise RuntimeError(f"Failed to process sample at index {index}. Ensure the data format matches the expected structure.\nSample type: {sample_type}. Error: {e}")
        

@dataclass
class MultiModalDataCollator:
    """
    Collate multimodal data with proper padding.
    
    Important: HunyuanImage-3.0 requires different attention mask formats depending on 
    the attention implementation:
    - flash_attention_2: 2D mask [batch, seq_len]
    - sdpa/eager: 4D causal mask [batch, 1, seq_len, seq_len]
      (Both use the same SDPA attention implementation in HunyuanImage-3.0)
    """
    
    tokenizer: Any
    model: Any = None
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Extract fields
        input_ids = [f['input_ids'] for f in features]
        labels = [f['labels'] for f in features]
        mode = features[0].get('mode', 'gen_text')
        tokenizer_output_list = [f['tokenizer_output'] for f in features]
        
        if mode == 'gen_image':
            image_size = [features[0].get('batch_gen_image_info')[0].image_height, features[0].get('batch_gen_image_info')[0].image_width]
            gen_timestep_scatter_index = torch.cat([f['gen_timestep_scatter_index'] for f in features], dim=0)
            vae_image = torch.cat([f['vae_image'] for f in features], dim=0)

        PAD_TOKEN_ID = self.tokenizer.pad_token_id
            
        # Pad sequences
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=PAD_TOKEN_ID
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        
        # Create attention mask
        # For HunyuanImage-3.0, we need to handle different attention implementations:
        # - flash_attention_2: 2D mask [batch, seq_len]
        # - sdpa/eager: Both use 4D causal mask
        attention_mask_2d = input_ids.ne(PAD_TOKEN_ID)
        batch_size, seq_len = input_ids.shape
        
        # Check if we're using flash_attention_2
        use_flash_attn = False
        if self.model is not None and hasattr(self.model, 'config'):
            attn_impl = getattr(self.model.config, '_attn_implementation', 'sdpa')
            use_flash_attn = (attn_impl == 'flash_attention_2')
        
        if use_flash_attn:
            # Flash attention: always use 2D mask [batch, seq_len]
            attention_mask = attention_mask_2d.long()
        else:
            # SDPA/Eager: Manually create a 4D causal mask that also masks padding tokens from both query and key dimensions.
            mask_dtype = self.model.dtype if self.model is not None else torch.bfloat16
            device = input_ids.device

            # 1. Create a boolean causal attention mask shape (b, 1, seqlen, seqlen)
            # This implementation can handle sequences with text and image modalities, where text tokens use causal
            # attention and image tokens use full attention.
            batch_image_slices = [
                tokenizer_output_list[i].joint_image_slices[0] + tokenizer_output_list[i].gen_image_slices[0]
                for i in range(batch_size)
            ]
            attention_mask_bool = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).tril(diagonal=0).repeat(batch_size, 1, 1)
            for i in range(batch_size):
                for j, image_slice in enumerate(batch_image_slices[i]):
                    attention_mask_bool[i, image_slice, image_slice] = True
            attention_mask_bool = attention_mask_bool.unsqueeze(1)

            # 2. Create a boolean key padding mask: [batch_size, 1, 1, seq_len]
            #    This mask is True for non-padding tokens in the key dimension.
            key_padding_mask_4d_bool = attention_mask_2d[:, None, None, :].to(torch.bool).to(device)

            # 3. Create a boolean query padding mask: [batch_size, 1, seq_len, 1]
            #    This mask is True for non-padding tokens in the query dimension.
            query_padding_mask_4d_bool = attention_mask_2d[:, None, :, None].to(torch.bool).to(device)

            # 4. Combine all boolean masks with logical AND.
            #    A position (query_idx, key_idx) is allowed if ALL conditions are True:
            #    - query_idx >= key_idx (causal)
            #    - token at query_idx is not padding
            #    - token at key_idx is not padding
            combined_attention_mask_bool = attention_mask_bool & key_padding_mask_4d_bool & query_padding_mask_4d_bool

            # 5. Convert the combined boolean mask to an additive mask.
            #    Allowed positions (True in combined_mask_bool) become 0.
            #    Disallowed positions (False) become a very large negative number (e.g., -inf).
            attention_mask = torch.full(
                (batch_size, 1, seq_len, seq_len),
                fill_value=torch.finfo(mask_dtype).min, # Start with disallowed (large negative)
                dtype=mask_dtype,
                device=device
            )
            attention_mask.masked_fill_(combined_attention_mask_bool, 0) # Fill allowed positions with 0


        # Get the maximum sequence length after padding
        batch_seq_len = input_ids.shape[1]
        
        # Handle custom_pos_emb (RoPE embeddings)
        # For batched training, we need to pad RoPE to match padded sequence length
        if 'custom_pos_emb' in features[0]:
            cos_list, sin_list = [], []
            
            for feature in features:
                cos, sin = feature['custom_pos_emb']
                # Pad RoPE embeddings to match padded sequence length
                if cos.shape[1] < batch_seq_len:
                    pad_len = batch_seq_len - cos.shape[1]
                    # Pad with zeros
                    cos_pad = torch.zeros((cos.shape[0], pad_len, cos.shape[2]), dtype=cos.dtype, device=cos.device)
                    sin_pad = torch.zeros((sin.shape[0], pad_len, sin.shape[2]), dtype=sin.dtype, device=sin.device)
                    cos = torch.cat([cos, cos_pad], dim=1)
                    sin = torch.cat([sin, sin_pad], dim=1)
                cos_list.append(cos)
                sin_list.append(sin)
            
            # Stack into batch
            cos_batch = torch.cat(cos_list, dim=0)  # [batch_size, seq_len, head_dim]
            sin_batch = torch.cat(sin_list, dim=0)
            custom_pos_emb = (cos_batch, sin_batch)
        else:
            custom_pos_emb = None
        
        # Handle position_ids
        # Pad position_ids to match padded sequence length
        if 'position_ids' in features[0]:
            position_ids_list = []
            
            for feature in features:
                pos_ids = feature['position_ids']  # Shape: [1, orig_seq_len]
                if pos_ids.shape[1] < batch_seq_len:
                    # Pad with sequential indices continuing from the last position
                    pad_len = batch_seq_len - pos_ids.shape[1]
                    last_pos = pos_ids[0, -1].item()
                    pad_ids = torch.arange(
                        last_pos + 1, 
                        last_pos + 1 + pad_len, 
                        dtype=pos_ids.dtype, 
                        device=pos_ids.device
                    ).unsqueeze(0)
                    pos_ids = torch.cat([pos_ids, pad_ids], dim=1)
                position_ids_list.append(pos_ids)
            
            # Stack into batch
            position_ids = torch.cat(position_ids_list, dim=0)  # [batch_size, seq_len]
        else:
            position_ids = None

        # Handle image_mask
        if 'image_mask' in features[0]:
            image_mask_list = []
            
            for feature in features:
                img_mask = feature['image_mask']  # Shape: [1, orig_seq_len]
                if img_mask.shape[1] < batch_seq_len:
                    # Pad with image_mask using False
                    pad_len = batch_seq_len - img_mask.shape[1]
                    pad_img_mask = torch.zeros(pad_len, dtype=torch.bool, device=img_mask.device).unsqueeze(0)
                    img_mask = torch.cat([img_mask, pad_img_mask], dim=1)
                image_mask_list.append(img_mask)
            
            # Stack into batch
            image_mask = torch.cat(image_mask_list, dim=0)  # [batch_size, seq_len]
        else:
            image_mask = None


        result = {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'mode': mode,
        }

        if mode == 'gen_image':
            result['image_size'] = image_size
            result['gen_timestep_scatter_index'] = gen_timestep_scatter_index
            result['vae_image'] = vae_image
        
        if custom_pos_emb is not None:
            result['custom_pos_emb'] = custom_pos_emb
        
        if position_ids is not None:
            result['position_ids'] = position_ids

        if image_mask is not None:
            result['image_mask'] = image_mask
        
        return result
