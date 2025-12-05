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
Multimodal training script for HunyuanImage-3.0 model.

Supports:
1. Text-only supervised fine-tuning
2. Image generation training

Training Modes:
- sequence_template="instruct": Only compute loss on assistant responses (SFT)
  * Uses <answer>...</answer><|endoftext|> tags to mark trainable regions
  * Example: "<|startoftext|>System message\n\nUser: Question\n\nAssistant: <answer>Response</answer><|endoftext|>"
  * Only tokens inside <answer>...</answer><|endoftext|> contribute to loss
  * The model learns to generate the EOS token to properly terminate responses
  
- sequence_template="pretrain": Compute loss on all tokens (continued pretraining)
  * Example: "<|startoftext|>System message\n\nQuestion\n\nResponse<|endoftext|>"
  * All tokens contribute to loss
"""

import os
import sys
import json
import torch
import shutil
import logging
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Union, Optional, Dict, Any, Tuple, Callable
import deepspeed

import transformers
from torch.utils.data import Dataset
from transformers import Trainer, TrainerCallback
from peft import LoraConfig, get_peft_model
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from PIL import Image
import numpy as np

from diffusers.training_utils import (
    compute_density_for_timestep_sampling, 
    compute_loss_weighting_for_sd3,
    free_memory,
)
from diffusers.utils.torch_utils import randn_tensor

from hunyuan_image_3.hunyuan import HunyuanImage3ForCausalMM, CausalMMOutputWithPast
from hunyuan_image_3.configuration_hunyuan import HunyuanImage3Config
from hunyuan_image_3.tokenizer_wrapper import TokenizerWrapper
from hunyuan_image_3.hunyuan_image_3_pipeline import FlowMatchDiscreteScheduler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import dataset components from separate module
from mm_dataset import (
    IGNORE_INDEX,
    BatchRaggedImages,
    BatchRaggedTensor,
    MultiModalDataset,
    MultiModalDataCollator,
    patch_tokenizer_wrapper_for_training,
)


class HunyuanImage3ForCausalMMTrain(HunyuanImage3ForCausalMM):
    """
    Subclass of HunyuanImage3ForCausalMM that customizes the forward function
    to always compute the logits, regardless of the mode (e.g., 'gen_text' or 'gen_image').
    This ensures that the output always includes logits for both text and image generation tasks.
    """

    def forward(
        self, 
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        custom_pos_emb: Optional[Tuple[torch.FloatTensor]] = None,
        mode: str = "gen_text",
        # for gen image
        first_step: Optional[bool] = None,
        images: Optional[BatchRaggedImages] = None,
        image_mask: Optional[torch.Tensor] = None,
        timestep: Optional[BatchRaggedTensor] = None,
        gen_timestep_scatter_index: Optional[torch.Tensor] = None,
        # for cond image
        cond_vae_images: Optional[BatchRaggedImages] = None,
        cond_timestep: Optional[BatchRaggedTensor] = None,
        cond_vae_image_mask: Optional[torch.Tensor] = None,
        cond_vit_images: Optional[BatchRaggedImages] = None,
        cond_vit_image_mask: Optional[torch.Tensor] = None,
        vit_kwargs: Optional[Dict[str, Any]] = None,
        cond_timestep_scatter_index: Optional[torch.Tensor] = None, 
        **kwargs):
        
        # Remove inputs_embeds if PEFT passed it
        kwargs.pop('inputs_embeds', None)
        kwargs.pop('labels', None)

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # Sanity Check of Inputs
        self._check_inputs(mode == "gen_image", "in `gen_image` mode", [
            ("images", images), ("timestep", timestep), ("gen_timestep_scatter_index", gen_timestep_scatter_index),
        ])
        self._check_inputs(mode == "gen_image" and first_step, "in `gen_image` mode at the first step", [
            ("image_mask", image_mask),
        ])
        self._check_inputs(cond_vae_images is not None, "`cond_vae_images` is provided", [
            ("cond_timestep", cond_timestep), ("cond_vae_image_mask", cond_vae_image_mask),
            ("cond_timestep_scatter_index", cond_timestep_scatter_index),
        ])
        self._check_inputs(cond_vit_images is not None, "`cond_vit_images` is provided", [
            ("cond_vit_image_mask", cond_vit_image_mask), ("vit_kwargs", vit_kwargs),
        ])

        custom_pos_emb = self.get_pos_emb(custom_pos_emb, position_ids)

        inputs_embeds = self.model.wte(input_ids)
        bsz, seq_len, n_embd = inputs_embeds.shape

        # Instantiate placeholder tokens: <timestep>, <img> for the gen image
        if mode == "gen_text":
            # For gen_text, make sure gen_timestep_scatter_index is None
            gen_timestep_scatter_index = None
            token_h, token_w = None, None
        else:
            if first_step:
                inputs_embeds, token_h, token_w = self.instantiate_vae_image_tokens(
                    inputs_embeds, images, timestep, image_mask)
                inputs_embeds = self.instantiate_timestep_tokens(
                    inputs_embeds, timestep, gen_timestep_scatter_index)
            else:
                t_emb = self.time_embed(timestep)
                image_emb, token_h, token_w = self.patch_embed(images, t_emb)
                timestep_emb = self.timestep_emb(timestep).reshape(bsz, -1, n_embd)
                inputs_embeds = torch.cat([timestep_emb, image_emb], dim=1)

        # Instantiate placeholder tokens: <timestep>, <img> for cond images
        # Should only run once with kv-cache enabled.
        if cond_vae_images is not None:
            inputs_embeds, _, _ = self.instantiate_vae_image_tokens(
                inputs_embeds, cond_vae_images, cond_timestep, cond_vae_image_mask)
            inputs_embeds = self.instantiate_timestep_tokens(
                inputs_embeds, cond_timestep, cond_timestep_scatter_index)
        if cond_vit_images is not None:
            inputs_embeds = self.instantiate_vit_image_tokens(
                inputs_embeds, cond_vit_images, cond_vit_image_mask, vit_kwargs)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            custom_pos_emb=custom_pos_emb,
            mode=mode,
            first_step=first_step,
            gen_timestep_scatter_index=gen_timestep_scatter_index,
        )
        hidden_states = outputs[0]

        # Always compute the logits
        logits = self.lm_head(self.model.ln_f(hidden_states))
        logits = logits.float()
        diffusion_prediction = None

        if mode == "gen_image":
            hidden_states = hidden_states.to(input_ids.device)
            diffusion_prediction = self.ragged_final_layer(
                hidden_states, image_mask, timestep, token_h, token_w, first_step)

        if not return_dict:
            output = (logits,) + outputs[1:] + (diffusion_prediction,)
            return output

        output = CausalMMOutputWithPast(
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            diffusion_prediction=diffusion_prediction,
        )

        return output



def print_args(args, name='arguments'):
    """Print arguments."""
    if torch.distributed.get_rank() == 0:
        print(f'------------------------ {name} ------------------------', flush=True)
        str_list = []
        for arg in vars(args):
            dots = '.' * (48 - len(arg))
            str_list.append('  {} {} {}'.format(arg, dots, getattr(args, arg)))
        for arg in sorted(str_list, key=lambda x: x.lower()):
            print(arg, flush=True)
        print(f'-------------------- end of {name} ---------------------', flush=True)


@dataclass
class ModelArguments:
    """Model configuration arguments"""
    use_flash_attn: bool = field(
        default=False,
        metadata={"help": "Use FlashAttention-2 as the attention implementation."}
    )
    use_lora: bool = field(
        default=False,
        metadata={"help": "Enable LoRA for parameter-efficient fine-tuning."}
    )
    lora_rank: int = field(default=64, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=128, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout"})

    lora_target_modules: str = field(
        default="qkv_proj,o_proj",  # gate_and_up_proj,down_proj
        metadata={"help": "Comma-separated list of target modules for LoRA"}
    )

    train_vision_model: bool = field(
        default=False,
        metadata={"help": "Whether to train the vision encoder (ViT)"}
    )
    train_vae: bool = field(
        default=False,
        metadata={"help": "Whether to train the VAE decoder"}
    )
    moe_drop_tokens: bool = field(
        default=True,
        metadata={"help": "Enable token dropping in MoE layers to stabilize VRAM usage during training."}
    )


@dataclass
class DataArguments:
    """Data configuration arguments"""
    train_data_file: str = field(
        metadata={"help": "Path to the training data (JSONL format)."}
    )
    max_seq_length: int = field(
        default=8192,
        metadata={"help": "Maximum sequence length for text."}
    )
    image_base_size: int = field(
        default=1024,
        metadata={"help": "Base image size for generation. Choose from [256, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192]"}
    )
    image_ratio: str = field(
        default="1:1",
        metadata={"help": "Image ratio for generation. Should be in the format of 'HxW', 'H:W' or <img_ratio_i>"}
    )
    max_images_per_sample: int = field(
        default=4,
        metadata={"help": "Maximum number of images per training sample."}
    )
    train_text_only: bool = field(
        default=True,
        metadata={"help": "Train on text-only samples (language modeling)."}
    )
    train_image_understanding: bool = field(
        default=False,
        metadata={"help": "Train on text+image samples for VL understanding."}
    )
    train_image_generation: bool = field(
        default=False,
        metadata={"help": "Train image generation capability."}
    )
    sequence_template: str = field(
        default="instruct",
        metadata={"help": "Template style: 'instruct' for SFT (only assistant responses have loss), 'pretrain' for all tokens"}
    )
    image_root_dir: str = field(
        default="",
        metadata={"help": "Root directory for resolving relative image paths. If empty, paths are resolved relative to data file."}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    """Extended training arguments"""
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=8192)
    tokenizer_name_or_path: Optional[str] = field(default=None)
    model_name_or_path: Optional[str] = field(default=None)
    min_lr: float = field(default=0.0)
    text_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for text loss"}
    )
    image_gen_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for image generation loss"}
    )
    make_moe_param_leaf_module: bool = field(
        default=False, 
        metadata={"help": "Make MoE parameters zero-3 leaf module."}
    )
    remove_unused_columns: bool = field(
        default=False, # We need to disable the check as the forward method does not take in 'labels'.
        metadata={"help": "Remove columns not required by the model forward method."}
    )

    weighting_scheme: str = field(
        default="none",
        metadata={
            "help": "We default to the 'none' weighting scheme for uniform sampling and uniform loss. Choices: ['sigma_sqrt', 'logit_normal', 'mode', 'cosmap', 'none']"
        }
    )
    logit_mean: float = field(
        default=0.0,
        metadata={"help": "mean to use when using the 'logit_normal' weighting scheme."}
    )
    logit_std: float = field(
        default=1.0,
        metadata={"help": "std to use when using the 'logit_normal' weighting scheme."}
    )
    mode_scale: float = field(
        default=1.29,
        metadata={"help": "Scale of mode weighting scheme. Only effective when using the 'mode' as the weighting_scheme."}
    )


def make_supervised_data_module(
    tokenizer_wrapper: TokenizerWrapper,
    model: HunyuanImage3ForCausalMM,
    data_args: DataArguments,
) -> Dict:
    """Create dataset and collator"""
    
    train_dataset = MultiModalDataset(
        data_file=data_args.train_data_file,
        tokenizer_wrapper=tokenizer_wrapper,
        model=model,
        max_seq_length=data_args.max_seq_length,
        train_text_only=data_args.train_text_only,
        train_image_understanding=data_args.train_image_understanding,
        train_image_generation=data_args.train_image_generation,
        image_base_size=data_args.image_base_size,
        image_ratio=data_args.image_ratio,
        sequence_template=data_args.sequence_template,
        image_root_dir=data_args.image_root_dir,
    )
    
    data_collator = MultiModalDataCollator(
        tokenizer=tokenizer_wrapper.tokenizer,
        model=model
    )
    
    return dict(
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator
    )


class CustomSaveCallback(TrainerCallback):
    """Callback to save additional model files"""
    
    def on_save(self, args, state, control, **kwargs):
        if torch.distributed.get_rank() == 0:
            output_dir = os.path.join(
                args.output_dir,
                f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
            )
            
            # Copy model files
            source_dir = args.model_name_or_path
            files_to_copy = [
                'hunyuan.py',
                'configuration_hunyuan.py',
                'tokenizer_wrapper.py',
                'hunyuan_image_3_pipeline.py',
                'image_processor.py',
                'autoencoder_kl_3d.py',
                'siglip2.py',
                'system_prompt.py',
                'tokenizer_config.json',
                'generation_config.json',
            ]
            
            for filename in files_to_copy:
                src = os.path.join(source_dir, filename)
                dst = os.path.join(output_dir, filename)
                if os.path.exists(src):
                    try:
                        shutil.copy(src, dst)
                    except Exception as e:
                        logger.warning(f"Failed to copy {filename}: {e}")
            
            # Update config.json with auto_map
            config_path = os.path.join(output_dir, "config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                config['auto_map'] = {
                    "AutoConfig": "configuration_hunyuan.HunyuanImage3Config",
                    "AutoModel": "hunyuan.HunyuanImage3Model",
                    "AutoModelForCausalLM": "hunyuan.HunyuanImage3ForCausalMM"
                }
                
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
        
        return control


class MultiModalTrainer(Trainer):
    """
    Custom trainer for multimodal training.
    Handles both text generation and image generation losses.
    """
    
    # We don't use num_items_in_batch in compute_loss, so set this to False
    # to avoid slightly inaccurate loss calculation during gradient accumulation
    model_accepts_loss_kwargs = False

    def get_sigmas(self, scheduler, timesteps, n_dim=4, dtype=torch.float32, device="cpu"):
        sigmas = scheduler.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = scheduler.timesteps.to(device)
        timesteps = timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute loss for multimodal training.
        """
        # labels should use IGNORE_INDEX on: Padding, Image Tokens, Timestep Tokens, (User/System Token for instruct tuning)
        labels = inputs.pop("labels")  # Remove labels from input

        if inputs.get("mode", "gen_text") == "gen_image":
            # Unwrap DeepSpeed engine to get the actual model
            actual_model = model.module if hasattr(model, 'module') else model

            scheduler = FlowMatchDiscreteScheduler(
                shift=actual_model.generation_config.flow_shift, 
                reverse=True, 
                solver="euler",
            )

            model_input = inputs.pop("vae_image") 
            image_size = inputs.pop("image_size")
        
            batch_size = model_input.shape[0]
            
            # Prepare latent variables
            latent_scale_factor = actual_model.config.vae_downsample_factor
            latent_channel = actual_model.config.vae["latent_channels"]
            
            latents_shape = (
                batch_size,
                latent_channel,
                *[int(s) // f for s, f in zip(image_size, latent_scale_factor)],
            )
            noise_latents = randn_tensor(latents_shape, generator=None, device=model_input.device, dtype=torch.bfloat16)
            
            # Check existence to make it compatible with FlowMatchEulerDiscreteScheduler
            if hasattr(scheduler, "init_noise_sigma"):
                # scale the initial noise by the standard deviation required by the scheduler
                noise_latents = noise_latents * scheduler.init_noise_sigma
            
            weighting_scheme = getattr(self.args, 'weighting_scheme', "none")
            logit_mean = getattr(self.args, 'logit_mean', 0.0)
            logit_std = getattr(self.args, 'logit_std', 1.0)
            mode_scale = getattr(self.args, 'mode_scale', 1.29)

            # Sample a random timestep for each image
            # For weighting schemes where we sample timesteps non-uniformly
            u = compute_density_for_timestep_sampling(
                weighting_scheme=weighting_scheme,
                batch_size=batch_size,
                logit_mean=logit_mean,
                logit_std=logit_std,
                mode_scale=mode_scale,
            )
            indices = (u * scheduler.config.num_train_timesteps).long()
            timesteps = scheduler.timesteps[indices].to(device=model_input.device)
            sigmas = self.get_sigmas(scheduler, timesteps, n_dim=model_input.ndim, dtype=model_input.dtype, device=model_input.device)
            
            # Add noise according to flow matching
            noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise_latents
        
            noisy_model_input = scheduler.scale_model_input(noisy_model_input, timesteps).to(torch.bfloat16)
        
            # Remove unused kwargs, add new kwargs ['images', 'timestep']
            model_inputs = actual_model.prepare_inputs_for_generation( 
                images=noisy_model_input,
                timestep=timesteps,
                **inputs,
            )
        else:
            model_inputs = inputs
        
        # Forward pass
        outputs = model(**model_inputs, first_step=True)
        
        logits = outputs.logits
        diffusion_prediction = outputs.diffusion_prediction

        # Compute text generation loss (standard causal LM loss)
        if logits is not None:
            # Shift for causal language modeling
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Compute loss
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
            text_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        else:
            text_loss = torch.tensor(0.0, device=model.device)
        
        # Compute flow matching loss
        if diffusion_prediction is not None:
            model_pred = diffusion_prediction
            model_pred = model_pred.to(dtype=torch.float32)
            
            weighting = compute_loss_weighting_for_sd3(weighting_scheme=weighting_scheme, sigmas=sigmas)
            
            target = noise_latents - model_input
            
            # Compute loss
            sample_losses = torch.mean(
                (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                1,
            )
            # sample_losses is a 1D tensor of size (batch_size,) where each element is the weighted MSE for one sample.
            image_loss = sample_losses.mean()
        else:
            image_loss = torch.tensor(0.0, device=model.device)
        
        
        # Combine losses
        text_weight = getattr(self.args, 'text_loss_weight', 1.0)
        image_weight = getattr(self.args, 'image_gen_loss_weight', 1.0)
        
        total_loss = text_weight * text_loss + image_weight * image_loss
        
        return (total_loss, outputs) if return_outputs else total_loss



def train():
    """Main training function"""
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Print arguments
    if torch.distributed.get_rank() == 0:
        print_args(model_args, 'model arguments')
        print_args(data_args, 'data arguments')
        print_args(training_args, 'training arguments')
    
    # Initialize kwargs for model
    init_kwargs = {}
    init_kwargs["moe_drop_tokens"] = model_args.moe_drop_tokens
    init_kwargs["image_base_size"] = data_args.image_base_size
    
    if model_args.use_flash_attn:
        init_kwargs["attn_implementation"] = "flash_attention_2"

    if training_args.bf16:
        init_kwargs["torch_dtype"] = torch.bfloat16
    elif training_args.fp16:
        init_kwargs["torch_dtype"] = torch.float16
    
    # Load model
    if training_args.model_name_or_path and os.path.exists(training_args.model_name_or_path):
        logger.info(f"Loading model from {training_args.model_name_or_path}")
        model = HunyuanImage3ForCausalMMTrain.from_pretrained(
            training_args.model_name_or_path,
            trust_remote_code=True,
            **init_kwargs
        )
    else:
        raise ValueError(
            f"Model path {training_args.model_name_or_path} does not exist. "
            "Random initialization not yet supported for HunyuanImage3."
        )

    # Load tokenizer wrapper
    # Use tokenizer_name_or_path if provided, otherwise fall back to model_name_or_path
    tokenizer_path = training_args.tokenizer_name_or_path or training_args.model_name_or_path
    logger.info(f"Loading tokenizer from {tokenizer_path}")
    model.load_tokenizer(tokenizer_path)
    
    # Freeze components based on training flags
    if not model_args.train_vision_model:
        logger.info("Freezing vision model (ViT)")
        for param in model.vision_model.parameters():
            param.requires_grad = False
        for param in model.vision_aligner.parameters():
            param.requires_grad = False
    
    if not model_args.train_vae:
        logger.info("Freezing VAE")
        for param in model.vae.parameters():
            param.requires_grad = False
    

    # --- START: Patch for PEFT ---
    # Add the required embedding accessor methods that PEFT needs for gradient checkpointing and proper LoRA adaptation

    model.get_input_embeddings = lambda: model.model.wte
    model.set_input_embeddings = lambda value: setattr(model.model, 'wte', value)
    
    # --- END: Patch for PEFT ---
    

    # Apply LoRA if requested
    if model_args.use_lora:
        logger.info(f"Applying LoRA with rank={model_args.lora_rank}")
        target_modules = [module.strip().lower() for module in model_args.lora_target_modules.split(',')]
        lora_config = LoraConfig(
            r=model_args.lora_rank,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    # Do not split MoE weights when using zero3
    if training_args.make_moe_param_leaf_module \
        and training_args.deepspeed_plugin.zero_stage == 3:
        from deepspeed.utils import set_z3_leaf_modules
        from hunyuan_image_3.hunyuan import HunyuanMoE
        set_z3_leaf_modules(model, [HunyuanMoE])

    # Create data module
    data_module = make_supervised_data_module(
        tokenizer_wrapper=model.tokenizer,
        model=model,
        data_args=data_args,
    )
    
    # Set model flags
    model.is_parallelizable = True
    model.model_parallel = True
    model.config.use_cache = False
    
    # Setup learning rate scheduler kwargs
    training_args.lr_scheduler_kwargs = {
        'min_lr': training_args.min_lr,
    }
    
    # Create trainer
    trainer = MultiModalTrainer(
        model=model,
        args=training_args,
        callbacks=[CustomSaveCallback],
        **data_module
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    
    # Save final model
    logger.info("Saving final model...")
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    train()
