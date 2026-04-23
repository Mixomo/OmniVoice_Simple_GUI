#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors:  Han Zhu)
#
# See ../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training loop for OmniVoice.

Wraps the HuggingFace Accelerate training loop with checkpoint saving/resuming,
evaluation, gradient accumulation, and learning rate scheduling.
Launched via ``omnivoice.cli.train``.
"""

import logging
import math
import os
import sys
import time
from datetime import timedelta
from typing import Any, Optional

import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import DeepSpeedPlugin, InitProcessGroupKwargs, set_seed
from torch.utils.data import DataLoader
from transformers import (
    get_cosine_schedule_with_warmup,
    get_constant_schedule_with_warmup,
)

from omnivoice.training.checkpoint import TrainLogger, load_checkpoint
from omnivoice.training.checkpoint import save_checkpoint as engine_save_checkpoint

logger = logging.getLogger(__name__)


def _to_device(batch, device):
    """Move all tensors in a batch dict to the target device."""
    return {
        k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


class OmniTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        config: Any,  # TrainingConfig
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        tokenizer: Optional[Any] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[Any] = None,
    ):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

        # 1. Initialize Accelerator
        self.accelerator = self._init_accelerator()

        # 2. Setup Optimizer & Scheduler if not provided
        if optimizer is None:
            self.optimizer, self.lr_scheduler = self.create_optimizer_and_scheduler()
        else:
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler

        # 3. DeepSpeed Hack (Batch Size fix)
        if self.accelerator.distributed_type == "DEEPSPEED":
            self.accelerator.state.deepspeed_plugin.deepspeed_config[
                "train_micro_batch_size_per_gpu"
            ] = 1

        # 4. Prepare with Accelerator
        (self.model, self.optimizer, self.lr_scheduler, self.train_dataloader) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.lr_scheduler,
            self.train_dataloader
        )
        if self.eval_dataloader is not None:
            self.eval_dataloader = self.accelerator.prepare(self.eval_dataloader)
            
        self.global_step = 0
        self.epoch = 0

    def _init_accelerator(self) -> Accelerator:
        """Initialize Accelerator, DeepSpeed, and Logging."""
        # TF32 setup
        if getattr(self.config, "allow_tf32", False):
            torch.set_float32_matmul_precision("high")

        # Init handlers
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
        init_kwargs = InitProcessGroupKwargs(timeout=timedelta(minutes=60))

        # DeepSpeed setup
        deepspeed_plugin = None
        if self.config.use_deepspeed and self.config.deepspeed_config:
            if not os.path.exists(self.config.deepspeed_config):
                raise FileNotFoundError(
                    f"DeepSpeed config not found: {self.config.deepspeed_config}"
                )
            deepspeed_plugin = DeepSpeedPlugin(
                hf_ds_config=self.config.deepspeed_config,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                gradient_clipping=self.config.max_grad_norm,
            )

        accelerator = Accelerator(
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            mixed_precision=self.config.mixed_precision,
            log_with="tensorboard",
            project_dir=self.config.output_dir,
            step_scheduler_with_optimizer=False,
            kwargs_handlers=[ddp_kwargs, init_kwargs],
            deepspeed_plugin=deepspeed_plugin,
            split_batches=False,
        )

        # Logging setup
        if accelerator.is_main_process:
            os.makedirs(self.config.output_dir, exist_ok=True)
            # Try to save config if it has the method
            if hasattr(self.config, "save_to_json"):
                self.config.save_to_json(
                    os.path.join(self.config.output_dir, "initial_config.json")
                )

            logging.basicConfig(
                format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                datefmt="%m/%d/%Y %H:%M:%S",
                level=logging.INFO,
                handlers=[
                    logging.StreamHandler(sys.stdout),
                    logging.FileHandler(
                        os.path.join(self.config.output_dir, "train.log")
                    ),
                ],
            )
        else:
            logging.basicConfig(level=logging.ERROR)

        logger.info(f"Loaded Config: {self.config}")
        set_seed(self.config.seed)
        accelerator.init_trackers("tensorboard")
        return accelerator

    def create_optimizer_and_scheduler(self):
        """Default AdamW + configurable LR Scheduler."""
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        if self.config.warmup_type == "ratio":
            final_warmup_steps = math.ceil(self.config.steps * self.config.warmup_ratio)
        else:
            final_warmup_steps = self.config.warmup_steps

        if self.config.lr_scheduler_type == "constant":
            lr_scheduler = get_constant_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=final_warmup_steps,
            )
        else:
            lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=final_warmup_steps,
                num_training_steps=self.config.steps,
            )
        return optimizer, lr_scheduler

    def save_checkpoint(self, step):
        """Wrapper for engine save_checkpoint."""
        engine_save_checkpoint(
            self.accelerator,
            self.model,
            self.tokenizer,
            self.config.output_dir,
            step,
            self.config.keep_last_n_checkpoints,
        )
        # Save config copy for convenience
        if self.accelerator.is_main_process and hasattr(self.config, "save_to_json"):
            checkpoint_dir = os.path.join(self.config.output_dir, f"checkpoint-{step}")
            self.config.save_to_json(os.path.join(checkpoint_dir, "train_config.json"))

    def load_checkpoint(self, checkpoint_path):
        """Wrapper for loading."""
        step = load_checkpoint(self.accelerator, checkpoint_path)
        self.global_step = step
        logger.info(f"Resumed from step {self.global_step}")
        return step

    def evaluate(self):
        """Evaluation loop."""
        if self.eval_dataloader is None:
            return {}

        self.model.eval()
        logger.info(f"Running evaluation at step {self.global_step}...")

        local_loss_sum = torch.tensor(0.0, device=self.accelerator.device)
        eval_count = 0

        with torch.no_grad():
            for eval_batch in self.eval_dataloader:
                eval_batch = _to_device(eval_batch, self.accelerator.device)
                outputs = self.model(**eval_batch)
                local_loss_sum += outputs.loss.detach()
                eval_count += 1

        if eval_count > 0:
            local_mean = local_loss_sum / eval_count
        else:
            local_mean = torch.tensor(0.0, device=self.accelerator.device)

        all_means = self.accelerator.gather(local_mean)
        final_eval_loss = all_means.mean().item()

        eval_metrics = {"eval/loss": final_eval_loss}
        self.accelerator.log(eval_metrics, step=self.global_step)
        logger.info(f"Eval Loss: {final_eval_loss:.4f}")

        self.accelerator.wait_for_everyone()
        self.model.train()
        return eval_metrics

    def train(self):
        """Main training loop."""
        logger.info("Starting Training Loop...")

        # Resume if configured
        if self.config.resume_from_checkpoint:
            self.load_checkpoint(self.config.resume_from_checkpoint)

        # Handle IterableDataset Epochs
        if hasattr(self.train_dataloader.dataset, "set_epoch"):
            self.train_dataloader.dataset.set_epoch(self.epoch)

        # Logger
        train_logger = TrainLogger(
            self.accelerator, self.config.steps, self.config.logging_steps
        )
        train_logger.start(self.global_step)

        self.model.train()
        train_iterator = iter(self.train_dataloader)

        logging_start_time = time.time()
        logging_start_step = self.global_step
        tr_loss = torch.tensor(0.0).to(self.accelerator.device)
        logging_loss_scalar = 0.0

        while self.global_step < self.config.steps:
            try:
                batch = next(train_iterator)
            except StopIteration:
                self.epoch += 1
                logger.info(f"Epoch {self.epoch} starting. Resetting dataloader...")
                if hasattr(self.train_dataloader.dataset, "set_epoch"):
                    self.train_dataloader.dataset.set_epoch(self.epoch)

                train_iterator = iter(self.train_dataloader)
                batch = next(train_iterator)

            batch = _to_device(batch, self.accelerator.device)

            with self.accelerator.accumulate(self.model):
                outputs = self.model(**batch)
                loss = outputs.loss
                tr_loss += loss.detach()
                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients:
                    # Clipping
                    grad_norm = 0.0
                    if self.config.max_grad_norm > 0:
                        grad_norm = self.accelerator.clip_grad_norm_(
                            self.model.parameters(), self.config.max_grad_norm
                        )
                        grad_norm = (
                            grad_norm.item() if grad_norm is not None else 0.0
                        )

                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

                    # Logging
                    current_lr = self.lr_scheduler.get_last_lr()[0]
                    train_logger.update(
                        step=self.global_step, loss=loss.item(), lr=current_lr
                    )

                    if self.global_step % self.config.logging_steps == 0:
                        elapsed = time.time() - logging_start_time
                        steps_per_sec = (
                            (self.global_step - logging_start_step) / elapsed
                            if elapsed > 0
                            else 0
                        )

                        tr_loss_scalar = self.accelerator.gather(tr_loss).mean().item()
                        current_interval_loss = tr_loss_scalar - logging_loss_scalar
                        avg_loss = current_interval_loss / (
                            self.config.logging_steps
                            * self.config.gradient_accumulation_steps
                        )
                        logging_loss_scalar = tr_loss_scalar

                        logs = {
                            "train/loss": avg_loss,
                            "train/learning_rate": current_lr,
                            "train/grad_norm": grad_norm,
                            "train/epoch": self.epoch,
                            "train/steps_per_sec": steps_per_sec,
                        }
                        train_logger.log_metrics(step=self.global_step, metrics=logs)

                        logging_start_time = time.time()
                        logging_start_step = self.global_step

                    # Evaluate
                    if (
                        self.eval_dataloader is not None
                        and self.global_step % self.config.eval_steps == 0
                    ):
                        self.evaluate()

                    # Save & Audio Validation
                    if self.global_step % self.config.save_steps == 0:
                        self.save_checkpoint(self.global_step)
                        # Log validation sample for Tensorboard
                        self.log_validation_sample(self.global_step)

        # Final Save
        self.save_checkpoint(self.global_step)
        self.log_validation_sample(self.global_step)
        train_logger.close()
        self.accelerator.end_training()

    def log_validation_sample(self, step: int):
        """Logs a real audio inference sample to Tensorboard by swapping models in VRAM."""
        if not self.accelerator.is_main_process or not self.config.enable_eval:
            return
        
        logger.info(f"🎨 Generating Evolution Audio at step {step}...")
        
        checkpoint_dir = os.path.join(self.config.output_dir, f"checkpoint-{step}")
        if not os.path.exists(checkpoint_dir):
            return

        try:
            # 1. Unload Training Model AND Optimizer from GPU
            original_device = self.accelerator.device
            model_unwrapped = self.accelerator.unwrap_model(self.model)
            model_unwrapped.to("cpu")
            
            # Move all optimizer states to CPU to truly free VRAM
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to("cpu")
            
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            
            # 2. Load Inference Model (Full dependencies: vocoders, etc.)
            from omnivoice.models.omnivoice import OmniVoice
            # Use sdpa for maximum stability
            inf_attn = "sdpa"
            inf_model = OmniVoice.from_pretrained(
                checkpoint_dir, 
                attn_implementation=inf_attn,
                dtype=torch.float32
            )
            inf_model.to(original_device)
            
            # 3. Generate Sample
            torch.manual_seed(42)
            import numpy as np
            np.random.seed(42)

            gen_text = self.config.eval_text
            ref_audio = self.config.eval_ref_audio
            
            final_ref = None
            if ref_audio and os.path.exists(ref_audio):
                final_ref = ref_audio
            elif self.eval_dataloader is not None:
                try:
                    raw_reader = self.eval_dataloader.dataset.dataset
                    sample = next(iter(raw_reader))
                    final_ref = (sample['audio'][0].cpu(), self.model.sampling_rate)
                    logger.info("Using automatic sample from evaluation set as reference.")
                except: pass

            gen_kwargs = {
                "text": gen_text,
                "num_step": 32,
            }
            if final_ref is not None:
                gen_kwargs["ref_audio"] = final_ref

            audios = inf_model.generate(**gen_kwargs)
            audio_np = audios[0]
            
            # 4. Log to Tensorboard
            for tracker in self.accelerator.trackers:
                if tracker.name == "tensorboard":
                    tracker.tracker.add_audio(
                        "val/audio_evolution", 
                        audio_np, 
                        global_step=step, 
                        sample_rate=inf_model.sampling_rate
                    )
            
            logger.info(f"✅ Audio evolution logged to Tensorboard at step {step}")

            # 5. Cleanup Inference Model Aggressively
            del inf_model
            gc.collect()
            torch.cuda.empty_cache()

            # 6. Restore Training Model AND Optimizer back to GPU
            model_unwrapped.to(original_device)
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(original_device)
                        
            self.model.train()

        except Exception as e:
            logger.warning(f"❌ Failed to generate evolution audio: {e}")
            import traceback
            traceback.print_exc()
            try: 
                self.accelerator.unwrap_model(self.model).to(self.accelerator.device)
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(self.accelerator.device)
                self.model.train()
            except: pass
