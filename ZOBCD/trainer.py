# coding=utf-8
# Copyright 2020-present the HuggingFace Inc. team.
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
"""
The Trainer class, to easily train a 🤗 Transformers from scratch or finetune it on a new task.
"""

import inspect
import math
import os
import re
import shutil
import sys
import time
from functools import partial
from pathlib import Path
from typing import Dict, Union, Any
from typing import TYPE_CHECKING, Optional, Iterable

import numpy as np
import torch
import torch.distributed as dist
from packaging import version

from torch import nn
import torch.nn.functional as F
from torch.func import functional_call, jvp
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
from transformers import Trainer
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.deepspeed import deepspeed_init, is_deepspeed_zero3_enabled
from transformers.dependency_versions_check import dep_version_check
# Integrations must be imported before ML frameworks:
from transformers.integrations import (  # isort: split
    hp_params,
)

from transformers.trainer_callback import (
    DefaultFlowCallback,
    ProgressCallback,
    TrainerState,
)
from transformers.trainer_pt_utils import (
    IterableDatasetShard,
)
from transformers.trainer_utils import (
    HPSearchBackend,
    TrainOutput,
    has_length,
    speed_metrics,
)
from transformers.utils import (
    WEIGHTS_NAME,
    is_apex_available,
    is_in_notebook,
    is_sagemaker_mp_enabled,
    is_torch_tpu_available,
    logging,
)

# from torch.optim.optimizer import StateDict, params_t
import wandb

from metrics import f1
import lora_ms
from lora import LoRALinear
from utils import OPT_PERTURBATION_LEVEL_TO_REGEX


DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from .utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

if is_apex_available():
    from apex import amp

if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl


if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")
else:
    IS_SAGEMAKER_MP_POST_1_10 = False

if TYPE_CHECKING:
    pass

logger = logging.get_logger(__name__)

# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"


class OurTrainer(Trainer):

    # ZO-Bench added: new parameters to our traininer
    def __init__(self, evaluate_func, dev_samples, eval_samples, perturb_module_regex, *args, **kwargs):
        super().__init__(*args, **kwargs)  # Initialize the base class
        self.evaluate_func = evaluate_func
        self.dev_samples = dev_samples
        self.eval_samples = eval_samples
        self.perturb_module_regex = perturb_module_regex

    def _inner_training_loop(
            self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):

        self._train_batch_size = batch_size
        # Data loader and number of training steps
        self.args.dataloader_num_workers = 0
        train_dataloader = self.get_train_dataloader()
        

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torch.distributed.launch)."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = (
                is_sagemaker_mp_enabled()
        )
        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        model = self._wrap_model(self.model_wrapped)

        if is_sagemaker_mp_enabled() and resume_from_checkpoint is not None:
            self._load_from_checkpoint(resume_from_checkpoint, model)

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # overload the optimizer here
        if args.trainer == "zo_adam":
            self.optimizer = Adam(self.model.parameters(), lr=args.learning_rate)
            # self.optimizer = {name: Adam([param], lr=args.learning_rate) for name, param in self.model.named_parameters()}
            # assert args.lr_scheduler
            assert args.lr_scheduler_type == 'constant', "we did not implement lr_schedule."
        elif args.trainer == "zo_sgd":
            self.optimizer = SGD(self.model.parameters(), lr=args.learning_rate, momentum=args.momentum)
            # self.optimizer = {name: SGD([param], lr=args.learning_rate) for name, param in self.model.named_parameters()}
            # print(f"### args.lr_scheduler: {args.lr_scheduler_type}")
            assert args.lr_scheduler_type == 'constant', "we did not implement lr_schedule."
        else:
            assert args.lr_scheduler_type == 'constant', "we did not implement lr_schedule."
            if args.optimizer == "adam":
                self.optimizer = Adam(self.model.parameters(), lr=args.learning_rate)
            elif args.optimizer == "sgd":
                self.optimizer = SGD(self.model.parameters(), lr=args.learning_rate, momentum=args.momentum)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")
        logger.info(
            f"  Number of trainable parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
        )

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
                    "batches in the first epoch. If this takes a lot of time, you can add the `--ignore_data_skip` "
                    "flag to your launch command, but you will resume the training on data already seen by your model."
                )
                if self.is_local_process_zero() and not args.disable_tqdm:
                    steps_trained_progress_bar = tqdm(total=steps_trained_in_current_epoch)
                    steps_trained_progress_bar.set_description("Skipping the first batches")

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                is_random_sampler = hasattr(train_dataloader, "sampler") and isinstance(
                    train_dataloader.sampler, RandomSampler
                )
                if not is_random_sampler:
                    # We just need to begin an iteration to create the randomization of the sampler.
                    # That was before PyTorch 1.11 however...
                    for _ in train_dataloader:
                        break
                else:
                    # Otherwise we need to call the whooooole sampler cause there is some random operation added
                    # AT THE VERY END!
                    _ = list(train_dataloader.sampler)

        # Main training loop
        total_steps = 0

        
        # para for adaSelect
        zeroshot_acc = 0
        best_acc = 0
        patience = 2
        tmp_patience = 0
        ada_stage = 1
        
        for epoch in range(epochs_trained, num_train_epochs):
            print(f"-------------------------- Training Epoch {epoch} --------------------------")
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            elif hasattr(train_dataloader, "dataset") and isinstance(train_dataloader.dataset, IterableDatasetShard):
                train_dataloader.dataset.set_epoch(epoch)

            if is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [args.device]).per_device_loader(args.device)
                epoch_iterator = parallel_loader
            else:
                epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            step = -1
            # Start one epoch training
            for step, inputs in enumerate(epoch_iterator):
                
                if total_steps == 0:
                    print(
                        f"=========================> Evaluating at beginning... <=========================")
                    val_metrics = self.evaluate_func([], self.dev_samples)
                    test_metrics = self.evaluate_func([], self.eval_samples)
                    if "accuracy" in test_metrics:
                        zeroshot_acc = test_metrics["accuracy"]
                        self.log({"test_acc": test_metrics["accuracy"], "val_acc": val_metrics["accuracy"]})
                        wandb.log({"test_acc": test_metrics["accuracy"], "val_acc": val_metrics["accuracy"]})
                    else:
                        keys = list(test_metrics.keys())
                        log_dict = {}
                        for k in keys:
                            log_dict['test_' + k] = test_metrics[k]
                            log_dict['val_' + k] = val_metrics[k]
                        self.log(log_dict)
                        wandb.log(log_dict)

                total_steps += 1

                # torch.cuda.synchronize()
                step_start_time = time.time()

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                # MeZO added: estimate gradient
                if args.trainer in ["zo_sgd", "zo_adam", "zo_sign_opt"]:
                    if args.selective:
                        print("selective optimization")
                        tr_loss_step, l2_A, l2_B, projected_grad = self.zo_step_s(model, inputs)
                        wandb.log({"loss": tr_loss_step, "l2_A": l2_A, "l2_B": l2_B, "projected_grad": projected_grad})
                        
                    elif args.lora_bcd:
                        tr_loss_step, l2_A, l2_B, projected_grad, mean_A, mean_B, var_A, var_B, cosine_similarity_A, cosine_similarity_B = self.zo_step_bcd(model, inputs, total_steps)
                        if cosine_similarity_A != -2 and cosine_similarity_B != -2:
                            wandb.log({"loss": tr_loss_step, "l2_A": l2_A, "l2_B": l2_B, "projected_grad": projected_grad, "mean_A": mean_A, "mean_B": mean_B, "var_A": var_A, "var_B": var_B, "cosine_similarity_A": cosine_similarity_A, "cosine_similarity_B": cosine_similarity_B})
                        else:
                            wandb.log({"loss": tr_loss_step, "l2_A": l2_A, "l2_B": l2_B, "projected_grad": projected_grad, "mean_A": mean_A, "mean_B": mean_B, "var_A": var_A, "var_B": var_B})
                    elif args.q == 1:
                        tr_loss_step, l2_A, l2_B, projected_grad, mean_A, mean_B, var_A, var_B, cosine_similarity_A, cosine_similarity_B = self.zo_step(model, inputs, total_steps)
                        if cosine_similarity_A != -2 and cosine_similarity_B != -2:
                            wandb.log({"loss": tr_loss_step, "l2_A": l2_A, "l2_B": l2_B, "projected_grad": projected_grad, "mean_A": mean_A, "mean_B": mean_B, "var_A": var_A, "var_B": var_B, "cosine_similarity_A": cosine_similarity_A, "cosine_similarity_B": cosine_similarity_B})
                        else:
                            wandb.log({"loss": tr_loss_step, "l2_A": l2_A, "l2_B": l2_B, "projected_grad": projected_grad, "mean_A": mean_A, "mean_B": mean_B, "var_A": var_A, "var_B": var_B})
                            
                    else:
                        raise ValueError(f"q={args.q} is not supported.")
                elif args.trainer == "zo_conserv":
                    tr_loss_step = self.zo_conserv_step(model, inputs)
                elif args.trainer == "forward_grad":
                    tr_loss_step = self.forward_grad_step(model, inputs)
                else:
                    if (
                            ((step + 1) % args.gradient_accumulation_steps != 0)
                            and args.local_rank != -1
                            and args._no_sync_in_gradient_accumulation
                    ):
                        # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                        with model.no_sync():
                            tr_loss_step = self.training_step(model, inputs)
                    else:
                        tr_loss_step, l2_A, l2_B, mean_A, mean_B, var_A, var_B, cosine_similarity_A, cosine_similarity_B = self.training_step(model, inputs, total_steps)
                        if cosine_similarity_A != -2 and cosine_similarity_B != -2:
                            wandb.log({"loss": tr_loss_step, "l2_A": l2_A, "l2_B": l2_B, "mean_A": mean_A, "mean_B": mean_B, "var_A": var_A, "var_B": var_B, "cosine_similarity_A": cosine_similarity_A, "cosine_similarity_B": cosine_similarity_B})
                        else:
                            wandb.log({"loss": tr_loss_step, "l2_A": l2_A, "l2_B": l2_B, "mean_A": mean_A, "mean_B": mean_B, "var_A": var_A, "var_B": var_B})

                if (
                        args.logging_nan_inf_filter
                        and not is_torch_tpu_available()
                        and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))


                if (step + 1) % args.gradient_accumulation_steps == 0 or (
                        # last step in epoch but step is always smaller than gradient_accumulation_steps
                        steps_in_epoch <= args.gradient_accumulation_steps
                        and (step + 1) == steps_in_epoch
                ):
                    # MeZO added: update model with the estimated gradient
                    if args.trainer in ["zo_sgd", "zo_adam", "zo_sign_opt", "zo_conserv"]:
                        self.zo_update(model)
                    elif args.trainer == "forward_grad":
                        self.forward_grad_update(model)
                    else:
                        self.gradient_update(model)

                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                    self.control.should_save = False
                    # self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                # torch.cuda.synchronize()
                train_step_duration = time.time() - step_start_time

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break

                if self.args.eval_steps is not None and (total_steps + 1) % self.args.eval_steps == 0:
                    print(
                        f"=========================> Evaluating at step {total_steps + 1}... <=========================")
                    val_metrics = self.evaluate_func([], self.dev_samples)
                    test_metrics = self.evaluate_func([], self.eval_samples)
                    if "accuracy" in test_metrics:
                        if args.adaSelect and test_metrics["accuracy"] > zeroshot_acc:                        
                            if test_metrics["accuracy"] >= best_acc:
                                best_acc = test_metrics["accuracy"]
                                lora_state_dict = {k: v for k, v in model.state_dict().items() if "lora" in k}
                                torch.save(lora_state_dict, f'/home/wzy/code3/ZO-LLM/zo-bench/tmp_best_lora_model/{self.args.project}_{self.args.wandb_name}.pth')
                                tmp_patience = 0
                            else:
                                if ada_stage <= 3:
                                    tmp_patience += 1
                                    if tmp_patience >= patience:
                                        lora_state_dict = torch.load(f'/home/wzy/code3/ZO-LLM/zo-bench/tmp_best_lora_model/{self.args.project}_{self.args.wandb_name}.pth')
                                        model.load_state_dict(lora_state_dict, strict=False) 
                                        tmp_patience = 0
                                        args.selective = True
                                        if ada_stage == 1 :
                                            args.selective_ratio = 0.5
                                            patience = 4
                                        elif ada_stage == 2:
                                            args.selective_ratio = 0.7
                                            patience = 6
                                        elif ada_stage == 3:
                                            args.selective_ratio = 0.9
                                        ada_stage += 1
                                    
                                
                                
                        self.log({"test_acc": test_metrics["accuracy"], "val_acc": val_metrics["accuracy"]})
                        wandb.log({"test_acc": test_metrics["accuracy"], "val_acc": val_metrics["accuracy"]})
                    else:
                        keys = list(test_metrics.keys())
                        log_dict = {}
                        for k in keys:
                            log_dict['test_' + k] = test_metrics[k]
                            log_dict['val_' + k] = val_metrics[k]
                        self.log(log_dict)
                        wandb.log(log_dict)

                max_memory_allocated = 0
                for device_id in range(torch.cuda.device_count()):
                    # this is not accurate since max memory does not happen simultaneously across all devices
                    max_memory_allocated += torch.cuda.max_memory_allocated(device_id)
                self.log({"peak_mem": max_memory_allocated / 1024 ** 3,
                          "step_consumption": train_step_duration * 1000})
                wandb.log({"peak_mem": max_memory_allocated / 1024 ** 3,
                           "step_consumption": train_step_duration * 1000})

            if step < 0:
                # Why would this happen? I don't know, but let's be safe.
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sur the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.local_rank != -1:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        wandb.log(metrics)
        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint.
        if self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if checkpoint != self.state.best_model_checkpoint:
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        return TrainOutput(self.state.global_step, train_loss, metrics)

    ############## MeZO ##############

    def gradient_update(self, model):
        args = self.args

        # Gradient clipping
        if args.max_grad_norm is not None and args.max_grad_norm > 0 and not self.deepspeed:
            # deepspeed does its own clipping

            # if self.do_grad_scaling:
            #     # Reduce gradients first for XLA
            #     if is_torch_tpu_available():
            #         gradients = xm._fetch_gradients(self.optimizer)
            #         xm.all_reduce("sum", gradients, scale=1.0 / xm.xrt_world_size())
            #     # AMP: gradients need unscaling
            #     self.scaler.unscale_(self.optimizer)

            if is_sagemaker_mp_enabled() and args.fp16:
                self.optimizer.clip_master_grads(args.max_grad_norm)
            elif hasattr(self.optimizer, "clip_grad_norm"):
                # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                self.optimizer.clip_grad_norm(args.max_grad_norm)
            elif hasattr(model, "clip_grad_norm_"):
                # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                model.clip_grad_norm_(args.max_grad_norm)
            else:
                # Revert to normal clipping otherwise, handling Apex or full precision
                nn.utils.clip_grad_norm_(
                    amp.master_params(self.optimizer) if self.use_apex else model.parameters(),
                    args.max_grad_norm,
                )

        # Optimizer step
        optimizer_was_run = True
        if self.deepspeed:
            pass  # called outside the loop
        elif is_torch_tpu_available():
            if self.do_grad_scaling:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                xm.optimizer_step(self.optimizer)
        # elif self.do_grad_scaling:
        #     scale_before = self.scaler.get_scale()
        #     self.scaler.step(self.optimizer)
        #     self.scaler.update()
        #     scale_after = self.scaler.get_scale()
        #     optimizer_was_run = scale_before <= scale_after
        else:
            self.optimizer.step()

        if optimizer_was_run and not self.deepspeed:
            self.lr_scheduler.step()
        model.zero_grad()

    def zo_perturb_parameters_lora_bcd(self, random_seed=None, scaling_factor=1, total_steps = None):
        """
        Perturb the parameters with random vector z.
        Input:
        - random_seed: random seed for MeZO in-place perturbation (if it's None, we will use self.zo_random_seed)
        - scaling_factor: theta = theta + scaling_factor * z * eps
        """

        # Set the random seed to ensure that we sample the same z for perturbation/update
        torch.manual_seed(random_seed if random_seed is not None else self.zo_random_seed)

        for name, param in self.named_parameters_to_optim:

            z = 0
            if total_steps % 220 <= 200:
                if "lora_B" in name:  
                    z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            else:
                if "lora_A" in name:  
                    z = torch.normal(mean=0, std=0.5, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                
            # if "lora_B" in name:  
            #     z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            # else:
            #     z = torch.normal(mean=0, std=0.1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
        
            param.data = param.data + scaling_factor * z * self.args.zo_eps 
  
            
    def zo_perturb_parameters(self, random_seed=None, scaling_factor=1):
        """
        Perturb the parameters with random vector z.
        Input:
        - random_seed: random seed for MeZO in-place perturbation (if it's None, we will use self.zo_random_seed)
        - scaling_factor: theta = theta + scaling_factor * z * eps
        """

        # Set the random seed to ensure that we sample the same z for perturbation/update
        torch.manual_seed(random_seed if random_seed is not None else self.zo_random_seed)

        for name, param in self.named_parameters_to_optim:
       
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            param.data = param.data + scaling_factor * z * self.args.zo_eps

    def zo_forward(self, model, inputs):
        """
        Get (no gradient) loss from the model. Dropout is turned off too.
        """
        model.eval()
        if self.args.non_diff:
            # Non-differentiable objective (may require autoregressive generation)
            return self.zo_forward_nondiff(model, inputs)

        with torch.inference_mode():
            inputs = self._prepare_inputs(inputs)
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1:
                # Warning: this is copied from the original Huggingface Trainer. Untested.
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
        return loss.detach()

    def zo_forward_nondiff(self, model, inputs):
        """
        Get (no gradient) non-diffiable loss from the model.
        """
        model.eval()
        assert self.args.task_name == "SQuAD", "Non differentiable objective only supports SQuAD for now."

        with torch.inference_mode():
            inputs = self._prepare_inputs(inputs)
            args = self.args
            outputs = self.model.generate(
                inputs["input_ids"], do_sample=args.sampling, temperature=args.temperature,
                num_beams=args.num_beams, top_p=args.top_p, top_k=args.top_k,
                max_new_tokens=min(args.max_new_tokens, args.max_length - inputs["input_ids"].size(1)),
                num_return_sequences=1,
                eos_token_id=[self.tokenizer.encode(args.eos_token, add_special_tokens=False)[-1],
                              self.tokenizer.eos_token_id],
            )
            output_text = []
            for i in range(len(outputs)):
                output_text.append(
                    self.tokenizer.decode(outputs[i][inputs["input_ids"].size(1):], skip_special_tokens=True).strip())
            f1s = [f1(output_text[i], inputs['gold'][i]) for i in range(len(output_text))]

        return -torch.tensor(np.mean(f1s), dtype=torch.float32)


    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], total_steps) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()


        l2_norm_A = 0
        l2_norm_B = 0
        mean_A = 0
        mean_B = 0
        var_A = 0
        var_B = 0
        module_num = 0
        cosine_similarity_A = 0
        cosine_similarity_B = 0
        for name, module in model.named_modules():
            if isinstance(module, LoRALinear):   
                module_num += 1
                
                A = module.lora_A.detach()
                B = module.lora_B.detach()
        
                l2_norm_A += torch.norm(A, p=2)       
                l2_norm_B += torch.norm(B, p=2) 
                
                mean_A += A.mean()
                mean_B += B.mean()
                var_A += A.var()
                var_B += B.var()
                
                if total_steps == 100:
                    module.lora_A_history_data = module.lora_A.data
                    module.lora_B_history_data = module.lora_B.data
                if total_steps > 100 and total_steps % 100 == 0:
                    # his_direction_A, his_direction_B = module.get_directions(module.lora_A_history_data, module.lora_B_history_data)
                    # direction_A, direction_B = module.get_directions(module.lora_A.data, module.lora_B.data)
                    # cosine_similarity_A += module.compute_overall_direction_similarity(his_direction_A, direction_A)     
                    # cosine_similarity_B += module.compute_overall_direction_similarity(his_direction_B, direction_B)     
                    cosine_similarity_A += torch.mean(torch.nn.functional.cosine_similarity(module.lora_A.data, module.lora_A_history_data)) 
                    cosine_similarity_B += torch.mean(torch.nn.functional.cosine_similarity(module.lora_B.data + 1e-6, module.lora_B_history_data + 1e-6))
                    module.lora_A_history_data = module.lora_A.data
                    module.lora_B_history_data = module.lora_B.data
                
        l2_norm_A = l2_norm_A / module_num
        l2_norm_B = l2_norm_B / module_num  
        mean_A /= module_num
        mean_B /= module_num
        var_A /= module_num
        var_B /= module_num
        cosine_similarity_A = cosine_similarity_A / module_num
        cosine_similarity_B = cosine_similarity_B / module_num
        
        if total_steps <= 100 or total_steps % 100 != 0:
            cosine_similarity_A = -2
            cosine_similarity_B = -2
        
        return loss.detach(), l2_norm_A, l2_norm_B, mean_A, mean_B, var_A, var_B, cosine_similarity_A, cosine_similarity_B




    @torch.no_grad()
    def zo_step(self, model, inputs, total_steps):
        """
        Estimate gradient by MeZO. Return the loss from f(theta + z)
        """
        args = self.args

        # What parameters to optimize
        self.named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))
                # # TODO avoid init the memory for grad.
                # param.grad = torch.zeros_like(param.data)
                param.grad = None  # Make sure the grad is empty and will not be updated.

        # Sample the random seed for sampling z
        self.zo_random_seed = np.random.randint(1000000000)

        # First function evaluation
        # NOTE: when sparse_grad is set to True, it will also check the args.gradient_sparsity,
        # so it does not necessarily use sparse grad.
        self.zo_perturb_parameters(scaling_factor=1)
        loss1 = self.zo_forward(model, inputs)

        # Second function evaluation
        assert args.q == 1, "only support q=1 for the memory efficiency. If you want to implement q>1, need to store random seeds to save memory. In addition, we need to set different random seed for different z in the q-loop."
        for _ in range(args.q):  # TODO shall we change the seed?
            if self.args.perturbation_mode == "one_side":
                self.zo_perturb_parameters(scaling_factor=-1)
                loss2 = self.zo_forward(model, inputs)
                self.projected_grad = ((loss1 - loss2) / self.args.zo_eps).item()
            else:  # two side perturbation
                self.zo_perturb_parameters(scaling_factor=-2)
                loss2 = self.zo_forward(model, inputs)
                self.projected_grad = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()

                # Reset model back to its parameters at start of step
                self.zo_perturb_parameters(scaling_factor=1)

            # Set the random seed to ensure that we sample the same z for perturbation/update
            torch.manual_seed(self.zo_random_seed)
            
            for name, param in self.named_parameters_to_optim:
                # Resample z
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device,
                                 dtype=param.data.dtype)

                if args.trainer == "zo_sign_opt":
                    # ----signOpt_orig
                    # TODo why do we multiply lr here? We will multiply lr twice?
                    graddiff_times_z = np.sign(self.projected_grad) * z
                    # ----signOpt_mul_sign
                    # graddiff_times_z = self._get_learning_rate() * torch.sign(self.projected_grad * z)
                else:
                    # ----mezo original
                    graddiff_times_z = self.projected_grad * z

                param.grad = graddiff_times_z / args.q  # NOTE this q division does not work for q>1.
                
                self.optimizer.step()  # will only update grad that is not None.
                # param.data = param.data - graddiff_times_z / args.q  # NOTE this q division does not work for q>1.
                param.grad = None  # avoid further update.

        
        # for name, param in self.named_parameters_to_optim:
        #     param.grad = param.grad / args.q
        
        l2_norm_A = 0
        l2_norm_B = 0
        mean_A = 0
        mean_B = 0
        var_A = 0
        var_B = 0
        module_num = 0
        cosine_similarity_A = 0
        cosine_similarity_B = 0
        for name, module in model.named_modules():
            if isinstance(module, LoRALinear):   
                module_num += 1
                
                A = module.lora_A.detach()
                B = module.lora_B.detach()
        
                l2_norm_A += torch.norm(A, p=2)       
                l2_norm_B += torch.norm(B, p=2) 
                
                mean_A += A.mean()
                mean_B += B.mean()
                var_A += A.var()
                var_B += B.var()
                
                if total_steps == 200:
                    module.lora_A_history_data = module.lora_A.data
                    module.lora_B_history_data = module.lora_B.data
                if total_steps > 200 and total_steps % 200 == 0:
                    # his_direction_A, his_direction_B = module.get_directions(module.lora_A_history_data, module.lora_B_history_data)
                    # direction_A, direction_B = module.get_directions(module.lora_A.data, module.lora_B.data)
                    # cosine_similarity_A += module.compute_overall_direction_similarity(his_direction_A, direction_A)     
                    # cosine_similarity_B += module.compute_overall_direction_similarity(his_direction_B, direction_B)     
                    cosine_similarity_A += torch.mean(torch.nn.functional.cosine_similarity(module.lora_A.data, module.lora_A_history_data)) 
                    cosine_similarity_B += torch.mean(torch.nn.functional.cosine_similarity(module.lora_B.data, module.lora_B_history_data))
                    module.lora_A_history_data = module.lora_A.data
                    module.lora_B_history_data = module.lora_B.data
                
        l2_norm_A = l2_norm_A / module_num
        l2_norm_B = l2_norm_B / module_num  
        mean_A /= module_num
        mean_B /= module_num
        var_A /= module_num
        var_B /= module_num
        cosine_similarity_A = cosine_similarity_A / module_num
        cosine_similarity_B = cosine_similarity_B / module_num
        
        
        
        if total_steps <= 200 or total_steps % 200 != 0:
            cosine_similarity_A = -2
            cosine_similarity_B = -2

        # No gradient accumulation support
        assert self.args.gradient_accumulation_steps == 1

        return loss1, l2_norm_A, l2_norm_B, self.projected_grad, mean_A, mean_B, var_A, var_B, cosine_similarity_A, cosine_similarity_B

    @torch.no_grad()
    def zo_step_bcd(self, model, inputs, total_steps):
        """
        Estimate gradient by MeZO. Return the loss from f(theta + z)
        """
        args = self.args

        # What parameters to optimize
        self.named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))
                # # TODO avoid init the memory for grad.
                # param.grad = torch.zeros_like(param.data)
                param.grad = None  # Make sure the grad is empty and will not be updated.

        # Sample the random seed for sampling z
        self.zo_random_seed = np.random.randint(1000000000)

        # First function evaluation
        # NOTE: when sparse_grad is set to True, it will also check the args.gradient_sparsity,
        # so it does not necessarily use sparse grad.
        self.zo_perturb_parameters_lora_bcd(scaling_factor=1, total_steps=total_steps)
        loss1 = self.zo_forward(model, inputs)

        # Second function evaluation
        assert args.q == 1, "only support q=1 for the memory efficiency. If you want to implement q>1, need to store random seeds to save memory. In addition, we need to set different random seed for different z in the q-loop."
        for _ in range(args.q):  # TODO shall we change the seed?
            if self.args.perturbation_mode == "one_side":
                self.zo_perturb_parameters_lora_bcd(scaling_factor=-1)
                loss2 = self.zo_forward(model, inputs)
                self.projected_grad = ((loss1 - loss2) / self.args.zo_eps).item()
            else:  # two side perturbation
                self.zo_perturb_parameters_lora_bcd(scaling_factor=-2, total_steps=total_steps)
                loss2 = self.zo_forward(model, inputs)
                self.projected_grad = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()
                print(self.projected_grad)

                # Reset model back to its parameters at start of step
                self.zo_perturb_parameters_lora_bcd(scaling_factor=1, total_steps=total_steps)

            # Set the random seed to ensure that we sample the same z for perturbation/update
            torch.manual_seed(self.zo_random_seed)
  
            for name, param in self.named_parameters_to_optim:
                # Resample z
                z = None
                if total_steps % 220 <= 200:
                    if "lora_B" in name:  
                        z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                else:
                    if "lora_A" in name:  
                        z = torch.normal(mean=0, std=0.5, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)

                
                if z is not None:
                    if args.trainer == "zo_sign_opt":
                        # ----signOpt_orig
                        # TODo why do we multiply lr here? We will multiply lr twice?
                        # graddiff_times_z = np.sign(self.projected_grad) * z                    
                            graddiff_times_z = torch.sign(self.projected_grad * z)
                        # ----signOpt_mul_sign
                        # graddiff_times_z = self._get_learning_rate() * torch.sign(self.projected_grad * z)
                    else:
                        # ----mezo original
                        graddiff_times_z = self.projected_grad * z

                # # previous implementation
                # # no param.grad involved
                # param.data -= self._get_learning_rate() * self.projected_grad * z

                # param.grad += graddiff_times_z.detach()
                # more mem-efficient:
                # run optimizer.step here to avoid caching all grad.
                
                # param.grad = graddiff_times_z / args.q  # NOTE this q division does not work for q>1.
                
                # lambda_ = 20
                
                # if "lora_B" in name:   
                #     param.grad += 5 * param.data                
                #     for group in self.optimizer.param_groups:
                #         group['lr'] = group['lr'] * lambda_
                #     self.optimizer.step() 
                #     for group in self.optimizer.param_groups:
                #         group['lr'] = group['lr'] / lambda_
                # else:
                #     param.grad += 1 * param.data
                #     self.optimizer.step()  # will only update grad that is not None.
                
                # if "lora_A" in name:             
                #     for group in self.optimizer.param_groups:
                #         group['lr'] = group['lr'] / lambda_
                #     self.optimizer.step() 
                #     for group in self.optimizer.param_groups:
                #         group['lr'] = group['lr'] * lambda_
                # else:
                #     param.grad += 1 * param.data
                #     self.optimizer.step()  # will only update grad that is not None.
                
                if total_steps % 220 <= 200:
                    if "lora_B" in name: 
                        param.grad = graddiff_times_z / args.q
                        self.optimizer.step() 
                        param.grad = None
                else:
                    if "lora_A" in name:
                        param.grad = graddiff_times_z / args.q
                        for group in self.optimizer.param_groups:
                            group['lr'] = group['lr'] / 2
                        self.optimizer.step() 
                        for group in self.optimizer.param_groups:
                            group['lr'] = group['lr'] * 2
                        param.grad = None
                    
                # self.optimizer.step() 
                # param.grad = None  # avoid further update.
                  
        
        l2_norm_A = 0
        l2_norm_B = 0
        mean_A = 0
        mean_B = 0
        var_A = 0
        var_B = 0
        module_num = 0
        cosine_similarity_A = 0
        cosine_similarity_B = 0
        
        for name, module in model.named_modules():
            if isinstance(module, LoRALinear):   
                module_num += 1
                A = module.lora_A.detach()
                B = module.lora_B.detach()
        
                l2_norm_A += torch.norm(A, p=2)       
                l2_norm_B += torch.norm(B, p=2) 
                
                mean_A += A.mean()
                mean_B += B.mean()
                var_A += A.var()
                var_B += B.var()
                
                if total_steps == 200:
                    module.lora_A_history_data = module.lora_A.data
                    module.lora_B_history_data = module.lora_B.data
                if total_steps > 200 and total_steps % 200 == 0:
                    # his_direction_A, his_direction_B = module.get_directions(module.lora_A_history_data, module.lora_B_history_data)
                    # direction_A, direction_B = module.get_directions(module.lora_A.data, module.lora_B.data)
                    # cosine_similarity_A += module.compute_overall_direction_similarity(his_direction_A, direction_A)     
                    # cosine_similarity_B += module.compute_overall_direction_similarity(his_direction_B, direction_B)     
                    cosine_similarity_A += torch.mean(torch.nn.functional.cosine_similarity(module.lora_A.data, module.lora_A_history_data)) 
                    cosine_similarity_B += torch.mean(torch.nn.functional.cosine_similarity(module.lora_B.data, module.lora_B_history_data))
                    module.lora_A_history_data = module.lora_A.data
                    module.lora_B_history_data = module.lora_B.data
                
                
                
        l2_norm_A = l2_norm_A / module_num
        l2_norm_B = l2_norm_B / module_num  
        mean_A /= module_num
        mean_B /= module_num
        var_A /= module_num
        var_B /= module_num
        cosine_similarity_A = cosine_similarity_A / module_num
        cosine_similarity_B = cosine_similarity_B / module_num
         
        if total_steps <= 200 or total_steps % 200 != 0:
            cosine_similarity_A = -2
            cosine_similarity_B = -2

        # for name, param in self.named_parameters_to_optim:
        #     param.grad = param.grad / args.q

        # No gradient accumulation support
        assert self.args.gradient_accumulation_steps == 1

        return loss1, l2_norm_A, l2_norm_B, self.projected_grad, mean_A, mean_B, var_A, var_B, cosine_similarity_A, cosine_similarity_B
    
    
    def zo_step_s(self, model, inputs):
        """
        Estimate gradient by MeZO. Return the loss from f(theta + z)
        """
        args = self.args

        # What parameters to optimize
        self.named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))
                # # TODO avoid init the memory for grad.
                # param.grad = torch.zeros_like(param.data)
                param.grad = None  # Make sure the grad is empty and will not be updated.

        # Sample the random seed for sampling z
        self.zo_random_seed = np.random.randint(1000000000)
        # self.zo_random_seed = np.random.randint(100000000)

        # First function evaluation
        # NOTE: when sparse_grad is set to True, it will also check the args.gradient_sparsity,
        # so it does not necessarily use sparse grad.
        self.zo_perturb_parameters(scaling_factor=1)
        loss1 = self.zo_forward(model, inputs)

        # Second function evaluation
        assert args.q == 1, "only support q=1 for the memory efficiency. If you want to implement q>1, need to store random seeds to save memory. In addition, we need to set different random seed for different z in the q-loop."
        for _ in range(args.q):  # TODO shall we change the seed?
            if self.args.perturbation_mode == "one_side":
                self.zo_perturb_parameters(scaling_factor=-1)
                loss2 = self.zo_forward(model, inputs)
                self.projected_grad = ((loss1 - loss2) / self.args.zo_eps).item()
            else:  # two side perturbation
                self.zo_perturb_parameters(scaling_factor=-2)
                loss2 = self.zo_forward(model, inputs)
                self.projected_grad = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()

                # Reset model back to its parameters at start of step
                self.zo_perturb_parameters(scaling_factor=1)

            # Set the random seed to ensure that we sample the same z for perturbation/update
            torch.manual_seed(self.zo_random_seed)

            for name, param in self.named_parameters_to_optim:
                # Resample z
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device,
                                 dtype=param.data.dtype)

                if args.trainer == "zo_sign_opt":
                    # ----signOpt_orig
                    # TODo why do we multiply lr here? We will multiply lr twice?
                    # graddiff_times_z = np.sign(self.projected_grad) * z
                    graddiff_times_z = torch.sign(self.projected_grad * z)
                    # ----signOpt_mul_sign
                    # graddiff_times_z = self._get_learning_rate() * torch.sign(self.projected_grad * z)
                else:
                    # ----mezo original
                    graddiff_times_z = self.projected_grad * z

                # # previous implementation
                # # no param.grad involved
                # param.data -= self._get_learning_rate() * self.projected_grad * z

                # param.grad += graddiff_times_z.detach()
                # more mem-efficient:
                # run optimizer.step here to avoid caching all grad.
                param.grad = graddiff_times_z / args.q  # NOTE this q division does not work for q>1.
        
        
        def apply_importance_mask(para, importance_mask):          
            assert para.grad is not None, f"{module} has no grad"
            if para.grad is not None:
                para.grad *= importance_mask.unsqueeze(dim=-1).to(para.grad.device)
                    
        def compute_importance_mask(activation, ini_threshold):
                """Compute the importance mask based on the provided method."""
                # activation, kwargs['ini_threshold'], kwargs['cluster_constructure_method'], cluster_indice
                device = activation[0].device
                # dist.barrier()
                # dist.all_reduce(activation, op=dist.ReduceOp.AVG)
                activation = activation.to(dtype=torch.float32)

                threshold = torch.quantile(activation, ini_threshold)
                importance_mask = (activation >= threshold).float().to(device)
                return importance_mask
            
        def auto_compute_importance_mask(activation):
                """Compute the importance mask based on the provided method."""
                # activation, kwargs['ini_threshold'], kwargs['cluster_constructure_method'], cluster_indice
                device = activation[0].device
                activation = activation
                
                clamped_activation = torch.clamp(activation, min=0, max=65504.0).to(dtype=torch.float16)
                clamped_activation = torch.nan_to_num(activation) # 将 nan 转换为 0
                
                min_val = activation.min()
                max_val = activation.max()
 
                normalized_activation = (clamped_activation - min_val) / (max_val - min_val)
                importance_mask = (torch.bernoulli(normalized_activation)).float().to(device)

                return importance_mask
        
            
        def compute_random_importance_mask(activation, ini_threshold):
            """Compute the importance mask based on random values."""
            device = activation.device
            shape = activation.shape

            # Generate a random tensor with the same shape and device as activation
            random_tensor = torch.rand(shape, device=device)

            # Calculate the threshold for the given proportion (ini_threshold)
            threshold = torch.quantile(random_tensor, ini_threshold)

            # Generate the mask based on the threshold
            importance_mask = (random_tensor > threshold).float()

            return importance_mask
        
            
        for name, module in model.named_modules():
            
            if isinstance(module, lora_ms.LoRALinear):                               
                if not args.bcd:
                    importance_mask_Lora_A = compute_importance_mask(module.temporal_activation_sum['lora_A'], args.selective_ratio)
                    importance_mask_Lora_B = compute_importance_mask(module.temporal_activation_sum['lora_B'], args.selective_ratio)
                    
                    # importance_mask_Lora_A = auto_compute_importance_mask(module.temporal_activation_sum['lora_A'])
                    # importance_mask_Lora_B = auto_compute_importance_mask(module.temporal_activation_sum['lora_B'])
                    
                    # if (name == 'model.decoder.layers.0.self_attn.v_proj'):
                    #     print(importance_mask_Lora_A)
                    
                    apply_importance_mask(module.lora_A, importance_mask_Lora_A)
                    apply_importance_mask(module.lora_B, importance_mask_Lora_B)
                    module.reset_activation_stats(args.selective_ratio)
                else:
                    module.bcd_compute_importance_mask(args.selective_ratio)
                    module.bcd_apply_importance_mask()
                    module.reset_activation_stats(args.selective_ratio)
                    if not module.bcd_activation:
                        module.lora_A.grad = None
                        module.lora_B.grad = None
        
                
        self.optimizer.step()  # will only update grad that is not None.
        # param.data = param.data - graddiff_times_z / args.q  # NOTE this q division does not work for q>1.
        for name, param in self.named_parameters_to_optim:
            param.grad = None  # avoid further update.

        # for name, param in self.named_parameters_to_optim:
        #     param.grad = param.grad / args.q

        # No gradient accumulation support
        assert self.args.gradient_accumulation_steps == 1
        
        l2_norm_A = 0
        l2_norm_B = 0
        module_num = 0
        for name, module in model.named_modules():
            if isinstance(module, lora_ms.LoRALinear):   
                module_num += 1
                l2_norm_A += torch.norm(module.lora_A, p=2)       
                l2_norm_B += torch.norm(module.lora_B, p=2) 
        l2_norm_A = l2_norm_A / module_num
        l2_norm_B = l2_norm_B / module_num     

        return loss1, l2_norm_A, l2_norm_B, self.projected_grad



    @torch.no_grad()
    def zo_conserv_step(self, model, inputs):
        """
        Estimate gradient by MeZO. Return the loss from f(theta + z)
        update in the conservative way, i.e. 
        reject the update if it's not decreasing
        """
        args = self.args

        # What parameters to optimize
        self.named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))
                param.grad = None

        loss0 = self.zo_forward(model, inputs)

        # Sample the random seed for sampling z
        self.zo_random_seed = np.random.randint(1000000000)

        # First function evaluation
        self.zo_perturb_parameters(scaling_factor=1)
        loss1 = self.zo_forward(model, inputs)

        # Second function evaluation
        if self.args.perturbation_mode == "one_side":
            self.zo_perturb_parameters(scaling_factor=-1)
            loss2 = self.zo_forward(model, inputs)
            self.projected_grad = ((loss1 - loss2) / self.args.zo_eps).item()
        else:  # two side perturbation
            self.zo_perturb_parameters(scaling_factor=-2)
            loss2 = self.zo_forward(model, inputs)
            self.projected_grad = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()

            # Reset model back to its parameters at start of step
            self.zo_perturb_parameters(scaling_factor=1)

        def update_params(sign=1.0):
            # Set the random seed to ensure that we sample the same z for perturbation/update
            torch.manual_seed(self.zo_random_seed)
            for name, param in self.named_parameters_to_optim:
                # Resample z
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device,
                                 dtype=param.data.dtype)

                if args.trainer == "zo_sign_opt":
                    # ----signOpt_orig
                    # TODo why do we multiply lr here? We will multiply lr twice?
                    graddiff_times_z = np.sign(self.projected_grad) * z
                    # ----signOpt_mul_sign
                    # graddiff_times_z = self._get_learning_rate() * torch.sign(self.projected_grad * z)
                else:
                    # ----mezo original
                    graddiff_times_z = self.projected_grad * z

                # # previous implementation
                # # no param.grad involved
                # param.data -= self._get_learning_rate() * self.projected_grad * z

                # param.grad += graddiff_times_z.detach()
                # more mem-efficient:
                # run optimizer.step here to avoid caching all grad.
                param.grad = sign * graddiff_times_z
                # self.optimizer[name].step()
                self.optimizer.step()
                # param.data = param.data - graddiff_times_z / args.q
                param.grad = None

        update_params()
        loss1 = self.zo_forward(model, inputs)

        update_params(sign=-2.0)
        loss2 = self.zo_forward(model, inputs)

        # conduct the update in the conservative way
        # choose from the three and take the minimum loss one
        if loss1 > loss0:
            if loss0 < loss2:
                update_params()
        else:
            if loss1 < loss2:
                update_params(2.0)

        # No gradient accumulation support
        assert self.args.gradient_accumulation_steps == 1

        return loss1

    def zo_update(self, model):
        """
        Update the parameters with the estimated gradients.
        """
        # # Optimizer step
        # self.optimizer.step()
        # print(type(self.optimizer), self.optimizer)
        self.lr_scheduler.step()  # NOTE When we use own optimizer, this will no longer update the lr anymore.
        # self.optimizer.zero_grad()
        # model.zero_grad()

    @staticmethod
    @torch.no_grad()
    def functional_call_loss(params, names, buffers, model, batch):
        params = {k: v for k, v in zip(names, params)}
        outputs = functional_call(model, (params, buffers), tuple(), kwargs=batch)
        return outputs

    def forward_grad_step(self, model, inputs):
        """
        Forward Gradient Method

        Paper: Gradients without Backpropagation
        https://arxiv.org/pdf/2202.08587.pdf
        """
        args = self.args
        # print(model.__dict__)
        # What parameters to optimize
        self.named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))
                param.grad = None
                # this is not memory efficient.
                # param.grad = torch.zeros_like(param.data)

        # Sample the random seed for sampling vs
        self.zo_random_seed = np.random.randint(1000000000)
        torch.manual_seed(self.zo_random_seed)

        loss = 0
        vs = [torch.randn_like(p) for _, p in self.named_parameters_to_optim]

        assert args.q == 1, "q > 1"

        # fixme: this is a workaround for device map error when using jvp
        inputs = {
            k: v.to(device=model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()
        }
        f = partial(
            self.functional_call_loss,
            names=[n for n, _ in self.named_parameters_to_optim], buffers=dict(model.named_buffers()),
            model=model, batch=inputs
        )

        # jvp profiling
        # torch.cuda.reset_peak_memory_stats()
        loss_, jvp_ = jvp(f, (list([p for _, p in self.named_parameters_to_optim]),), (vs,))
        # print(f"JVP peak memory usage: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB")

        # print(len(jvp_), jvp_[0], jvp_[1].shape, len(jvp_[2][0][0]))
        # for v, p in zip(vs, [p for _, p in self.named_parameters_to_optim]):
        #     p.grad += v * jvp_[0].to(p.device)
        jvp_ = jvp_[0]
        with torch.no_grad():
            for v, (n, p) in zip(vs, [(n, p) for n, p in self.named_parameters_to_optim]):
                # p.grad += v * jvp_[0].to(p.device)
                # p.grad.add_(v * jvp_[0].to(p.device))
                # grad = v * jvp_[0].to(p.device) / args.q

                if "bias" not in n and "layer_norm" not in n and "layernorm" not in n:
                    p.data.sub_(self._get_learning_rate() * (v * jvp_.to(p.device) + args.weight_decay * p.data))
                else:
                    p.data.sub_(self._get_learning_rate() * (v * jvp_.to(p.device)))
        loss += loss_[0].item()

        # for name, param in self.named_parameters_to_optim:
        #     param.grad = param.grad / args.q

        # for name, param in self.named_parameters_to_optim:
        #     if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
        #         param.data = param.data - self._get_learning_rate() * (param.grad + args.weight_decay * param.data)
        #     else:
        #         param.data = param.data - self._get_learning_rate() * (param.grad)

        return torch.tensor(loss)

    def forward_grad_update(self, model):
        """
        Update the parameters with the estimated gradients from forward_grad_step.
        """
        args = self.args
        # for name, param in self.named_parameters_to_optim:
        #     if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
        #         param.data = param.data - self._get_learning_rate() * (param.grad + args.weight_decay * param.data)
        #     else:
        #         param.data = param.data - self._get_learning_rate() * (param.grad)

        self.lr_scheduler.step()

    ############## Misc overload functions ##############

    def _set_signature_columns_if_needed(self):
        """
        We overload this function for non-differentiable objective training to pass "gold" -- the gold text for the task
        """
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            # Labels may be named label or label_ids, the default data collator handles that.
            self._signature_columns += list(set(["label", "label_ids"] + self.label_names))
            self._signature_columns += ["gold"]

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        We overload this function to fix an FSDP saving bug (before fix, it will likely cause OOM)
        """

        if output_dir is None:
            output_dir = self.args.output_dir

        if is_torch_tpu_available():
            self._save_tpu(output_dir)
        elif is_sagemaker_mp_enabled():
            # Calling the state_dict needs to be done on the wrapped model and on all processes.
            os.makedirs(output_dir, exist_ok=True)
            state_dict = self.model_wrapped.state_dict()
            if self.args.should_save:
                self._save(output_dir, state_dict=state_dict)
            if IS_SAGEMAKER_MP_POST_1_10:
                # 'user_content.pt' indicates model state_dict saved with smp >= 1.10
                Path(os.path.join(output_dir, "user_content.pt")).touch()
        # elif (
        #         self.fsdp is not None
        # ):
        #     from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
        #     full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

        #     # Fix the FSDP loading bug
        #     with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, full_state_dict_config):
        #         state_dict = self.model.state_dict()
        #     # state_dict = self.model.state_dict()

        #     if self.args.should_save:
        #         self._save(output_dir, state_dict=state_dict)
        elif self.deepspeed:
            # this takes care of everything as long as we aren't under zero3
            if self.args.should_save:
                self._save(output_dir)

            if is_deepspeed_zero3_enabled():
                # It's too complicated to try to override different places where the weights dump gets
                # saved, so since under zero3 the file is bogus, simply delete it. The user should
                # either user deepspeed checkpoint to resume or to recover full weights use
                # zero_to_fp32.py stored in the checkpoint.
                if self.args.should_save:
                    file = os.path.join(output_dir, WEIGHTS_NAME)
                    if os.path.isfile(file):
                        # logger.info(f"deepspeed zero3: removing {file}, see zero_to_fp32.py to recover weights")
                        os.remove(file)

                # now save the real model if stage3_gather_16bit_weights_on_model_save=True
                # if false it will not be saved.
                # This must be called on all ranks
                if not self.deepspeed.save_16bit_model(output_dir, WEIGHTS_NAME):
                    logger.warning(
                        "deepspeed.save_16bit_model didn't save the model, since"
                        " stage3_gather_16bit_weights_on_model_save=false. Saving the full checkpoint instead, use"
                        " zero_to_fp32.py to recover weights"
                    )
                    self.deepspeed.save_checkpoint(output_dir)

        elif self.args.should_save:
            self._save(output_dir)

        # Push to the Hub when `save_model` is called by the user.
        if self.args.push_to_hub and not _internal_call:
            self.push_to_hub(commit_message="Model save")
