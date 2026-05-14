# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import threading
from collections import namedtuple
from typing import Any, Dict, Optional, Set, Tuple, Union

import torch
import torch.distributed
from megatron.core import parallel_state
from torch.distributed import ProcessGroup, get_process_group_ranks

from cosmos3._src.imaginaire.checkpointer.base import AbstractCheckpointer
from cosmos3._src.imaginaire.checkpointer.safe_broadcast import broadcast_object
from cosmos3._src.imaginaire.model import ImaginaireModel
from cosmos3._src.imaginaire.utils import distributed, log, misc
from cosmos3._src.imaginaire.utils.easy_io import easy_io

StateDictItemPath = namedtuple("StateDictItemPath", ["state_dict", "save_path"])


class Checkpointer(AbstractCheckpointer):
    """
    Checkpointer for DDP.
    Note: This implementation only supports local filesystem.
    """

    KEYS_TO_SAVE = ["model", "optim", "scheduler", "trainer"]
    KEYS_TO_POSTFIX = {
        "model": "model",
        "optim": "optim",
        "scheduler": "scheduler",
        "trainer": "",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pp_world_size = parallel_state.get_pipeline_model_parallel_world_size()
        ep_world_size = parallel_state.get_expert_model_parallel_world_size()
        assert pp_world_size < 2, "Pipeline Parallelism (PP) is not tested yet."
        assert ep_world_size < 2, "Expert Parallelism (EP) is not tested yet."
        self.mp_world_size = parallel_state.get_model_parallel_group().size()
        if self.mp_world_size > 1 and self.__class__ == Checkpointer:
            raise NotImplementedError(
                "Model Parallelism (MP) is enabled - you should use TensorParallel Checkpointer instead of DDP Checkpointer."
            )
        # DDP rank (with context parallelism considered)
        self.rank_dp_w_cp = parallel_state.get_data_parallel_rank(with_context_parallel=True)
        # Context parallelism rank
        self.cp_rank = parallel_state.get_context_parallel_rank()
        # Model parallelism rank (including Tensor+Pipeline+Expert Parallelisms)
        self.mp_rank = parallel_state.get_model_parallel_group().rank()

        # self.mp_rank = parallel_state.get_model_parallel_group(with_expert_parallel=ep_world_size > 1).rank()
        if self.broadcast_via_filesystem:
            log.info("Broadcasting checkpoint data via the local filesystem.")
        if not self.strict_resume:
            log.warning("Strict resume mode is off. Some model parameters may not be loaded.")

        # collect ranks of all model parallel groups
        all_ranks = [None for _ in range(distributed.get_world_size())]
        torch.distributed.all_gather_object(
            all_ranks, get_process_group_ranks(parallel_state.get_model_parallel_group())
        )
        all_ranks = list(set(tuple(rank) if isinstance(rank, list) else rank for rank in all_ranks))
        for ranks in all_ranks:
            group = torch.distributed.new_group(list(ranks), backend="gloo")
            if distributed.get_rank() in ranks:
                self.mp_gloo_pg = group

        self.print("Checkpointer Initialized.")

    def print(self, message: str):
        """
        Print message to the console. Include the parallelism rank information when verbose is set to True.
        """
        if self.verbose:
            log.info(
                f"[Parallelism Rank: DP-{self.rank_dp_w_cp}, TP-{self.mp_rank}, CP-{self.cp_rank}]: {message}",
                rank0_only=False,
            )
        else:
            log.info(message, rank0_only=True)

    def add_type_postfix_to_checkpoint_path(self, key: str, checkpoint_path: str, model: ImaginaireModel) -> str:
        del model
        assert key in self.KEYS_TO_SAVE
        post_fix = self.KEYS_TO_POSTFIX[key]

        if post_fix:
            _ckpt_path = checkpoint_path.replace(".pt", f"_{post_fix}.pt")
        else:
            _ckpt_path = checkpoint_path
        return _ckpt_path

    def save(
        self,
        model: ImaginaireModel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_scaler: torch.amp.GradScaler,
        iteration: int,
    ) -> None:
        """Save network weights, optimizer parameters, scheduler parameters to a checkpoint.

        Args:
            model (ImaginaireModel): The PyTorch model.
            optimizer (torch.optim.Optimizer): The model optimizer.
            scheduler (torch.optim.lr_scheduler.LRScheduler): The optimization scheduler.
            grad_scaler (torch.amp.GradScaler): The gradient scaler (for mixed precision training).
            iteration (int): Current iteration number.
        """
        self.callbacks.on_save_checkpoint_start(model, iteration)

        checkpoint_file = self.format_checkpoint_filename(model, iteration)
        state_dict = self.generate_save_state_dict(model, optimizer, scheduler, grad_scaler, iteration)
        state_dict = self._map_state_dict_path_during_save(state_dict, checkpoint_file, model)
        if state_dict:
            # Wait for previous saver thread to end.
            if self.save_thread:
                self.save_thread.join()
            # Run the checkpoint saver in a separate thread.
            self.save_thread = threading.Thread(
                target=self._save_worker,
                daemon=False,
                args=(state_dict, checkpoint_file, distributed.get_rank()),
            )
            self.save_thread.start()

        # Note: Checkpoints are saved on a separate thread and this callback is not accurate.
        # Please check logs from on_save_checkpoint_success() for better accuracy
        self.callbacks.on_save_checkpoint_end(model=None, iteration=iteration)

    def _map_state_dict_path_during_save(self, state_dict, checkpoint_file, model) -> dict[str, StateDictItemPath]:
        new_dict = {}
        for key, _state_dict in state_dict.items():
            _ckpt_path = self.add_type_postfix_to_checkpoint_path(key, checkpoint_file, model)
            checkpoint_path = os.path.join(self.save_dirname, _ckpt_path)
            new_dict[key] = StateDictItemPath(_state_dict, checkpoint_path)
        return new_dict

    @misc.timer("checkpoint saving")
    def _save_worker(self, state_dict: dict[str, StateDictItemPath], checkpoint_file: str, rank: int = 0) -> None:
        """Worker to upload checkpoint to object store, spawned with a child thread (in parallel with the training).

        Args:
            state_dict (dict[str, StateDictItemPath]): The state dict of the model/optimizer/scheduler.
            checkpoint_file (str): The file name of the model checkpoint.
            rank (int): GPU device (default: 0).
        """
        try:
            for key, item in state_dict.items():
                self.print(f"Saving {key} to {item.save_path}")
                try:
                    easy_io.dump(
                        item.state_dict,
                        item.save_path,
                        fast_backend=True,  # optional for fast backend, cpu heavy
                        backend_key=self.save_s3_backend_key,
                    )
                    self.print(f"Saved {key} to {item.save_path}")
                except Exception as e:
                    self.print(f"Failed to save {key} to {item.save_path}: {str(e)}")
                    raise  # Re-raise the exception after logging

            # Synchronize only rank 0 of each model parallel group
            if self.mp_world_size > 1:
                torch.distributed.barrier(group=self.mp_gloo_pg)

            # Only rank 0 of MP group and rank 0 of DP with CP updates latest_checkpoint.txt
            if self.mp_rank == 0 and self.rank_dp_w_cp == 0:
                self._write_latest_checkpoint_file(checkpoint_file)

            if distributed.get_rank() == 0:  # only rank 0 saves trained_data_record
                if "trained_data_record" in state_dict["model"].state_dict:
                    self._write_trained_data_record(
                        checkpoint_file, state_dict["model"].state_dict["trained_data_record"]
                    )

            iteration = int(checkpoint_file.replace("iter_", "").replace(".pt", ""))
            self.callbacks.on_save_checkpoint_success(iteration=iteration)
        except Exception as e:  # noqa: BLE001
            log.exception(f"Checkpoint failed to upload: {e}", rank0_only=not self.verbose)

    def format_checkpoint_filename(self, model: ImaginaireModel, iteration: int) -> str:
        """Generate the checkpoint file name.

        Args:
            iteration (int): The current iteration number.

        Returns:
            checkpoint_file (str): The checkpoint file name.
        """
        del self, model
        return f"iter_{iteration:09}.pt"

    @misc.timer("generate saving state dict")
    def generate_save_state_dict(
        self,
        model: ImaginaireModel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_scaler: torch.amp.GradScaler,
        iteration: int,
    ) -> Optional[Dict[str, Any]]:
        state_dict = {}

        if self.rank_dp_w_cp == 0:
            trainer_state = dict(
                grad_scaler=grad_scaler.state_dict(),
                iteration=iteration,
            )
            model_state = model.state_dict()
            optim_state = optimizer.state_dict()
            scheduler_state = scheduler.state_dict()
            self.callbacks.on_save_checkpoint(model, state_dict=trainer_state)

            trainer_state, model_state, optim_state, scheduler_state = misc.to(
                [trainer_state, model_state, optim_state, scheduler_state], device="cpu"
            )

            state_dict = {
                "model": model_state,
                "optim": optim_state,
                "scheduler": scheduler_state,
            }
            if distributed.get_rank() == 0:  # only rank 0 saves trainer state
                state_dict["trainer"] = trainer_state
            return state_dict
        return state_dict

    def load_broadcast_state_dict(
        self, checkpoint_path: str, model: ImaginaireModel, resume_keys: Set
    ) -> dict[str, Any]:
        """
        Load state_dict and broadcast.

        The main steps are:
        1. Download TP-rank-specific checkpoints for every GPU of DDP-rank 0 and CP-rank 0.
        2. Each rank loads its corresponding checkpoint from the local cache or receives it via broadcast.

        This approach ensures that each MP rank loads its specific part of the model, which is
        crucial for Model Parallelism where different parts of the model are distributed across
        multiple GPUs.

        When using Model Parallelism (e.g., Tensor Parallelism), the `broadcast_via_filesystem` option can
        be set to True. This allows each rank to load its specific checkpoint from the local filesystem
        instead of receiving it via network broadcast, which could be more efficient in some cases.

        For standard DDP without TP, `broadcast_via_filesystem` should remain False (default).

        Args:
            checkpoint_path (str): The base path of the checkpoint.
            model (ImaginaireModel): The model being loaded.
            resume_keys (Set): Set of keys to resume from the checkpoint.

        Returns:
            dict[str, Any]: A dictionary containing the loaded state for each resumed key.
        """
        state_dict = {}
        sorted_resume_keys = sorted(resume_keys)
        # Step 1: Download TP-rank-specific checkpoints for every GPU of DDP-rank 0 and CP-rank 0.
        if self.rank_dp_w_cp == 0:
            for key in sorted_resume_keys:
                _ckpt_path = self.add_type_postfix_to_checkpoint_path(key, checkpoint_path, model)
                local_cache_path = os.path.join(self.load_dirname, os.path.basename(_ckpt_path))
                if os.path.exists(local_cache_path):
                    # If the local checkpoint exists, we can directly load it
                    self.print(f"Checkpoint is already in local cache: {local_cache_path}. Loading...")
                    _state_dict = easy_io.load(local_cache_path, fast_backend=True)
                else:

                    _state_dict = easy_io.load(_ckpt_path, fast_backend=True, backend_key=self.load_s3_backend_key)
                    self.print(f"Downloading checkpoint from: {_ckpt_path}")
                    if self.broadcast_via_filesystem:
                        # Save the checkpoint to the local filesystem
                        easy_io.dump(_state_dict, local_cache_path, fast_backend=True)
                state_dict[key] = _state_dict
        # Ensure all ranks wait for the download to complete
        distributed.barrier()

        # Step 2: Broadcast checkpoint data
        log.info(
            "Start broadcasting checkpoint from the source rank to all other ranks in the same DDP group.",
            rank0_only=True,
        )
        for key in sorted_resume_keys:
            if self.broadcast_via_filesystem:
                # Load the checkpoint from the local filesystem for other ranks
                if self.rank_dp_w_cp != 0:
                    _ckpt_path = self.add_type_postfix_to_checkpoint_path(key, checkpoint_path, model)
                    local_cache_path = os.path.join(self.load_dirname, os.path.basename(_ckpt_path))
                    self.print(f"Loading checkpoint from: {local_cache_path}")
                    state_dict[key] = easy_io.load(local_cache_path, fast_backend=True)
            else:
                # Broadcast the checkpoint to all GPUs of the current DDP rank
                group: ProcessGroup = parallel_state.get_data_parallel_group(with_context_parallel=True)
                min_rank = min(get_process_group_ranks(group))

                _state_dict = broadcast_object(
                    state_dict[key] if self.rank_dp_w_cp == 0 else None,
                    min_rank,
                    group=group,
                    device=torch.device(torch.cuda.current_device()),
                )
                if self.rank_dp_w_cp == 0:
                    self.print(f'Broadcasted checkpoint["{key}"] to all other ranks in the same DDP group.')
                else:
                    state_dict[key] = _state_dict
                    self.print(f'Received checkpoint["{key}"] from source rank {min_rank}.')

        return state_dict

    def keys_to_resume_during_load(self) -> Tuple[Set, Union[str, None]]:
        latest_checkpoint_file = self._read_latest_checkpoint_file()

        resume_keys = []

        if latest_checkpoint_file is not None:
            # 1. Resume training from latest_checkpoint.txt under the same name.
            checkpoint_path = os.path.join(self.load_dirname, latest_checkpoint_file)
            resume_keys.extend(self.KEYS_TO_SAVE)
        else:
            if self.load_path:
                # 2. Load the module weights specified by config_checkpoint.path.
                checkpoint_path = self.load_path
                if self.load_s3_backend_key:
                    checkpoint_path = f"s3://ckpt/{checkpoint_path}"
                if self.load_training_state:
                    resume_keys.extend(self.KEYS_TO_SAVE)
                else:
                    resume_keys.append("model")
                    if self.only_load_scheduler_state:
                        resume_keys.append("scheduler")
            else:
                checkpoint_path = None
        if len(self.keys_not_to_resume) > 0:
            for key in self.keys_not_to_resume:
                assert key in self.KEYS_TO_SAVE, f"Invalid key to resume: {key} not in {self.KEYS_TO_SAVE}"
            resume_keys = [key for key in resume_keys if key not in self.keys_not_to_resume]
        return set(resume_keys), checkpoint_path

    @misc.timer("checkpoint loading")
    def load(
        self,
        model: ImaginaireModel,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        grad_scaler: torch.amp.GradScaler | None = None,
    ) -> int:
        """Load network weights and optimizer states from a checkpoint in a single process.

        The priority of the checkpoint loading logic is:
        1. Attempt to resume training if possible by looking for latest_checkpoint.txt under the same name.
        2. If no latest checkpoint were found, it loads the model weights specified by config_checkpoint.path.
           - This is typically used for inference mode.
           - If config_checkpoint.load_optimizer_state is True, then also load the optimizer and scheduler states.
        3. If none of the above, randomly initialize the model parameters and train from scratch.

        Args:
            model (ImaginaireModel): The PyTorch model.
            optimizer (torch.optim.Optimizer | None): The model optimizer (default: None).
            scheduler (torch.optim.lr_scheduler.LRScheduler | None): The optimization scheduler (default: None).
            grad_scaler (torch.amp.GradScaler | None): The gradient scaler (for mixed precision training).

        Returns:
            iteration (int): the iteration number to start/resume from.
        """
        self.callbacks.on_load_checkpoint_start(model)

        resume_keys, checkpoint_path = self.keys_to_resume_during_load()

        iteration = 0

        # Load checkpoint.
        if checkpoint_path is not None:
            self._check_checkpoint_exists(checkpoint_path)
            state_dict = self.load_broadcast_state_dict(checkpoint_path, model, set(resume_keys))

            if "trainer" in state_dict:
                trainer_state = state_dict["trainer"]
                log.critical(state_dict.keys(), rank0_only=False)
                log.critical(trainer_state, rank0_only=False)
                log.info("- Loading the gradient scaler...")
                grad_scaler.load_state_dict(trainer_state["grad_scaler"])
                self.callbacks.on_load_checkpoint(model, state_dict=trainer_state)
                iteration = trainer_state["iteration"]
            if "optim" in state_dict:
                assert optimizer
                optimizer_state = state_dict["optim"]
                log.info("- Loading the optimizer...")
                optimizer.load_state_dict(optimizer_state)
            if "scheduler" in state_dict:
                assert scheduler
                scheduler_state = state_dict["scheduler"]
                log.info("- Loading the scheduler...")
                scheduler.load_state_dict(scheduler_state)
                scheduler.last_epoch = iteration
            if "model" in state_dict:
                model_state = state_dict["model"]
                log.info("- Loading the model...")
                # Filter out keys_to_skip_loading before loading
                if len(self.keys_to_skip_loading) > 0:
                    filtered_state = {k: v for k, v in model_state.items() if k not in self.keys_to_skip_loading}
                    skipped_keys = [k for k in model_state.keys() if k in self.keys_to_skip_loading]
                    if skipped_keys:
                        log.info(f"\t Skipping {len(skipped_keys)} keys: {skipped_keys}")
                    model_state = filtered_state
                # model.load_state_dict(model_state)
                if self.strict_resume:
                    log.info("\t Strict resume mode is on.")
                else:
                    log.info("\t Strict resume mode is off.")
                model_load_info = model.load_state_dict(model_state, strict=self.strict_resume)
                log.info(f"\t {model_load_info}")
                if not model_load_info.missing_keys and not model_load_info.unexpected_keys:
                    log.info("\t Checkpoint weights loaded successfully (all keys matched).")
                else:
                    log.warning("\t Checkpoint weights loaded; review missing_keys/unexpected_keys above.")
            self.print(f"Loaded checkpoint from {checkpoint_path} in iteration {iteration}")
        else:
            log.info("Training from scratch.")
        torch.cuda.empty_cache()

        self.callbacks.on_load_checkpoint_end(model, iteration=iteration, checkpoint_path=checkpoint_path)

        return iteration

    def _write_trained_data_record(self, checkpoint_file: str, trained_data_record: dict[str, int]) -> None:
        """Write json file to save number of seen samples and number of iterations.

        Args:
            checkpoint_file (str): iteration number for the saved checkpoint
            trained_data_record (dict[str, int]): example {"image": 0, "video": 0, "iteration": 0}.
        """
        # filename: iter_xxxxxxxxx_trained_data_record.json
        checkpoint_path = os.path.join(
            self.save_dirname, f"{checkpoint_file.replace('.pt', '')}_trained_data_record.json"
        )
        easy_io.dump(
            trained_data_record,
            checkpoint_path,
            backend_key=self.save_s3_backend_key,
        )
