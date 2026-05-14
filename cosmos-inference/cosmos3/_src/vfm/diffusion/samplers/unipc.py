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

from typing import Callable, Optional

import attrs
import torch

from cosmos3._src.imaginaire.config import make_freezable
from cosmos3._src.imaginaire.utils.progress_bar import progress_bar
from cosmos3._src.vfm.diffusion.samplers.fm_solvers_unipc import FlowUniPCMultistepScheduler
from cosmos3._src.vfm.diffusion.samplers.utils import run_multiseed


@make_freezable
@attrs.define(slots=False)
class UniPCSamplerConfig:
    num_train_timesteps: int = 1000
    shift: float = 1.0
    use_dynamic_shifting: bool = False


class UniPCSampler(torch.nn.Module):
    def __init__(self, cfg: Optional[UniPCSamplerConfig] = None, tensor_kwargs: Optional[dict] = None):
        super().__init__()
        if cfg is None:
            cfg = UniPCSamplerConfig()
        self.cfg = cfg
        self.tensor_kwargs = tensor_kwargs

    @torch.no_grad()
    def forward(
        self,
        velocity_fn: Callable,
        noise: torch.Tensor | list[torch.Tensor],
        num_steps: int = 35,
        shift: float | None = None,
        seed: int | list[int] | None = None,
    ) -> torch.Tensor | list[torch.Tensor]:
        """Run the UniPC multi-step sampling loop.

        ``noise`` and ``seed`` must both be single values or both be lists
        (of the same length).  When lists are provided, each element
        corresponds to one independent sample with its own RNG generator
        and scheduler; the return value is then a list of denoised tensors.
        When single values are provided, a single tensor is returned.

        Args:
            velocity_fn: ``velocity_fn(noise=..., timestep=...) -> velocity``.
            noise: Initial noise.  Either a single ``torch.Tensor`` of shape
                ``(C, T, H, W)`` or a ``list[torch.Tensor]`` where each
                element has shape ``(C, T, H, W)``.
            seed: RNG seed.  Either a single ``int`` or a ``list[int]`` with
                the same length as ``noise``.
            num_steps: Number of denoising steps.
            shift: Flow-matching shift factor.  Defaults to ``self.cfg.shift``.

        Returns:
            Denoised sample(s).  A single ``torch.Tensor`` when ``noise`` is a
            tensor, or a ``list[torch.Tensor]`` when ``noise`` is a list.
        """
        if shift is None:
            shift = self.cfg.shift
        assert isinstance(shift, float), "Shift must be a float"

        def _init_sample_scheduler(seed: int | None) -> tuple[torch.Generator, FlowUniPCMultistepScheduler]:
            seed_g = torch.Generator(device=self.tensor_kwargs["device"])
            if seed is not None:
                seed_g.manual_seed(seed)
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.cfg.num_train_timesteps,
                shift=self.cfg.shift,
                use_dynamic_shifting=self.cfg.use_dynamic_shifting,
            )
            sample_scheduler.set_timesteps(num_steps, device=self.tensor_kwargs["device"], shift=shift)
            return seed_g, sample_scheduler

        seed_g, sample_scheduler = run_multiseed(_init_sample_scheduler, seed=seed)

        timesteps = sample_scheduler[0].timesteps if isinstance(sample_scheduler, list) else sample_scheduler.timesteps
        latent = noise

        for timestep in progress_bar(timesteps, desc="Sampling", total=len(timesteps)):
            velocity_pred = velocity_fn(latent, timestep.reshape(1, 1))

            def _scheduler_step(
                seed_g: torch.Generator,
                sample_scheduler: FlowUniPCMultistepScheduler,
                velocity_pred: torch.Tensor,
                latent: torch.Tensor,
            ) -> torch.Tensor:
                # multistep_uni_p_bh_update and multistep_uni_c_bh_update both use einsum patterns
                # like "k,bkc...->bc...", which expect the tensor to have at least shape
                # [B, C, ...] — where b is the batch dimension. Therefore, we need to unsqueeze
                # the latent tensor to [B, C, ...] before passing it to the scheduler.
                return sample_scheduler.step(
                    model_output=velocity_pred,
                    timestep=timestep,
                    sample=latent.unsqueeze(0),
                    return_dict=False,
                    generator=seed_g,
                )[0].squeeze(0)

            latent = run_multiseed(
                _scheduler_step,
                seed_g=seed_g,
                sample_scheduler=sample_scheduler,
                velocity_pred=velocity_pred,
                latent=latent,
            )

        return latent
