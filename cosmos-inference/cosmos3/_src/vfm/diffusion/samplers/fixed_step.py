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

"""Fixed-step sampler for DMD2-distilled student models.

Uses an explicit, fixed sigma schedule (t_list) baked in at construction time.
Each step predicts x0 via a single velocity forward pass, then either:
  - ODE: Euler step  x_next = x_t + (sigma_next - sigma_cur) * v
  - SDE: re-noise x0 to sigma_next with fresh noise

This is incompatible with multi-step solvers (UniPC, EDM) because DMD2 students
are trained as one-shot denoisers at specific discrete sigmas, not as smooth
score functions.

When ``shift`` is passed at call time, the schedule is derived dynamically via
the flow-matching shift formula (same as UniPC):
  sigmas = shift * s / (1 + (shift - 1) * s),  s = linspace(sigma_max, sigma_min, num_steps)
In this case ``num_steps`` is required. Otherwise ``self.t_list`` is used.
"""

import torch

from cosmos3._src.vfm.diffusion.samplers.utils import run_multiseed


class FixedStepSampler:
    def __init__(
        self,
        t_list: list[float],
        sample_type: str = "ode",
        num_train_timesteps: float = 1000.0,
    ) -> None:
        assert len(t_list) >= 1, "t_list must have at least 1 entry"
        assert sample_type in ("ode", "sde"), f"sample_type must be 'ode' or 'sde', got {sample_type}"
        # Auto-append 0.0 if not present (convention: t_list in config excludes final step)
        self.t_list = t_list if t_list[-1] == 0.0 else t_list + [0.0]
        assert len(self.t_list) >= 2, "t_list must have at least 2 entries after appending 0.0"
        self.sample_type = sample_type
        self.num_train_timesteps = num_train_timesteps

    def _build_t_list(self, num_steps: int, shift: float, device: torch.device) -> list[float]:
        """Compute a shifted sigma schedule with ``num_steps`` integration steps."""
        sigma_max = 1.0
        sigma_min = 1.0 / self.num_train_timesteps
        sigmas = torch.linspace(sigma_max, sigma_min, num_steps, device=device)
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        return sigmas.tolist() + [0.0]

    def __call__(
        self,
        velocity_fn,
        noise: torch.Tensor | list[torch.Tensor],
        num_steps: int | None = None,
        shift: float | None = None,
        seed: int | list[int] | None = None,
    ) -> torch.Tensor | list[torch.Tensor]:
        """Run the fixed-step sampling loop.

        Matches the UniPC sampler call signature so both can be used
        interchangeably in ``generate_samples_from_batch``.

        ``noise`` and ``seed`` must both be single values or both be lists
        (of the same length).  When lists are provided, each element
        corresponds to one independent sample; the return value is then a
        list of denoised tensors.  When single values are provided, a
        single tensor is returned.

        Args:
            velocity_fn: ``velocity_fn(noise=..., timestep=...) -> velocity``.
            noise: Initial noise.  Either a single ``torch.Tensor`` of shape
                ``(D,)`` or a ``list[torch.Tensor]`` where each element has
                shape ``(D,)``.
            seed: RNG seed for SDE mode.  Either a single ``int`` or a
                ``list[int]`` with the same length as ``noise``.
            num_steps: Number of denoising steps.  Required when ``shift`` is
                given; optional otherwise (asserted to equal
                ``len(t_list) - 1`` when provided).
            shift: When set, derive the sigma schedule dynamically using the
                flow-matching shift formula instead of ``self.t_list``.

        Returns:
            Denoised sample(s).  A single ``torch.Tensor`` when ``noise`` is a
            tensor, or a ``list[torch.Tensor]`` when ``noise`` is a list.
        """
        if isinstance(noise, list):
            device = noise[0].device
        else:
            device = noise.device

        if shift is not None:
            assert num_steps is not None, "num_steps is required when shift is provided"
            t_list = self._build_t_list(num_steps, shift, device)
        else:
            if num_steps is not None:
                assert num_steps == len(self.t_list) - 1, (
                    f"num_steps={num_steps} must match the schedule length len(t_list)-1={len(self.t_list) - 1}"
                )
            t_list = self.t_list

        latent = noise

        for step_idx, (sigma_cur, sigma_next) in enumerate(
            zip(t_list[:-1], t_list[1:]),
        ):
            timestep = torch.tensor(sigma_cur * self.num_train_timesteps, device=device)
            v_pred = velocity_fn(latent, timestep.reshape(1, 1))

            def _sde_step(seed: int | None, latent: torch.Tensor, v_pred: torch.Tensor) -> torch.Tensor:
                x0_pred = latent - sigma_cur * v_pred

                if sigma_next > 0:
                    if self.sample_type == "ode":
                        # Euler ODE step
                        latent = latent + (sigma_next - sigma_cur) * v_pred
                    else:
                        if seed is not None:
                            torch.manual_seed(seed + step_idx)
                        eps_fresh = torch.randn_like(x0_pred)
                        latent = (1.0 - sigma_next) * x0_pred + sigma_next * eps_fresh
                else:
                    latent = x0_pred
                return latent

            latent = run_multiseed(
                _sde_step,
                seed=seed,
                latent=latent,
                v_pred=v_pred,
            )

        return latent
