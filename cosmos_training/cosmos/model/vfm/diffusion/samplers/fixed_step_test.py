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

import pytest
import torch

from cosmos.model.vfm.diffusion.samplers.fixed_step import FixedStepSampler


@pytest.mark.L0
def test_auto_appends_zero():
    sampler = FixedStepSampler(t_list=[0.999, 0.75, 0.5, 0.25])
    assert sampler.t_list[-1] == 0.0
    assert sampler.t_list == [0.999, 0.75, 0.5, 0.25, 0.0]


@pytest.mark.L0
def test_no_double_append_zero():
    t_list = [0.999, 0.75, 0.5, 0.25, 0.0]
    sampler = FixedStepSampler(t_list=t_list)
    assert sampler.t_list == t_list
    assert sampler.t_list.count(0.0) == 1


@pytest.mark.L0
def test_ode_single_step():
    # 1-step ODE schedule [1.0, 0.0], v=0 => x0 = noise - 1*0 = noise, last step => x = x0_pred = noise
    sampler = FixedStepSampler(t_list=[1.0, 0.0], sample_type="ode")
    noise = torch.randn(1, 8)

    def zero_vel(x, _t):
        return torch.zeros_like(x)

    result = sampler(zero_vel, noise)
    assert torch.allclose(result, noise)


@pytest.mark.L0
def test_ode_multi_step_correctness():
    # 2-step ODE: t_list=[1.0, 0.5, 0.0]
    # Step 0: sigma_cur=1.0, sigma_next=0.5
    #   v_pred = v0, x0_pred = noise - 1.0 * v0
    #   Euler: x1 = noise + (0.5 - 1.0) * v0 = noise - 0.5 * v0
    # Step 1: sigma_cur=0.5, sigma_next=0.0
    #   v_pred = v1, x0_pred = x1 - 0.5 * v1
    #   last step (sigma_next==0): x = x0_pred = x1 - 0.5 * v1
    noise = torch.tensor([[1.0, 2.0, 3.0]])
    v0 = torch.tensor([[0.1, 0.2, 0.3]])
    v1 = torch.tensor([[0.4, 0.5, 0.6]])
    call_count = [0]

    def velocity_fn(x, t):
        i = call_count[0]
        call_count[0] += 1
        return v0 if i == 0 else v1

    sampler = FixedStepSampler(t_list=[1.0, 0.5, 0.0], sample_type="ode")
    result = sampler(velocity_fn, noise)

    x1 = noise - 0.5 * v0
    expected = x1 - 0.5 * v1
    assert torch.allclose(result, expected, atol=1e-6)


@pytest.mark.L0
def test_sde_differs_from_ode():
    sampler_ode = FixedStepSampler(t_list=[1.0, 0.5, 0.0], sample_type="ode")
    sampler_sde = FixedStepSampler(t_list=[1.0, 0.5, 0.0], sample_type="sde")
    noise = torch.randn(1, 16)
    # Use non-zero velocity so paths diverge

    def velocity_fn(x, _t):
        return torch.ones_like(x) * 0.1

    result_ode = sampler_ode(velocity_fn, noise.clone())
    result_sde = sampler_sde(velocity_fn, noise.clone(), seed=42)
    assert not torch.allclose(result_ode, result_sde)


@pytest.mark.L0
def test_sde_reproducible_with_seed():
    sampler = FixedStepSampler(t_list=[1.0, 0.5, 0.0], sample_type="sde")
    noise = torch.randn(1, 16)

    def velocity_fn(x, _t):
        return torch.ones_like(x) * 0.1

    result_a = sampler(velocity_fn, noise.clone(), seed=123)
    result_b = sampler(velocity_fn, noise.clone(), seed=123)
    assert torch.allclose(result_a, result_b)


@pytest.mark.L0
def test_output_shape_preserved():
    sampler = FixedStepSampler(t_list=[1.0, 0.5, 0.25, 0.0])
    noise = torch.randn(1, 32)

    def velocity_fn(x, _t):
        return torch.zeros_like(x)

    result = sampler(velocity_fn, noise)
    assert result.shape == noise.shape


@pytest.mark.L0
def test_validates_empty_t_list():
    with pytest.raises(AssertionError):
        FixedStepSampler(t_list=[])


@pytest.mark.L0
def test_validates_single_entry_t_list():
    # t_list=[0.0] has len==1 after construction; 0.0 is already last so no append,
    # but then len(self.t_list)==1 < 2 triggers the second assert
    with pytest.raises(AssertionError):
        FixedStepSampler(t_list=[0.0])


@pytest.mark.L0
def test_validates_sample_type():
    with pytest.raises(AssertionError):
        FixedStepSampler(t_list=[1.0, 0.0], sample_type="invalid")


@pytest.mark.L0
def test_velocity_fn_called_correct_times():
    t_list = [1.0, 0.75, 0.5, 0.25]  # auto-appends 0.0 → 5 entries, 4 steps
    sampler = FixedStepSampler(t_list=t_list)
    noise = torch.randn(1, 8)
    call_count = [0]

    def velocity_fn(x, _t):
        call_count[0] += 1
        return torch.zeros_like(x)

    sampler(velocity_fn, noise)
    expected_calls = len(sampler.t_list) - 1  # 4 steps
    assert call_count[0] == expected_calls


@pytest.mark.L0
def test_num_steps_assertion_passes_when_correct():
    sampler = FixedStepSampler(t_list=[1.0, 0.5, 0.0])  # 2 steps
    noise = torch.randn(1, 4)

    def velocity_fn(x, _t):
        return torch.zeros_like(x)

    result = sampler(velocity_fn, noise, num_steps=2)
    assert result.shape == noise.shape


@pytest.mark.L0
def test_num_steps_assertion_fails_when_wrong():
    sampler = FixedStepSampler(t_list=[1.0, 0.5, 0.0])  # 2 steps
    noise = torch.randn(1, 4)

    def velocity_fn(x, _t):
        return torch.zeros_like(x)

    with pytest.raises(AssertionError):
        sampler(velocity_fn, noise, num_steps=4)


@pytest.mark.L0
def test_shift_requires_num_steps():
    sampler = FixedStepSampler(t_list=[1.0, 0.5, 0.0])
    noise = torch.randn(1, 4)

    def velocity_fn(x, _t):
        return torch.zeros_like(x)

    with pytest.raises(AssertionError, match="num_steps is required"):
        sampler(velocity_fn, noise, shift=2.0)


@pytest.mark.L0
def test_shift_one_gives_uniform_schedule():
    # shift=1.0 is identity: sigmas = 1*s / (1 + 0*s) = s (unchanged)
    num_steps = 4
    sampler = FixedStepSampler(t_list=[0.9, 0.6, 0.3, 0.0], num_train_timesteps=1000.0)
    noise = torch.zeros(1, 4)
    recorded_sigmas = []

    def velocity_fn(x, timestep):
        recorded_sigmas.append(timestep.item() / 1000.0)
        return torch.zeros_like(x)

    sampler(velocity_fn, noise, num_steps=num_steps, shift=1.0)

    sigma_max = 1.0
    sigma_min = 1.0 / 1000.0
    expected = torch.linspace(sigma_max, sigma_min, num_steps).tolist()
    assert len(recorded_sigmas) == num_steps
    for got, exp in zip(recorded_sigmas, expected):
        assert abs(got - exp) < 1e-4, f"got {got}, expected {exp}"


@pytest.mark.L0
def test_shift_greater_than_one_frontloads_schedule():
    # shift > 1 pushes sigmas higher (more steps near sigma_max)
    num_steps = 8
    sampler = FixedStepSampler(t_list=[1.0] + [0.0] * (num_steps - 1) + [0.0])
    noise = torch.zeros(1, 4)
    recorded_sigmas_uniform = []
    recorded_sigmas_shifted = []

    def make_recorder(lst):
        def velocity_fn(x, timestep):
            lst.append(timestep.item() / 1000.0)
            return torch.zeros_like(x)

        return velocity_fn

    sampler(make_recorder(recorded_sigmas_uniform), noise.clone(), num_steps=num_steps, shift=1.0)
    sampler(make_recorder(recorded_sigmas_shifted), noise.clone(), num_steps=num_steps, shift=7.0)

    # With shift>1, all intermediate sigmas should be larger than uniform
    for uniform, shifted in zip(recorded_sigmas_uniform[1:], recorded_sigmas_shifted[1:]):
        assert shifted > uniform, f"expected shifted {shifted} > uniform {uniform}"


@pytest.mark.L0
def test_shift_none_uses_t_list():
    # When shift is None, self.t_list is used (not a dynamic schedule)
    t_list = [0.9, 0.5, 0.1, 0.0]
    sampler = FixedStepSampler(t_list=t_list)
    noise = torch.zeros(1, 4)
    recorded_sigmas = []

    def velocity_fn(x, timestep):
        recorded_sigmas.append(timestep.item() / 1000.0)
        return torch.zeros_like(x)

    sampler(velocity_fn, noise)
    expected = t_list[:-1]  # last step uses sigma=0.0 but loop already excludes it
    assert len(recorded_sigmas) == len(expected)
    for got, exp in zip(recorded_sigmas, expected):
        assert abs(got - exp) < 1e-4


@pytest.mark.L0
def test_timestep_scaling():
    num_train_timesteps = 1000.0
    t_list = [0.8, 0.0]
    sampler = FixedStepSampler(t_list=t_list, num_train_timesteps=num_train_timesteps)
    noise = torch.randn(1, 4)
    recorded_timesteps = []

    def velocity_fn(x, timestep):
        recorded_timesteps.append(timestep.item())
        return torch.zeros_like(x)

    sampler(velocity_fn, noise)
    assert len(recorded_timesteps) == 1
    assert abs(recorded_timesteps[0] - 0.8 * num_train_timesteps) < 1e-5
