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

import torch
import transformer_engine as te
import transformer_engine_torch as tex

from cosmos.utils import distributed, log
from cosmos.utils.misc import get_local_tensor_if_DTensor


class FusedAdam(torch.optim.Optimizer):
    """Implements Adam algorithm.


    This version of fused Adam implements 2 fusions.

      * Fusion of the Adam update's elementwise operations
      * A multi-tensor apply launch that batches the elementwise updates applied to all the model's parameters
        into one or a few kernel launches.

    :class:`FusedAdam` may be used as a drop-in replacement for ``torch.optim.AdamW``,
    or ``torch.optim.Adam`` with ``adam_w_mode=False``::

        opt = FusedAdam(model.parameters(), lr = ....)
        ...
        opt.step()


    .. warning::
        A previous version of :class:`FusedAdam` allowed a number of additional arguments to ``step``.
        These additional arguments are now deprecated and unnecessary.

    Adam was been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float, optional): learning rate. (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square. (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability. (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False) NOT SUPPORTED in FusedAdam!
        adam_w_mode (boolean, optional): Apply L2 regularization or weight decay
            True for decoupled weight decay(also known as AdamW) (default: True)
        capturable (bool, optional): whether to use the version of the optimizer
            that can be used with CUDA Graphs. (default: False)
        master_weights (bool, optional): whether to maintain FP32 master weights
           in the optimizer with FP16 mixed precision training, currently can
           only be used with capturable set to True. (default: False)

    .. _Adam - A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        bias_correction=True,
        betas=(0.9, 0.999),
        eps=1e-8,
        adam_w_mode=True,
        weight_decay=0.0,
        amsgrad=False,
        capturable=False,
        master_weights=False,
    ):
        if amsgrad:
            raise RuntimeError("FusedAdam does not support the AMSGrad variant.")
        if master_weights and not capturable:
            raise RuntimeError("Master weights is currently only supported with the capturable version.")

        # If the optimizer is capturable then LR should be a tensor (on GPU)
        log.info(f"FusedAdam master_weights: {master_weights} capturable: {capturable}")

        lr = torch.tensor(lr, dtype=torch.float32) if capturable else lr
        defaults = dict(lr=lr, bias_correction=bias_correction, betas=betas, eps=eps, weight_decay=weight_decay)
        super(FusedAdam, self).__init__(params, defaults)
        self.adam_w_mode = 1 if adam_w_mode else 0

        self.capturable = capturable
        self.master_weights = master_weights

        self.param_groups_master = None

        if capturable:
            for idx, group in enumerate(self.param_groups):
                if len(group["params"]) == 0:
                    continue
                device = group["params"][0].device
                for item in ["lr"]:
                    if isinstance(group[item], float):
                        group[item] = torch.tensor(group[item], dtype=torch.float32)
                    self.param_groups[idx][item] = group[item].to(device=device)

            self._step_supports_amp_scaling = True

        self._dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device="cuda")
        self.multi_tensor_adam = tex.multi_tensor_adam
        self.multi_tensor_adam_capturable = tex.multi_tensor_adam_capturable
        self.multi_tensor_adam_capturable_master = tex.multi_tensor_adam_capturable_master

    def step(self, closure=None, grads=None, output_params=None, scale=None, grad_norms=None, grad_scaler=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.

        The remaining arguments are deprecated, and are only retained (for the moment) for error-checking purposes.
        """
        if any(p is not None for p in [grads, output_params, scale, grad_norms]):
            raise RuntimeError(
                "FusedAdam has been updated. "
                "Simply initialize it identically to torch.optim.Adam, and call step() with no arguments."
            )
        loss = None
        if closure is not None:
            loss = closure()

        if self.param_groups_master is None:
            # Create full precision master weights
            self.param_groups_master = []
            for i, pg in enumerate(self.param_groups):
                param_list = pg["params"]
                self.param_groups_master.append(
                    {
                        # Change related to master weights
                        "params": [p.clone().detach().float() if self.master_weights else None for p in param_list],
                    }
                )

        for group, group_master in zip(self.param_groups, self.param_groups_master):
            if len(group["params"]) == 0:
                continue
            device = group["params"][0].device
            bias_correction = 1 if "bias_correction" in group and group["bias_correction"] else 0
            beta1, beta2 = group["betas"]

            # assume same step across group now to simplify things
            # per parameter step can be easily support by making it tensor, or pass list into kernel
            if "step" in group:
                if self.capturable:
                    group["step"] = (
                        group["step"].to(device=device)
                        if isinstance(group["step"], torch.Tensor)
                        else torch.tensor(group["step"], dtype=torch.int32, device=device)
                    )
                    group["step"] += (self._dummy_overflow_buf != 1).to(torch.int)
                else:
                    group["step"] += 1
            else:
                group["step"] = 1 if not self.capturable else torch.tensor([1], dtype=torch.int, device=device)

            if self.capturable:
                group["lr"] = (
                    group["lr"].to(device=device)
                    if isinstance(group["lr"], torch.Tensor)
                    else torch.tensor(group["lr"], dtype=torch.float32, device=device)
                )

            # create lists for multi-tensor apply
            g_16, p_16, m_16, v_16 = [], [], [], []
            g_bf, p_bf, m_bf, v_bf = [], [], [], []
            g_32, p_32, m_32, v_32 = [], [], [], []
            p_16_master = []
            p_32_master = []
            bf16_master = []

            for p, p_master in zip(group["params"], group_master["params"]):
                if p.grad is None:
                    continue
                if p.grad.data.is_sparse:
                    raise RuntimeError(
                        "FusedAdam does not support sparse gradients, please consider SparseAdam instead"
                    )

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p).float()
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p).float()

                if p.dtype == torch.float16:
                    if self.master_weights:
                        p_16_master.append(get_local_tensor_if_DTensor(p_master))
                    g_16.append(get_local_tensor_if_DTensor(p.grad))
                    p_16.append(get_local_tensor_if_DTensor(p))
                    m_16.append(get_local_tensor_if_DTensor(state["exp_avg"]))
                    v_16.append(get_local_tensor_if_DTensor(state["exp_avg_sq"]))
                elif p.dtype == torch.bfloat16:
                    if self.master_weights:
                        bf16_master.append(get_local_tensor_if_DTensor(p_master))
                    g_bf.append(get_local_tensor_if_DTensor(p.grad))
                    p_bf.append(get_local_tensor_if_DTensor(p))
                    m_bf.append(get_local_tensor_if_DTensor(state["exp_avg"]))
                    v_bf.append(get_local_tensor_if_DTensor(state["exp_avg_sq"]))
                elif p.dtype == torch.float32:
                    if self.master_weights:
                        p_32_master.append(get_local_tensor_if_DTensor(p_master))
                    g_32.append(get_local_tensor_if_DTensor(p.grad))
                    p_32.append(get_local_tensor_if_DTensor(p))
                    m_32.append(get_local_tensor_if_DTensor(state["exp_avg"]))
                    v_32.append(get_local_tensor_if_DTensor(state["exp_avg_sq"]))
                else:
                    raise RuntimeError("FusedAdam only support fp16, bf16 and fp32.")

            # If the optimizer is capturable, then if there's a grad scaler it works
            # on the GPU + a different multi_tensor_applier should be called
            if self.capturable:
                # overflow check of gradients
                found_inf = (
                    grad_scaler._check_inf_per_device(self)[device]
                    if grad_scaler is not None
                    else torch.zeros((1,), device=device)
                )
                self._dummy_overflow_buf.copy_(found_inf)

                # get unscale scale factor
                scale, inv_scale = None, None
                if grad_scaler:
                    scale = grad_scaler._get_scale_async()
                    inv_scale = scale.double().reciprocal().float()
                else:
                    scale = torch.ones((1,), device=device, dtype=torch.float32)
                    inv_scale = torch.ones((1,), device=device, dtype=torch.float32)

                if len(g_16) > 0:
                    te.pytorch.optimizers.multi_tensor_applier(
                        (
                            self.multi_tensor_adam_capturable_master
                            if self.master_weights
                            else self.multi_tensor_adam_capturable
                        ),
                        self._dummy_overflow_buf,
                        [g_16, p_16, m_16, v_16, p_16_master] if self.master_weights else [g_16, p_16, m_16, v_16],
                        group["lr"],
                        beta1,
                        beta2,
                        group["eps"],
                        group["step"],
                        self.adam_w_mode,
                        bias_correction,
                        group["weight_decay"],
                        inv_scale,
                    )

                if len(g_bf) > 0:
                    te.pytorch.optimizers.multi_tensor_applier(
                        (
                            self.multi_tensor_adam_capturable_master
                            if self.master_weights
                            else self.multi_tensor_adam_capturable
                        ),
                        self._dummy_overflow_buf,
                        [g_bf, p_bf, m_bf, v_bf, bf16_master] if self.master_weights else [g_bf, p_bf, m_bf, v_bf],
                        group["lr"],
                        beta1,
                        beta2,
                        group["eps"],
                        group["step"],
                        self.adam_w_mode,
                        bias_correction,
                        group["weight_decay"],
                        inv_scale,
                    )

                if len(g_32) > 0:
                    te.pytorch.optimizers.multi_tensor_applier(
                        (
                            self.multi_tensor_adam_capturable_master
                            if self.master_weights
                            else self.multi_tensor_adam_capturable
                        ),
                        self._dummy_overflow_buf,
                        [g_32, p_32, m_32, v_32, p_32_master] if self.master_weights else [g_32, p_32, m_32, v_32],
                        group["lr"],
                        beta1,
                        beta2,
                        group["eps"],
                        group["step"],
                        self.adam_w_mode,
                        bias_correction,
                        group["weight_decay"],
                        inv_scale,
                    )
            else:
                if len(g_16) > 0:
                    te.pytorch.optimizers.multi_tensor_applier(
                        self.multi_tensor_adam,
                        self._dummy_overflow_buf,
                        [g_16, p_16, m_16, v_16],
                        group["lr"],
                        beta1,
                        beta2,
                        group["eps"],
                        group["step"],
                        self.adam_w_mode,
                        bias_correction,
                        group["weight_decay"],
                    )

                if len(g_bf) > 0:
                    te.pytorch.optimizers.multi_tensor_applier(
                        self.multi_tensor_adam,
                        self._dummy_overflow_buf,
                        [g_bf, p_bf, m_bf, v_bf],
                        group["lr"],
                        beta1,
                        beta2,
                        group["eps"],
                        group["step"],
                        self.adam_w_mode,
                        bias_correction,
                        group["weight_decay"],
                    )

                if len(g_32) > 0:
                    te.pytorch.optimizers.multi_tensor_applier(
                        self.multi_tensor_adam,
                        self._dummy_overflow_buf,
                        [g_32, p_32, m_32, v_32],
                        group["lr"],
                        beta1,
                        beta2,
                        group["eps"],
                        group["step"],
                        self.adam_w_mode,
                        bias_correction,
                        group["weight_decay"],
                    )

        return loss

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        for group in self.param_groups:
            if self.capturable:
                group["lr"] = (
                    group["lr"].cuda()
                    if isinstance(group["lr"], torch.Tensor)
                    else torch.tensor(group["lr"], dtype=torch.float32).cuda()
                )

            if "step" in group:
                if self.capturable:
                    if distributed.get_rank() == 0:
                        step = (
                            group["step"].cuda()
                            if isinstance(group["step"], torch.Tensor)
                            else torch.tensor([group["step"]], dtype=torch.int32).cuda()
                        )
                    else:
                        step = torch.zeros(1, dtype=torch.int32).cuda()
                    # make it compatible with FSDP optimizer
                    distributed.broadcast(step, 0)
                    group["step"] = step
                elif isinstance(group["step"], torch.Tensor):
                    group["step"] = group["step"].item()
            for p in group["params"]:
                state = self.state[p]
                if "exp_avg" in state:
                    state["exp_avg"] = state["exp_avg"].float()
                    state["exp_avg_sq"] = state["exp_avg_sq"].float()
