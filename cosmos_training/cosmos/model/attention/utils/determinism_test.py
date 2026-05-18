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

"""
Unit tests for ``torch_deterministic_mode`` context manager.

Verifies that global PyTorch deterministic state is never leaked, regardless of
how the context is entered, exited, nested, or interrupted.
"""

import unittest

import pytest
import torch

from cosmos.model.attention.utils import torch_deterministic_mode


@pytest.mark.L0
class TestTorchDeterministicMode(unittest.TestCase):
    def setUp(self):
        # Always start from a known clean state
        torch.use_deterministic_algorithms(False)

    def tearDown(self):
        torch.use_deterministic_algorithms(False)

    # ------------------------------------------------------------------
    # Basic on/off restoration
    # ------------------------------------------------------------------

    def test_enables_inside_context(self):
        assert not torch.are_deterministic_algorithms_enabled()
        with torch_deterministic_mode():
            assert torch.are_deterministic_algorithms_enabled()
        assert not torch.are_deterministic_algorithms_enabled()

    def test_restores_off_when_was_off(self):
        torch.use_deterministic_algorithms(False)
        with torch_deterministic_mode():
            pass
        assert not torch.are_deterministic_algorithms_enabled()

    def test_restores_on_when_was_on(self):
        torch.use_deterministic_algorithms(True)
        with torch_deterministic_mode():
            assert torch.are_deterministic_algorithms_enabled()
        assert torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(False)  # cleanup

    # ------------------------------------------------------------------
    # warn_only flag preservation
    # ------------------------------------------------------------------

    def test_restores_warn_only_false(self):
        torch.use_deterministic_algorithms(False, warn_only=False)
        with torch_deterministic_mode():
            pass
        assert not torch.is_deterministic_algorithms_warn_only_enabled()

    def test_restores_warn_only_true(self):
        torch.use_deterministic_algorithms(True, warn_only=True)
        assert torch.is_deterministic_algorithms_warn_only_enabled()
        with torch_deterministic_mode():
            # Inside the context, warn_only should be False (strict mode)
            assert not torch.is_deterministic_algorithms_warn_only_enabled()
        assert torch.is_deterministic_algorithms_warn_only_enabled()
        torch.use_deterministic_algorithms(False)  # cleanup

    def test_restores_off_with_warn_only_false(self):
        torch.use_deterministic_algorithms(False, warn_only=False)
        with torch_deterministic_mode():
            pass
        assert not torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()

    # ------------------------------------------------------------------
    # All four (mode, warn_only) combinations
    # ------------------------------------------------------------------

    def test_all_prior_state_combinations(self):
        combos = [
            (False, False),
            (True, False),
            (True, True),
            # (False, True) is not a valid torch state — warn_only requires mode=True
        ]
        for mode, warn_only in combos:
            torch.use_deterministic_algorithms(mode, warn_only=warn_only)
            with torch_deterministic_mode():
                assert torch.are_deterministic_algorithms_enabled()
                assert not torch.is_deterministic_algorithms_warn_only_enabled()
            assert torch.are_deterministic_algorithms_enabled() == mode
            assert torch.is_deterministic_algorithms_warn_only_enabled() == warn_only

        torch.use_deterministic_algorithms(False)  # cleanup

    # ------------------------------------------------------------------
    # Exception safety
    # ------------------------------------------------------------------

    def test_restores_after_exception(self):
        torch.use_deterministic_algorithms(False)
        with pytest.raises(ValueError):
            with torch_deterministic_mode():
                assert torch.are_deterministic_algorithms_enabled()
                raise ValueError("boom")
        assert not torch.are_deterministic_algorithms_enabled()

    def test_restores_warn_only_after_exception(self):
        torch.use_deterministic_algorithms(True, warn_only=True)
        with pytest.raises(RuntimeError):
            with torch_deterministic_mode():
                assert not torch.is_deterministic_algorithms_warn_only_enabled()
                raise RuntimeError("crash")
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.is_deterministic_algorithms_warn_only_enabled()
        torch.use_deterministic_algorithms(False)  # cleanup

    # ------------------------------------------------------------------
    # Nesting
    # ------------------------------------------------------------------

    def test_nested_contexts_restore_correctly(self):
        torch.use_deterministic_algorithms(False)
        with torch_deterministic_mode():
            assert torch.are_deterministic_algorithms_enabled()
            with torch_deterministic_mode():
                assert torch.are_deterministic_algorithms_enabled()
            # Inner exit should restore to True (outer's state)
            assert torch.are_deterministic_algorithms_enabled()
        # Outer exit should restore to False
        assert not torch.are_deterministic_algorithms_enabled()

    def test_nested_with_different_prior_states(self):
        torch.use_deterministic_algorithms(True, warn_only=True)
        with torch_deterministic_mode():
            assert not torch.is_deterministic_algorithms_warn_only_enabled()
            # Manually change state inside, then nest
            torch.use_deterministic_algorithms(False)
            with torch_deterministic_mode():
                assert torch.are_deterministic_algorithms_enabled()
            # Inner restores to False (its captured state)
            assert not torch.are_deterministic_algorithms_enabled()
        # Outer restores to (True, warn_only=True)
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.is_deterministic_algorithms_warn_only_enabled()
        torch.use_deterministic_algorithms(False)  # cleanup

    def test_nested_with_exception_in_inner(self):
        torch.use_deterministic_algorithms(False)
        with torch_deterministic_mode():
            try:
                with torch_deterministic_mode():
                    raise ValueError("inner boom")
            except ValueError:
                pass
            # Inner restored, outer still active
            assert torch.are_deterministic_algorithms_enabled()
        assert not torch.are_deterministic_algorithms_enabled()

    def test_nested_with_exception_in_outer(self):
        torch.use_deterministic_algorithms(False)
        with pytest.raises(ValueError):
            with torch_deterministic_mode():
                with torch_deterministic_mode():
                    pass
                raise ValueError("outer boom")
        assert not torch.are_deterministic_algorithms_enabled()

    # ------------------------------------------------------------------
    # Repeated use
    # ------------------------------------------------------------------

    def test_repeated_use_no_state_drift(self):
        for _ in range(100):
            torch.use_deterministic_algorithms(False)
            assert not torch.are_deterministic_algorithms_enabled()
            with torch_deterministic_mode():
                assert torch.are_deterministic_algorithms_enabled()
            assert not torch.are_deterministic_algorithms_enabled()

    def test_repeated_use_alternating_prior_state(self):
        for i in range(50):
            prior_mode = i % 2 == 0
            torch.use_deterministic_algorithms(prior_mode)
            with torch_deterministic_mode():
                assert torch.are_deterministic_algorithms_enabled()
            assert torch.are_deterministic_algorithms_enabled() == prior_mode
        torch.use_deterministic_algorithms(False)  # cleanup

    # ------------------------------------------------------------------
    # State not modified between enter and body
    # ------------------------------------------------------------------

    def test_no_intermediate_state_between_enter_and_body(self):
        """The context should set deterministic=True atomically on __enter__."""
        torch.use_deterministic_algorithms(False)
        cm = torch_deterministic_mode()
        # Before entering, state is still off
        assert not torch.are_deterministic_algorithms_enabled()
        cm.__enter__()
        # Immediately after enter, state is on
        assert torch.are_deterministic_algorithms_enabled()
        cm.__exit__(None, None, None)
        assert not torch.are_deterministic_algorithms_enabled()


if __name__ == "__main__":
    unittest.main()
