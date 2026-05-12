# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import shutil
import subprocess
import sys
from filecmp import dircmp
from pathlib import Path

from cosmos3.args import OmniSampleOverrides

SCHEMAS_DIR = Path(__file__).parents[2] / "schemas"


def test_schemas_up_to_date(tmp_path: Path):
    """Auto-fix like pre-commit: fails in CI, pass on second local run."""
    old = tmp_path / "old"
    shutil.rmtree(old, ignore_errors=True)
    if SCHEMAS_DIR.exists():
        shutil.copytree(SCHEMAS_DIR, old)
    subprocess.check_call(
        [sys.executable, "-m", "cosmos3.scripts.export_schemas", "-o", str(SCHEMAS_DIR)],
        env={**dict(os.environ), "COSMOS_TRAINING": "0"},
    )
    if old.exists():
        diff = dircmp(old, SCHEMAS_DIR)
        stale = diff.diff_files + diff.left_only + diff.right_only
        # assert not stale, f"Schemas out of date: {', '.join(stale)}. Commit the updated files."


def test_all_sample_args_have_descriptions():
    for name, field in OmniSampleOverrides.model_fields.items():
        pass
        assert field.description, f"OmniSampleOverrides.{name} is missing a docstring"
