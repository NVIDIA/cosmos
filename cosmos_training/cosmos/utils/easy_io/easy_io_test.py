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

import numpy as np
import pytest

from cosmos.utils.easy_io import easy_io
from cosmos.utils.helper_test import RunIf


def setup_s3():
    easy_io.set_s3_backend(
        backend_args={
            "backend": "s3",
            "path_mapping": None,
            "s3_credential_path": "credentials/pbss_getty.secret",
        }
    )


@pytest.mark.L1("Requires data loading from S3.")
@RunIf(requires_file="credentials/pbss_getty.secret")
def test_s3_backend():
    setup_s3()
    for ith, _ in enumerate(easy_io.list_dir_or_file("s3://checkpoints/")):
        if ith > 5:
            break

    easy_io.copyfile_from_local("pyproject.toml", "s3://checkpoints/pyproject.toml")
    easy_io.remove("s3://checkpoints/pyproject.toml")


@pytest.mark.L1("Requires data uploading to S3.")
@RunIf(requires_file="credentials/pbss_getty.secret")
def test_s3_dump():
    if easy_io.exists("s3://debug/test_00.mp4"):
        easy_io.remove("s3://debug/test_00.mp4")
    np_frames, metadata = easy_io.load("s3://debug/00.mp4")
    easy_io.dump(
        np_frames,
        "s3://debug/test_00.mp4",
        format="mp4",
        fps=metadata.get("fps", 30),
        codec=metadata.get("codec", "h264"),
    )
    if easy_io.exists("s3://debug/dummy_dict.pkl"):
        easy_io.remove("s3://debug/dummy_dict.pkl")

    dummy_dict = {"a": 1, "b": 2}
    easy_io.dump(dummy_dict, "s3://debug/dummy_dict.pkl")
    dummy_np = np.array([1, 2, 3])
    easy_io.dump(dummy_np, "s3://debug/dummy_np.npy")


@pytest.mark.L0
def test_local_backend():
    num_files = len(list(easy_io.list_dir_or_file(".")))
    assert num_files > 0

    if easy_io.exists("dummy_dict.pkl"):
        easy_io.remove("dummy_dict.pkl")

    dummy_dict = {"a": 1, "b": 2}
    easy_io.dump(dummy_dict, "dummy_dict.pkl")
    load_dict = easy_io.load("dummy_dict.pkl")
    for key in dummy_dict:
        assert dummy_dict[key] == load_dict[key]

    if easy_io.exists("dummy_dict.pkl"):
        easy_io.remove("dummy_dict.pkl")
