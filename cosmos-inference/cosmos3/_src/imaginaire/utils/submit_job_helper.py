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
import os.path as osp

import git
from loguru import logger as logging


def is_git(path):
    try:
        _ = git.Repo(path, search_parent_directories=True).git_dir
        return True
    except git.exc.InvalidGitRepositoryError:
        return False


def git_rootdir(path=""):
    if is_git(os.getcwd()):
        git_repo = git.Repo(os.getcwd(), search_parent_directories=True)
        root = git_repo.git.rev_parse("--show-toplevel")
        return osp.join(root, path)
    logging.info("not a git repo")
    return osp.join(os.getcwd(), path)
