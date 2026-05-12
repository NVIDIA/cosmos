#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#

# Generate uv lock files for projects.

set -euo pipefail

for file in "$@"; do
  project_dir="$(dirname "$file")"
  if ! uv lock -q --check --project "$project_dir" &>/dev/null; then
    echo "Updating lock file for '$project_dir'" >&2
    uv lock -q --project "$project_dir"
  fi
done
