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

# Generate uv lock files for scripts.

set -euo pipefail

for file in "$@"; do
  if head -n1 "$file" | grep -q '^#!/usr/bin/env -S uv run --script'; then
    if ! uv lock -q --check --script "$file" &>/dev/null; then
      echo "Updating lock file for '$file'" >&2
      uv lock -q --script "$file"
    fi
  fi
done
