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

# Runtime load tracer.
#
# Auto-imported by every Python process started with PYTHONPATH=.
# When LOAD_TRACE_DIR is set, registers an atexit hook that walks
# sys.modules at shutdown and writes the file paths (filtered to those
# under LOAD_TRACE_ROOT) into {LOAD_TRACE_DIR}/{LOAD_TRACE_TAG}_pid{PID}.txt.
#
# Used to inventory which released files are actually touched by each
# end-to-end smoke. Union the per-experiment traces, diff against the full
# .py list, and the residual is dead code (relative to that smoke set).
import atexit
import os
import sys

_DIR = os.environ.get("LOAD_TRACE_DIR", "")
if _DIR:
    _TAG = os.environ.get("LOAD_TRACE_TAG", "default")
    _ROOT = os.path.realpath(os.environ.get("LOAD_TRACE_ROOT", os.getcwd()))

    os.makedirs(_DIR, exist_ok=True)

    def _dump():
        seen = set()
        for mod in list(sys.modules.values()):
            f = getattr(mod, "__file__", None)
            if not f:
                continue
            try:
                rp = os.path.realpath(f)
            except OSError:
                continue
            if rp.startswith(_ROOT):
                seen.add(rp)
        path = os.path.join(_DIR, f"{_TAG}_pid{os.getpid()}.txt")
        try:
            with open(path, "w") as h:
                for p in sorted(seen):
                    h.write(p + "\n")
        except OSError:
            pass

    atexit.register(_dump)
