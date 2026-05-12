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

from cosmos3._src.imaginaire.utils.env_parsers.env_parser import EnvParser
from cosmos3._src.imaginaire.utils.validator import Bool, String


class CustomizationEnvParser(EnvParser):
    FLEET_FUNCTION = Bool(default=False)
    CUSTOMIZATION_TYPE = String(default="")
    DEBUG_SKIP_CUSTOMIZATION_DOWNLOAD = Bool(default=False)
    FT_AWS_ACCESS_KEY_ID = String(default="")
    FT_AWS_SECRET_ACCESS_KEY = String(default="")
    FT_AWS_REGION_NAME = String(default="")
    FT_AWS_GATEWAY_URL = String(default="")
    LAMBDA_STAGE = String(default="prod")


CUSTOMIZATION_ENVS = CustomizationEnvParser()
