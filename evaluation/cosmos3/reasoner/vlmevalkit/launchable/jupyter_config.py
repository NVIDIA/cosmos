# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Start Jupyter without placing the authentication token in process arguments."""
import os

from jupyter_server.serverapp import ServerApp
from traitlets.config import Config

config = Config()
config.ServerApp.ip = "0.0.0.0"
config.ServerApp.port = 8888
config.ServerApp.port_retries = 0
config.ServerApp.open_browser = False
config.ServerApp.allow_root = True
# Brev's reverse proxy preserves a non-local Host header. Token auth remains on.
config.ServerApp.allow_remote_access = True
config.ServerApp.root_dir = "/"
config.ServerApp.default_url = "/lab/tree/opt/vlmevalkit/launchable/Evaluate_Cosmos3.ipynb"
config.ServerApp.jpserver_extensions = {"jupyterlab": True}
config.IdentityProvider.token = os.environ["JUPYTER_TOKEN"]
ServerApp.launch_instance(config=config)
