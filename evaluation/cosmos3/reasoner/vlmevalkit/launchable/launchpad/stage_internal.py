#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare internal VQA on a PDX-reachable workstation; transfer only native data."""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import subprocess

from launchpad import source_digest


NATIVE_PREPARE = r'''
import json, pathlib, sys
sys.path.insert(0, '/opt/vlmevalkit')
sys.path.insert(0, '/opt/vlmevalkit/cosmos_eval')
from vlmeval.dataset.metropolis_vqa import MetropolisVQA
import launchable
import run_all
from run import build_dataset_from_config
root = pathlib.Path('/data/internal/LMUData/datasets/VANTAGE_VQA')
converter = MetropolisVQA(dataset='VANTAGE_VQA', test_mode=True)
converter._generate_tsv_from_annotations(None, '', str(root), 'VANTAGE_VQA')
launchable.prepare(['VANTAGE_VQA'])
cfg = run_all.build_config(run_all.load_model_layer('cosmos'), 'metropolis', 'VANTAGE_VQA', render=False)
dataset = build_dataset_from_config(cfg['data'], 'VANTAGE_VQA')
for _, row in dataset.data.iterrows():
    videos = [part['value'] for part in dataset.build_prompt(row, video_llm=True) if part['type'] == 'video']
    if len(videos) != 1 or not pathlib.Path(videos[0]).is_file():
        raise RuntimeError('Native dataset references an unavailable video')
print(json.dumps({'native_prepared_questions':len(dataset.data),'native_video_count':len(dataset.videos),'all_referenced_videos_present':True}))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local-state-dir', required=True, type=Path, help='Private workstation data/cache/log directory outside source')
    parser.add_argument('--source-root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--image', default='cosmos3-vantage-eval:local')
    parser.add_argument('--host', required=True)
    parser.add_argument('--user')
    parser.add_argument('--control-path')
    parser.add_argument('--remote-state-dir', required=True, help='Absolute isolated LaunchPad state directory')
    args = parser.parse_args()
    root, state = args.source_root.resolve(), args.local_state_dir.resolve()
    source_digest(root)  # Refuse secret-bearing source before any container mount.
    if state == root or root in state.parents:
        raise ValueError('Local state must be outside source')
    remote_state = PurePosixPath(args.remote_state_dir)
    if not remote_state.is_absolute() or str(remote_state) == '/' or '..' in remote_state.parts:
        raise ValueError('Use an absolute isolated remote state directory')
    # Read canonical dispatch without importing the heavyweight evaluation stack
    # on the workstation or maintaining a second benchmark-source table.
    tree = ast.parse((root / 'vlmeval/dataset/metropolis_vqa.py').read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'MetropolisVQA')
    table = next(node.value for node in cls.body if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == '_S3_PATHS' for target in node.targets))
    bucket, prefix, profile = ast.literal_eval(table)['VANTAGE_VQA']
    if profile != 'team-cosmos':
        raise ValueError('Only the explicitly authorized team-cosmos profile is supported')
    state.mkdir(mode=0o700, parents=True, exist_ok=True)
    state.chmod(0o700)
    data = state / 'data'
    dataset = data / 'internal/LMUData/datasets/VANTAGE_VQA'
    dataset.mkdir(mode=0o700, parents=True, exist_ok=True)
    cache = state / 'cache'
    cache.mkdir(mode=0o700, exist_ok=True)
    subprocess.run(['aws', '--profile', profile, '--endpoint-url', 'https://pdx.s8k.io', 's3', 'sync',
                    f's3://{bucket}/{prefix}/', str(dataset), '--only-show-errors'], check=True)
    log_path = state / 'native-staged-prepare.log'
    fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w') as log:
        subprocess.run(['docker', 'run', '--rm', '--platform', 'linux/amd64', '--network', 'none',
                        '-v', f'{root}:/opt/vlmevalkit:ro', '-v', f'{data}:/data', '-v', f'{cache}:/cache',
                        '-e', 'LMUData=/data/internal/LMUData', '-e', 'HOME=/tmp/cosmos-home',
                        '-e', 'AWS_CONFIG_FILE=/dev/null', '-e', 'AWS_SHARED_CREDENTIALS_FILE=/dev/null',
                        '-e', 'AWS_EC2_METADATA_DISABLED=true', '--entrypoint', 'python', args.image,
                        '-c', NATIVE_PREPARE], stdout=log, stderr=subprocess.STDOUT, check=True)
    if any(path.is_symlink() for path in dataset.rglob('*')):
        raise ValueError('Refusing symlinks in the private dataset transfer')
    ssh = ['ssh']
    if args.user:
        ssh += ['-l', args.user]
    if args.control_path:
        ssh += ['-o', 'ControlMaster=auto', '-o', 'ControlPersist=10m', '-o', 'ControlPath=' + args.control_path]
    target = str(remote_state / 'data/internal/LMUData/datasets/VANTAGE_VQA')
    subprocess.run(ssh + [args.host, 'umask 077; mkdir -p -- ' + shlex.quote(target) + ' && chmod 700 -- ' + shlex.quote(target)], check=True)
    # No --delete, no credentials/cache/home directory and no result paths.
    subprocess.run(['rsync', '-rlt', '--partial', '--chmod=Du=rwx,Dgo=,Fu=rw,Fgo=', '-e', shlex.join(ssh),
                    str(dataset) + '/', args.host + ':' + shlex.quote(target) + '/'], check=True)
    print(json.dumps({'native_preparation_passed': True, 'private_log': str(log_path),
                      'transferred_only': 'data/internal/LMUData/datasets/VANTAGE_VQA',
                      'remote_state_dir': str(remote_state), 'credentials_transferred': False,
                      's3_writes': False, 'resume': 'Rerun the same command; aws sync and rsync preserve completed files; no --delete'}))


if __name__ == '__main__':
    main()
