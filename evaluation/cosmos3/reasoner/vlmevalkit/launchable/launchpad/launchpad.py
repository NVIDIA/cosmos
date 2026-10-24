#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Private single-GPU LaunchPad lifecycle using the existing launchable controller."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def call(args, **kwargs):
    return subprocess.run(args, check=True, **kwargs)


def capture(args, **kwargs):
    return subprocess.check_output(args, text=True, **kwargs).strip()


def source_digest(root):
    digest = hashlib.sha256()
    forbidden = {'.git', '.aws', '.ssh', '.cache', '.venv', 'state', 'credentials'}
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root)
        if any(part in forbidden or part.startswith('.env') for part in relative.parts):
            raise ValueError(f'Source contains runtime/private artifacts: {relative}; use a fresh source-only export, without modifying retained state')
        if path.is_symlink() and root not in path.resolve().parents:
            raise ValueError(f'Source symlink escapes the source-only tree: {relative}')
        if path.suffix in {'.pem', '.key'}:
            raise ValueError(f'Private-key-like source file refused: {relative}')
        if not path.is_file() or any(part in {'__pycache__', '.ipynb_checkpoints'} for part in relative.parts):
            continue
        if path.suffix in {'.pyc', '.pyo'}:
            continue
        digest.update(str(relative).encode() + b'\0')
        digest.update(path.read_bytes())
    return digest.hexdigest()


def configure(args):
    root, state = args.source_root.resolve(), args.state_dir.resolve()
    if state == root or root in state.parents:
        raise ValueError('Keep --state-dir outside the source tree')
    if not re.fullmatch(r'[a-z0-9][a-z0-9_-]+', args.project):
        raise ValueError('Use a lowercase Compose project name of at least two characters')
    commit = args.commit or capture(['git', '-C', str(root), 'rev-parse', 'HEAD'])
    if not re.fullmatch(r'[0-9a-f]{40}', commit):
        raise ValueError('--commit must identify the full base source revision; runtime source is additionally hashed')
    digest = source_digest(root)
    inventory = [tuple(value.strip() for value in line.split(',')) for line in capture(
        ['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits']).splitlines()]
    selected = [uuid for index, uuid in inventory if args.gpu in {index, uuid}]
    if len(selected) != 1:
        raise ValueError('--gpu must resolve to exactly one actual nvidia-smi index or UUID; all/lists are forbidden')
    gpu = selected[0]
    image = json.loads(capture(['docker', 'image', 'inspect', args.image]))[0]
    if image['Architecture'] != 'amd64':
        raise ValueError('The evaluator image must be linux/amd64')
    # Reuse dependencies only when their declared inputs match. Runtime source is
    # mounted explicitly; an old image's baked COSMOS_COMMIT is never trusted.
    inputs = ['requirements.txt', 'setup.py', 'setup.cfg', 'launchable/Dockerfile', 'launchable/public-harness.patch']
    program = 'import hashlib,json,pathlib; print(json.dumps({p:hashlib.sha256(pathlib.Path("/opt/vlmevalkit",p).read_bytes()).hexdigest() for p in ' + repr(inputs) + '}))'
    baked = json.loads(capture(['docker', 'run', '--rm', '--network', 'none', '--entrypoint', 'python', args.image, '-c', program]))
    for name in inputs:
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != baked[name]:
            raise ValueError(f'Prebuilt dependency input differs: {name}; build a matching evaluator image explicitly')
    env = dict(os.environ, JUPYTER_TOKEN=os.urandom(32).hex(), LAUNCHABLE_STATE_DIR=str(state), COSMOS_COMMIT=commit)
    base = json.loads(capture(['docker', 'compose', '--env-file', '/dev/null', '--profile', 'local', '-f', str(root / 'launchable/compose.yaml'), 'config', '--format', 'json'], env=env))
    model = base['services']['cosmos-endpoint']
    model.pop('profiles', None)
    model['deploy']['resources']['reservations']['devices'] = [{'driver': 'nvidia', 'device_ids': [gpu], 'capabilities': ['gpu']}]
    model['networks'] = ['private', 'download']
    model['volumes'] = [f'{state}/cache:/cache', f'{state}/data:/data:ro']
    common = {
        'image': args.image, 'platform': 'linux/amd64',
        'working_dir': '/opt/vlmevalkit/launchable',
        'environment': {'COSMOS_COMMIT': commit, 'COSMOS_SOURCE_SHA256': digest,
                        'HF_HOME': '/cache/huggingface', 'HOME': '/tmp/cosmos-home',
                        'PYTHONUNBUFFERED': '1', 'AWS_EC2_METADATA_DISABLED': 'true',
                        'COSMOS_API_BASE': 'http://cosmos-endpoint:8080/v1/chat/completions',
                        'COSMOS_MODEL': base['services']['vantage-eval']['environment']['COSMOS_MODEL'],
                        'COSMOS_API_KEY': 'EMPTY'},
        'volumes': [f'{root}:/opt/vlmevalkit:ro', f'{state}/cache:/cache', f'{state}/data:/data', f'{state}/results:/results'],
        'entrypoint': ['python', 'launchable.py'],
    }
    evaluate = dict(common, networks=['private'])
    evaluate['environment'] = dict(common['environment'], AWS_CONFIG_FILE='/dev/null', AWS_SHARED_CREDENTIALS_FILE='/dev/null')
    prepare = dict(common, networks=['download'])
    prepare['environment'] = dict(common['environment'], AWS_CONFIG_FILE='/credentials/config', AWS_SHARED_CREDENTIALS_FILE='/credentials/credentials', AWS_PROFILE='team-cosmos')
    prepare['volumes'] = common['volumes'] + [f'{state}/credentials/team-cosmos:/credentials:ro']
    config = {'services': {'cosmos-endpoint': model, 'evaluate': evaluate, 'prepare': prepare},
              'networks': {'private': {'internal': True}, 'download': {}}}
    state.mkdir(mode=0o700, parents=True, exist_ok=True)
    for name in ['cache', 'data', 'results', 'credentials/team-cosmos']:
        (state / name).mkdir(mode=0o700, parents=True, exist_ok=True)
    path = state / 'launchpad-compose.json'
    path.write_text(json.dumps(config, indent=2) + '\n')
    path.chmod(0o600)
    evidence = {'cosmos_commit': commit, 'source_sha256': common['environment']['COSMOS_SOURCE_SHA256'],
                'source_root': str(root), 'evaluator_image': args.image, 'evaluator_image_id': image['Id'],
                'dependency_inputs_match': True, 'runtime_source_bind_mounted_read_only': True,
                'serving_image': model['image'], 'serving_command': model['command'],
                'gpu': gpu, 'project': args.project, 'state_dir': str(state),
                'inference_credentials_mounted': False, 'inference_network_internal': True}
    (state / 'launchpad-provenance.json').write_text(json.dumps(evidence, indent=2) + '\n')
    return state, ['docker', 'compose', '--project-name', args.project, '-f', str(path)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state-dir', required=True, type=Path)
    parser.add_argument('--project', default='cosmos-launchpad')
    parser.add_argument('--gpu', default='0', help='One GPU index or UUID')
    parser.add_argument('--image', default='cosmos3-vantage-eval:local', help='Existing, dependency-compatible evaluator image')
    parser.add_argument('--source-root', type=Path, default=HERE.parents[1], help='VLMEvalKit root in a Git checkout')
    parser.add_argument('--commit', help='Full base Git revision for an exported tree; working changes are identified by the separate source digest')
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ['up', 'status', 'stop', 'check', 'probe', 'prepare']:
        sub.add_parser(name)
    smoke = sub.add_parser('smoke')
    smoke.add_argument('--run-name', required=True, help='Fresh directory name under state/results')
    smoke.add_argument('--score', action='store_true', help='Native local scoring; requires actual internal ground truth')
    args = parser.parse_args()
    if args.command in {'status', 'stop'}:
        config = args.state_dir.resolve() / 'launchpad-compose.json'
        if not config.is_file():
            raise ValueError('No configured project at this state path; run up first')
        compose = ['docker', 'compose', '--project-name', args.project, '-f', str(config)]
        call(compose + (['ps', '--all'] if args.command == 'status' else ['stop', 'cosmos-endpoint']))
        return
    state, compose = configure(args)
    if args.command == 'up':
        call(compose + ['up', '-d', '--no-build', 'cosmos-endpoint'])
        call(compose + ['run', '--rm', '--no-deps', 'evaluate', 'check', '--wait', '1800'])
    elif args.command == 'check':
        call(compose + ['run', '--rm', '--no-deps', 'evaluate', 'check', '--wait', '1800'])
    elif args.command == 'probe':
        call(compose + ['run', '--rm', '--no-deps', '--entrypoint', 'python', 'prepare', 'launchpad/runtime.py', 'probe'])
    elif args.command == 'prepare':
        call(compose + ['run', '--rm', '--no-deps', 'prepare', 'prepare', '--source', 'internal', '--benchmarks', 'VANTAGE_VQA'])
    else:
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.run_name):
            raise ValueError('Use a simple run directory name')
        if (state / 'results' / args.run_name).exists():
            raise ValueError('Run directory already exists; select a fresh --run-name')
        work = '/results/' + args.run_name
        call(compose + ['run', '--rm', '--no-deps', 'evaluate', 'run', '--source', 'internal', '--benchmarks', 'VANTAGE_VQA', '--smoke', '--mode', 'infer', '--work-dir', work])
        call(compose + ['run', '--rm', '--no-deps', '--entrypoint', 'python', 'evaluate', 'launchpad/runtime.py', 'accept', '--work-dir', work] + (['--score'] if args.score else []))
        # Attach exact runtime source/dependency provenance to private outputs.
        provenance = json.loads((state / 'launchpad-provenance.json').read_text())
        call(compose + ['run', '--rm', '--no-deps', '--entrypoint', 'python', 'evaluate', '-c', 'import pathlib,sys; pathlib.Path(sys.argv[1]).write_text(sys.argv[2])', work + '/launchpad-provenance.json', json.dumps(provenance, indent=2) + '\n'])


if __name__ == '__main__':
    try:
        main()
    except (ValueError, subprocess.CalledProcessError) as exc:
        print(f'LaunchPad setup failed: {exc}', file=sys.stderr)
        sys.exit(1)
