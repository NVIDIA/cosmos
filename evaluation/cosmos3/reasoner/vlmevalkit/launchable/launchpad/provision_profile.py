#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export only team-cosmos privately through SSH stdin; never copy ~/.aws."""
import argparse
import json
import shlex
import subprocess


RECEIVER = r'''
import configparser, json, os, pathlib, sys
payload = json.load(sys.stdin)
directory = pathlib.Path(payload['directory']).expanduser()
if not directory.is_absolute():
    raise ValueError('Remote credentials directory must be absolute')
if directory.name != 'team-cosmos' or directory.parent.name != 'credentials':
    raise ValueError('Remote target must end with credentials/team-cosmos; refusing unrelated profile directories')
if any(path.is_symlink() for path in (directory, *directory.parents)):
    raise ValueError('Remote credentials path must not contain symlinks')
directory.mkdir(mode=0o700, parents=True, exist_ok=True)
directory.chmod(0o700)
credentials = configparser.ConfigParser(interpolation=None)
values = payload['credentials']
credentials['team-cosmos'] = {'aws_access_key_id': values['AccessKeyId'], 'aws_secret_access_key': values['SecretAccessKey']}
if values.get('SessionToken'):
    credentials['team-cosmos']['aws_session_token'] = values['SessionToken']
config = configparser.ConfigParser(interpolation=None)
config['profile team-cosmos'] = {'region': 'us-east-1', 'endpoint_url': 'https://pdx.s8k.io', 's3': '\nsignature_version = s3v4\npayload_signing_enabled = true\naddressing_style = path'}
for name, document in [('credentials', credentials), ('config', config)]:
    path = directory / name
    # Refuse symlinks and create with restrictive mode before writing bytes.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, 'w') as stream:
        document.write(stream)
print('Provisioned only team-cosmos; directory 0700, files 0600; no secret output.')
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', required=True, help='SSH host, from your authorized instance locator')
    parser.add_argument('--user', help='SSH user if not in SSH config')
    parser.add_argument('--remote-dir', required=True, help='Absolute state-dir/credentials/team-cosmos on the host')
    parser.add_argument('--control-path', help='Existing SSH ControlPath, if used')
    args = parser.parse_args()
    # Do not print/export credential values or pass them in argv/environment.
    exported = subprocess.run(['aws', 'configure', 'export-credentials', '--profile', 'team-cosmos', '--format', 'process'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    credentials = json.loads(exported.stdout)
    payload = json.dumps({'directory': args.remote_dir, 'credentials': credentials}).encode()
    command = ['ssh']
    if args.user:
        command += ['-l', args.user]
    if args.control_path:
        command += ['-o', 'ControlMaster=auto', '-o', 'ControlPersist=10m', '-o', 'ControlPath=' + args.control_path]
    command += [args.host, 'python3 -c ' + shlex.quote(RECEIVER)]
    subprocess.run(command, input=payload, check=True)


if __name__ == '__main__':
    main()
