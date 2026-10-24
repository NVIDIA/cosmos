#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded S3 read probe and independent five-question internal smoke acceptance."""
import argparse
import contextlib
import json
import os
from pathlib import Path
import sys

LAUNCHABLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAUNCHABLE))


def probe():
    import boto3
    from botocore.config import Config

    client = boto3.Session(profile_name='team-cosmos').client(
        's3', endpoint_url='https://pdx.s8k.io', region_name='us-east-1',
        config=Config(signature_version='s3v4', s3={'addressing_style': 'path', 'payload_signing_enabled': True},
                      connect_timeout=10, read_timeout=30, retries={'max_attempts': 2}))
    response = client.list_objects_v2(Bucket='cosmos_understanding', Prefix='benchmark/vantage_benchmark_hf_release_annotations/VQA/', MaxKeys=16)
    objects = [item for item in response.get('Contents', []) if item['Size'] > 0]
    if not objects:
        raise RuntimeError('Bounded VQA listing returned no nonempty object')
    body = client.get_object(Bucket='cosmos_understanding', Key=objects[0]['Key'], Range='bytes=0-0')['Body']
    try:
        if len(body.read()) != 1:
            raise RuntimeError('Ranged read did not return exactly one byte')
    finally:
        body.close()
    print(json.dumps({'vqa_bounded_listing': True, 'one_byte_get': True, 's3_writes': False}))


def accept(work, score):
    import pandas as pd
    import launchable

    # No credentials, metadata provider, or non-local network are available to
    # this service. The upstream cache may attempt S3 initialization, but cannot
    # authenticate or reach S3; generated entries stay in the local cache.
    import boto3
    if boto3.Session().get_credentials() is not None:
        raise RuntimeError('Inference unexpectedly has AWS credentials')
    if os.environ.get('AWS_CONFIG_FILE') != '/dev/null' or os.environ.get('AWS_SHARED_CREDENTIALS_FILE') != '/dev/null':
        raise RuntimeError('Inference credential files are not disabled')
    paths = [path for path in launchable.prediction_files(work) if path.stem == 'cosmos_VANTAGE_VQA']
    if len(paths) != 1:
        raise RuntimeError(f'Expected one fresh native prediction file; found {len(paths)}')
    path = paths[0]
    frame = pd.read_csv(path, sep='\t') if path.suffix == '.tsv' else pd.read_json(path) if path.suffix == '.json' else pd.read_excel(path)
    if len(frame) != 5 or 'prediction' not in frame or 'index' not in frame:
        raise RuntimeError('Smoke must contain exactly five indexed predictions')
    values = frame['prediction'].fillna('').astype(str).str.strip()
    markers = ['VIDEO_LOAD_FAILED', 'Failed to obtain answer via API', 'Traceback (most recent call last)', 'Internal Server Error', 'ConnectionError']
    if values.eq('').any() or any(values.str.contains(marker, regex=False, case=False).any() for marker in markers):
        raise RuntimeError('Empty, video-load or API-failure prediction rejected')
    for log in work.rglob('run.log'):
        content = log.read_text(errors='replace')
        if any(marker.lower() in content.lower() for marker in markers):
            raise RuntimeError('Failure marker in native run log')
    sys.path.insert(0, str(launchable.ROOT))
    sys.path.insert(0, str(launchable.KIT))
    import run_all
    from run import build_dataset_from_config

    cfg = run_all.build_config(run_all.load_model_layer('cosmos'), 'metropolis', 'VANTAGE_VQA', render=False)
    cfg['data']['VANTAGE_VQA'].update(limit=5, random_state=0)
    # Dataset initialization and native scorer may print private labels. Retain
    # that output only alongside private results, never in the console summary.
    with (work / 'native-score.log').open('w') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        os.environ['LMUData'] = '/data/internal/LMUData'
        dataset = build_dataset_from_config(cfg['data'], 'VANTAGE_VQA')
        expected = [str(value) for value in dataset.data['index']]
        actual = [str(value) for value in frame['index']]
        if len(expected) != 5 or len(set(actual)) != 5 or set(expected) != set(actual):
            raise RuntimeError('Missing, duplicate, dropped or unexpected sample IDs')
        native = None
        if score:
            if 'answer' not in frame or frame['answer'].isna().any() or frame['answer'].astype(str).str.strip().eq('').any():
                raise RuntimeError('Native scoring requires real ground truth for all five samples')
            expected_answers = dataset.data.set_index('index')['answer'].astype(str).to_dict()
            if any(str(row['answer']) != expected_answers[row['index']] for _, row in frame.iterrows()):
                raise RuntimeError('Prediction ground truth differs from the prepared native dataset')
            summary = dataset.evaluate(str(path), verbose=False)
            metrics = pd.read_csv(path.with_name(path.stem + '_metrics.csv'))
            overall = metrics.loc[metrics['Category'] == 'Overall'].iloc[0]
            native = {'acc': float(summary['Overall']), 'correct': int(overall['Correct']), 'total': int(overall['Total'])}
            if native['total'] != 5:
                raise RuntimeError('Native scorer did not account for all five samples')
    evidence = {'accepted': True, 'questions': 5, 'sample_indexes': sorted(actual),
                'source': 'internal', 'cosmos_commit': os.environ['COSMOS_COMMIT'],
                'source_sha256': os.environ['COSMOS_SOURCE_SHA256'], 'aws_credentials_available': False,
                'inference_network_internal': True, 'native_overall': native,
                'scope': 'five-question smoke only; not published-reference parity'}
    (work / 'launchpad-acceptance.json').write_text(json.dumps(evidence, indent=2) + '\n')
    print(json.dumps(evidence, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('probe')
    check = sub.add_parser('accept')
    check.add_argument('--work-dir', type=Path, required=True)
    check.add_argument('--score', action='store_true')
    args = parser.parse_args()
    probe() if args.command == 'probe' else accept(args.work_dir, args.score)


if __name__ == '__main__':
    main()
