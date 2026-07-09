#!/bin/bash
DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cp $DIR/../config.py $DIR/../vlmeval/
cp $DIR/../misc/* $DIR/../vlmeval/vlm/misc/