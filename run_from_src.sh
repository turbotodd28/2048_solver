#!/bin/bash
# Script to run reward_sweep_GPU.py from src directory
export PYTHONPATH="${PYTHONPATH}:$(dirname $(dirname $(readlink -f $0)))"
cd "$(dirname "$0")"
python reward_sweep_GPU.py "$@" 