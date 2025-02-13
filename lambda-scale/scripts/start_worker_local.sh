#!/bin/bash
set -x

script_dir=$(dirname "$(realpath "$0")")

cd $script_dir/../serve/server && torchrun --nproc-per-node=2 server.py $script_dir/../../..

# > server_debug.log 2>&1 &