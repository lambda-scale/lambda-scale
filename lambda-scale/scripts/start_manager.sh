#!/bin/bash
set -x

script_dir=$(dirname "$(realpath "$0")")

cd $script_dir/../serve/manager && python3 main.py $script_dir/../../.. 
# > controller_debug.log 2>&1 & 