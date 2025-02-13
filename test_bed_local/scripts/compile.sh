#!/bin/bash
set -x

script_dir=$(dirname "$(realpath "$0")")

cd $script_dir/../../ && python3 setup.py install

cd test_bed_local/serve/communication/ipc
python3 setup.py install 