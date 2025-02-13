set -x

script_dir=$(dirname "$(realpath "$0")")

cd $script_dir/../test

python3 start.py $script_dir/../../..