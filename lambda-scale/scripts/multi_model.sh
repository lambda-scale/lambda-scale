set -x

script_dir=$(dirname "$(realpath "$0")")

cd $script_dir/../test

python3 multi_model.py $script_dir/../../..