set -x

script_dir=$(dirname "$(realpath "$0")")

cd $script_dir/../../test_bed_local/serve/model_info/models/llama

export HF_ENDPOINT=https://hf-mirror.com

# huggingface-cli download --resume-download --local-dir-use-symlinks False meta-llama/llama-2-7b --local-dir llama-2-7b

# huggingface-cli download --resume-download --local-dir-use-symlinks False meta-llama/llama-2-13b --local-dir llama-2-13b

# huggingface-cli download --resume-download --local-dir-use-symlinks False meta-llama/llama-2-70b --local-dir llama-2-70b

cd ../../../../../test_bed_local/serve/server/model_storage

huggingface-cli download --resume-download --local-dir-use-symlinks False jcbjcc/llama-2-7b --local-dir llama-2-7b

huggingface-cli download --resume-download --local-dir-use-symlinks False jcbjcc/llama-2-13b --local-dir llama-2-13b

huggingface-cli download --resume-download --local-dir-use-symlinks False jcbjcc/llama-2-70b --local-dir llama-2-70b
