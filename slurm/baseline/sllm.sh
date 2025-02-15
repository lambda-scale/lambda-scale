#!/bin/bash


# Retrieve the list of hostnames from SLURM_NODELIST and store them in an array
export NODE_HOSTNAMES=($(srun hostname | sort -u))  # Retrieve node hostnames

echo "SLURM assigned nodes: ${NODE_HOSTNAMES[*]}"
export BASE_DIR=$(sed -n 's/^node_base_dir[[:space:]]*=[[:space:]]*\(.*\)/\1/p' derecho_node.cfg | sed 's/[[:space:]]*$//')

# # # Loop through the nodes and execute node-specific logic
# for node_id in "${!NODE_HOSTNAMES[@]}"; do
#     # Extract the hostname for the current node
#     node_hostname=${NODE_HOSTNAMES[$node_id]}
    
#     # Define a node-specific directory based on its ID
#     node_dir="${BASE_DIR}/slurm_node${node_id}"
    
#     # Export environment variables specific to the current node
#     export NODE_ID="$node_id"
#     export NODE_HOSTNAME="$node_hostname"
#     export NODE_DIR="$node_dir"
#     node_conda_dir="${node_dir}/utils/miniconda3"

#     echo "Exported node-specific variables: NODE_ID=${NODE_ID}, NODE_HOSTNAME=${NODE_HOSTNAME}, NODE_DIR=${NODE_DIR}"

#     if [[ "$node_id" -eq 0 ]]; then
#         # Logic for the head node
#         echo "Executing head node setup on ${node_hostname} (ID: ${node_id})"
#         echo "Node conda dir: ${node_conda_dir}"

#         # Define the command to set up the head node
#         head_command="source ${node_conda_dir}/bin/activate base && \
#         conda activate sllm && \
#         ray start --head --port=6379 --num-cpus=4  --num-gpus=0 \
#         --resources='{\"control_node\": 1}' --block"

#         echo "Head command: $head_command"

#         # Execute the head node setup
#         srun --nodes=1 --ntasks=1 --nodelist=${node_hostname} --gres=none --job-name=head_setup --output=${node_dir}/head_setup.log bash -c "${head_command}" &
#     else
#         # Logic for other worker nodes
#         echo "Executing worker node setup on ${node_hostname} (ID: ${node_id})"
#         head_node_ip=$(ping -c 1 ${NODE_HOSTNAMES[0]} | grep 'PING' | awk -F'[()]' '{print $2}')
#         echo "Head node IP: ${head_node_ip}"
#         # Define the command to set up a worker node
#         worker_command="source ${node_conda_dir}/bin/activate base && \
#         conda activate sllm-worker && \
#         ray start --address=${head_node_ip}:6379 --num-cpus=4 --num-gpus=1 \
#         --resources='{\"worker_node\": 1, \"worker_id_${node_id}\": ${node_id}}' --block"

#         # Execute the worker node setup
#         srun --nodes=1 --ntasks=1 --nodelist=${node_hostname}  --gres=gpu:v100:1 --job-name=worker_setup --output=${node_dir}/worker_setup.log bash -c "${worker_command}" &
#     fi
# done


# Define the head node ID (assumed to be 0)
head_node_id=0
head_node_hostname=${NODE_HOSTNAMES[$head_node_id]}

# # Step 1: Process all worker nodes first
for node_id in "${!NODE_HOSTNAMES[@]}"; do
    if [[ "$node_id" -ne "$head_node_id" ]]; then
        # Extract the hostname for the current node
        node_hostname=${NODE_HOSTNAMES[$node_id]}
        
        # Define a node-specific directory based on its ID
        node_dir="${BASE_DIR}/slurm_node${node_id}"
        
        # Export environment variables specific to the current worker node
        export NODE_ID="$node_id"
        export NODE_HOSTNAME="$node_hostname"
        export NODE_DIR="$node_dir"
        node_conda_dir="${node_dir}/utils/miniconda3"
        
        echo "Exported worker node-specific variables: NODE_ID=${NODE_ID}, NODE_HOSTNAME=${NODE_HOSTNAME}, NODE_DIR=${NODE_DIR}"
        
        # Head node IP address (needed by workers)
        head_node_ip=$(ping -c 1 ${head_node_hostname} | grep 'PING' | awk -F'[()]' '{print $2}')

        # export LD_LIBRARY_PATH=\"${node_dir}/utils/lib:${node_dir}/utils/lib64:\${LD_LIBRARY_PATH}\" && \
        # export PATH=\"${node_dir}/utils/bin:\${PATH}\" && \
        # export LIBRARY_PATH=\"${node_dir}/utils/lib:${node_dir}/utils/lib64:\${LIBRARY_PATH}\" && \
        
        # Define the worker node command
        # worker_start_command="
        # export LD_LIBRARY_PATH=\"${node_dir}/utils/lib:\${LD_LIBRARY_PATH}\" && \
        # export PATH=\"${node_dir}/utils/bin:\${PATH}\" && \
        # export LIBRARY_PATH=\"${node_dir}/utils/lib:\${LIBRARY_PATH}\" && \
        # source ${node_conda_dir}/bin/activate base && \
        # module load gcc/13.3.0 && \
        # module load cuda/12.4.1 && \
        # conda activate sllm-worker && \
        # export CUDA_VISIBLE_DEVICES=0 && \
        # sllm-store-server --storage_path ${node_dir}/FaaScale/baselines/ServerlessLLM/models --mem_pool_size 32"


        CUSTOM_GLIBC_PATH="/scratch/qgh4hm/chaobo/slurm_nodes/slurm_node0/utils/glibc-2.34"
        DYNAMIC_LINKER="$CUSTOM_GLIBC_PATH/lib/ld-linux-x86-64.so.2"
        GLIBC_DIR="$CUSTOM_GLIBC_PATH/lib"
        GCC_LIB="/apps/software/standard/core/gcc/13.3.0/lib64"

        PYTHON_PATH="/scratch/qgh4hm/chaobo/slurm_nodes/slurm_node0/utils/miniconda3/envs/sllm-worker/bin/python3.10"

        SLLM_STORE="/scratch/qgh4hm/chaobo/slurm_nodes/slurm_node0/utils/miniconda3/envs/sllm-worker/lib/python3.10/site-packages/sllm_store"


        worker_start_command="
        export LD_LIBRARY_PATH=\"${GLIBC_DIR}:\${GCC_LIB}:\${SLLM_STORE}:\${LD_LIBRARY_PATH}\" && \
        export PATH=\"${SLLM_STORE}:\${PATH}\" && \
        export LIBRARY_PATH=\"${GLIBC_DIR}:\${SLLM_STORE}:\${LIBRARY_PATH}\" && \
        source ${node_conda_dir}/bin/activate base && \
        module load gcc/13.3.0 && \
        module load cuda/12.2.2 && \
        conda activate sllm-worker && \
        export CUDA_VISIBLE_DEVICES=0 && \
        $DYNAMIC_LINKER --library-path "$GLIBC_DIR:$GCC_LIB:$SLLM_STORE:$LIBRARY_PATH" $node_dir/utils/miniconda3/envs/sllm-worker/lib/python3.10/site-packages/sllm_store/sllm_store_server"
        
        # Execute the worker node setup
        echo "Executing worker node setup on ${node_hostname} (ID: ${node_id})"
        srun --nodes=1 --ntasks=1 --nodelist=${node_hostname} --gres=gpu:v100:1 --job-name=worker_start --output=${node_dir}/worker_start.log bash -c "${worker_start_command}" &
    fi
done

# Wait for worker nodes to complete setup
# sleep 10

# export HF_TOKEN="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"
# huggingface-cli login --token $HF_TOKEN

# # Step 2: Process the head node
# node_hostname=${NODE_HOSTNAMES[$head_node_id]}
# node_dir="${BASE_DIR}/slurm_node${head_node_id}"
# node_conda_dir="${node_dir}/utils/miniconda3"

# # Export environment variables specific to the head node
# export NODE_ID="$head_node_id"
# export NODE_HOSTNAME="$node_hostname"
# export NODE_DIR="$node_dir"

# echo "Exported head node-specific variables: NODE_ID=${NODE_ID}, NODE_HOSTNAME=${NODE_HOSTNAME}, NODE_DIR=${NODE_DIR}"

# # Define the head node command
# head_start_command="source ${node_conda_dir}/bin/activate base && \
# module load gcc/13.3.0 && \
# conda activate sllm && \
# sllm_serve start"
# # python3 ${node_dir}/FaaScale/baselines/ServerlessLLM/sllm/serve/commands/serve/sllm_serve.py start"

# # Execute the head node setup
# echo "Executing head node setup on ${node_hostname} (ID: ${head_node_id})"
# echo "Node conda dir: ${node_conda_dir}"
# echo "Head command: $head_start_command"
# srun --nodes=1 --ntasks=1 --nodelist=${node_hostname} --gres=none --job-name=head_start --output=${node_dir}/head_start.log bash -c "${head_start_command}" &


# node_hostname=${NODE_HOSTNAMES[0]}

# # Define a node-specific directory based on its ID
# node_dir="${BASE_DIR}/slurm_node${node_id}"

# # Export environment variables specific to the current node
# export NODE_ID="$node_id"
# export NODE_HOSTNAME="$node_hostname"
# export NODE_DIR="$node_dir"
# node_conda_dir="${node_dir}/utils/miniconda3"

# echo "Exported node-specific variables: NODE_ID=${NODE_ID}, NODE_HOSTNAME=${NODE_HOSTNAME}, NODE_DIR=${NODE_DIR}"
# echo "Executing head node setup on ${node_hostname} (ID: ${node_id})"
# echo "Node conda dir: ${node_conda_dir}"
# head_node_ip=$(ping -c 1 ${NODE_HOSTNAMES[0]} | grep 'PING' | awk -F'[()]' '{print $2}')

# # Define the command to set up the head node
# deploy_command="source ${node_conda_dir}/bin/activate base && \
# conda activate sllm && \
# export LLM_SERVER_URL=http://${head_node_ip}:8343/ && \
# module load gcc/13.3.0 && \
# bash ${node_dir}/FaaScale/baselines/ServerlessLLM/scripts/start_sllm_client.sh"

# echo "Head command: $head_command"
# # Execute the head node setup
# srun --nodes=1 --ntasks=1 --nodelist=${node_hostname} --gres=none  --job-name=head_setup --output=${node_dir}/worker_deploy.log bash -c "${deploy_command}" &

# request_command="curl http://${head_node_ip}:8343/v1/chat/completions \
# -H 'Content-Type: application/json' \
# -d '{\"model\": \"facebook/opt-1.3b\", \"messages\": [{\"role\": \"system\", \"content\": \"You are a helpful assistant.\"}, {\"role\": \"user\", \"content\": \"What is your name?\"}]}'"


# srun --nodes=1 --ntasks=1 --nodelist=${node_hostname}  --gres=none --job-name=head_setup --output=${node_dir}/worker_request.log bash -c "${request_command}" &

wait
echo "All nodes have been set up successfully."