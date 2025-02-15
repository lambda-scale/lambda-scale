#!/bin/bash

# 使用salloc分配资源
salloc --job-name=gpu-hostname1 \
       --nodes=2 \
       --ntasks-per-node=1 \
       --cpus-per-task=10 \
       --gres=gpu:a40:1 \
       --partition=gpu &

salloc --job-name=gpu-hostname2 \
       --nodes=2 \
       --ntasks-per-node=1 \
       --cpus-per-task=10 \
       --gres=gpu:a40:1 \
       --partition=gpu &



# Enable or disable debug output
export DEBUG=true # Set to "true" to enable debug output, "false" to disable

# Function to print debug messages
debug() {
    if [ "$DEBUG" = "true" ]; then
        echo -e "$@" >&2
    fi
}

# Define application port
APP_PORT=5000

# Define the node configuration file
NODE_CONFIG_FILE="node_config.txt"
: >"$NODE_CONFIG_FILE" # Clear or create the file

# Retrieve the list of node hostnames using squeue for the current user's jobs
debug "Fetching node hostnames using squeue..."

sleep 20

node_hostnames=($(squeue -u qgh4hm -o "%.15N" | tail -n +2)) > host.log


echo $(squeue -u qgh4hm)

if [[ ${#node_hostnames[@]} -eq 0 ]]; then
    echo "No nodes found for the current user in the SLURM queue."
    exit 1
fi

# Iterate over each hostname to resolve IP and construct workers_addr
workers_addr=()
for hostname in "${node_hostnames[@]}"; do
    debug "Resolving IP for hostname: $hostname"
    node_ip=$(ping -c 1 "$hostname" | grep "PING" | awk '{print $3}' | tr -d '()')
    
    if [[ -n "$node_ip" ]]; then
        workers_addr+=("${node_ip}:${APP_PORT}")
        debug "Resolved $hostname to $node_ip and added to workers_addr."
        echo "$hostname,$node_ip" >>"$NODE_CONFIG_FILE"
    else
        echo "Failed to resolve IP for hostname: $hostname" >&2
    fi
done

# Convert the workers_addr array to a comma-separated string
workers_addr_str=$(
    IFS=','
    echo "${workers_addr[*]}"
)

# Output the constructed workers address string
echo "Workers Address String: $workers_addr_str"

# Export the workers address string for use in other scripts
export WORKERS_ADDR=$workers_addr_str

# Final message
echo "SLURM script completed successfully."
