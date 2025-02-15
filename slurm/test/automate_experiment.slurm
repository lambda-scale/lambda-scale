#!/bin/bash
#SBATCH --job-name=gpu-scaling
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --partition=normal
#SBATCH --mem-per-gpu=40G
#SBATCH --account=infattllm


# Enable or disable debug output
export DEBUG=true # Set to "true" to enable debug output, "false" to disable

# Function to print debug messages
debug() {
    if [ "$DEBUG" = "true" ]; then
        echo -e "$@" >&2
    fi
}
# Set the port configuration for the application
export APP_PORT=6000
export P2P_PORT=31675
export VIEW_PORT=23580

# Set and export environment variables at the top
export LOG_LEVEL="debug"
export LOG_FILE_PREFIX="bcast_worker"

# Extract and export target application from config
export TARGET_APP=$(sed -n 's/^target_app[[:space:]]*=[[:space:]]*\(.*\)/\1/p' derecho_node.cfg | sed 's/[[:space:]]*$//')

# Set and export directories and configurations
export FGS_DIR="/src/fast-gpu-scaling"
export GFS_DIR="/src/gpu-fast-scaling"
export REMOTE_RDMC_DIR="/src/RDMC-GDR"
export CONFIGURE_SCRIPT="/src/RDMC-GDR/slurm/deploy/configure.slurm"
export READY_MSG="stand by"

# Extract and export local RDMC directory and node configuration file
export BASE_DIR=$(sed -n 's/^node_base_dir[[:space:]]*=[[:space:]]*\(.*\)/\1/p' derecho_node.cfg | sed 's/[[:space:]]*$//')
export LOCAL_RDMC_DIR=$(sed -n 's/^local_rdmc_dir[[:space:]]*=[[:space:]]*\(.*\)/\1/p' derecho_node.cfg | sed 's/[[:space:]]*$//')
export LOCAL_SLURM_DIR=$LOCAL_RDMC_DIR/slurm
export NODE_CONFIG_FILE=$(sed -n 's/^node_config_file[[:space:]]*=[[:space:]]*\(.*\)/\1/p' derecho_node.cfg | sed 's/[[:space:]]*$//')

# Set and export experiment and wait times
export EXPERIMENT_WAIT_TIME=6
export WORKER_INTER_WAIT_TIME=$(sed -n 's/^worker_inter_wait_time[[:space:]]*=[[:space:]]*\(.*\)/\1/p' derecho_node.cfg | sed 's/[[:space:]]*$//')
export WORKER_COMM_ESTABLISH_WAIT_TIME=$(sed -n 's/^worker_comm_establish_wait_time[[:space:]]*=[[:space:]]*\(.*\)/\1/p' derecho_node.cfg | sed 's/[[:space:]]*$//')
export CONTROLLER_WAIT_TIME=$(sed -n 's/^controller_wait_time[[:space:]]*=[[:space:]]*\(.*\)/\1/p' derecho_node.cfg | sed 's/[[:space:]]*$//')
export CLIENT_WAIT_TIME=$(sed -n 's/^client_wait_time[[:space:]]*=[[:space:]]*\(.*\)/\1/p' derecho_node.cfg | sed 's/[[:space:]]*$//')

# Extract and export number of experiments and workers
export NUM_EXPERIMENTS=$(sed -n 's/^num_experiments[[:space:]]*=[[:space:]]*\(.*\)/\1/p' derecho_node.cfg | sed 's/[[:space:]]*$//')
export TOTAL_WORKERS=$(sed -n 's/^total_workers[[:space:]]*=[[:space:]]*\(.*\)/\1/p' derecho_node.cfg | sed 's/[[:space:]]*$//')
export MUL_FACTOR=2 # Number of workers scaling factor
export GPU_PER_NODE=1

# Extract and export resource platform
export RESOURCE_PLATFORM=$(sed -n 's/^resource_platform[[:space:]]*=[[:space:]]*\(.*\)/\1/p' derecho_node.cfg | sed 's/[[:space:]]*$//')

export NODE_HOSTNAMES=($(scontrol show hostname ${SLURM_NODELIST}))
# export NODE_HOSTNAMES=($(scontrol show hostname "${SLURM_NODELIST}"))

# Set model
export MODEL_NAME="llama-2-7b"


# Set and export log patterns
export LOG_PATTERN_1="*.log"
export LOG_PATTERN_2="*.cfg"

export FI_LOG_LEVEL=debug

# Main variables for fast-gpu-scaling or gpu-fast-scaling
if [[ "$TARGET_APP" == "fgs" ]]; then
    export TARGET_REMOTE_DIR="$FGS_DIR"
    export REMOTE_APP_DIR="$FGS_DIR"

    # Path variables
    export WORKER_DIR="${FGS_DIR}/src/worker"
    export CTRL_DIR="${FGS_DIR}/src/controller"
    export CLIENT_DIR="${FGS_DIR}/test"

    # Python script variables
    export WORKER_FILE="__main__.py"
    export CTRL_FILE="__main__.py"
    export CLIENT_FILE="client.py"

else
    export TARGET_REMOTE_DIR="$GFS_DIR"
    export REMOTE_APP_DIR="$GFS_DIR"

    # Path variables
    export START_SCRIPT_DIR="${GFS_DIR}/test_bed_local/scripts"
    export WORKER_DIR="${GFS_DIR}/test_bed_local/serve/server"
    export CTRL_DIR="${GFS_DIR}/test_bed_local/serve/controller"
    export CLIENT_DIR="${GFS_DIR}/test"

    # Python script variables
    export WORKER_FILE="start_worker.sh"
    export CTRL_FILE="start_manager.sh"
    export START_FILE="start.sh"
    export CLIENT_FILE="send_request.sh"
fi

# Begin script execution
echo "-----------------------------------------"
echo "Unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_MEM_PER_NODE..."
echo "-----------------------------------------"
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_MEM_PER_NODE
echo ""
echo ""


echo "-----------------------------------------"
echo "Set logging configuration and SSH details..."
echo "-----------------------------------------"
echo "Log Level: $LOG_LEVEL, Log File Prefix: $LOG_FILE_PREFIX"
echo ""
echo ""

echo "-----------------------------------------"
echo "Extract target application from config..."
echo "-----------------------------------------"
echo "Target Application: $TARGET_APP"
if [[ "$TARGET_APP" != "fgs" && "$TARGET_APP" != "gfs" ]]; then
    echo "derecho_node.cfg error. target_app should be either 'fgs' or 'gfs'. Exiting..."
    exit 1
fi
echo ""
echo ""

echo "-----------------------------------------"
echo "Define remote directories and configurations..."
echo "-----------------------------------------"
echo "FGS Directory: $FGS_DIR"
echo "GFS Directory: $GFS_DIR"
echo "Remote RDMC Directory: $REMOTE_RDMC_DIR"
echo "Configure Script: $CONFIGURE_SCRIPT"
echo "Log Level: $LOG_LEVEL"
echo ""
echo ""

echo "-----------------------------------------"
echo "Set local RDMC directory and node configuration file..."
echo "-----------------------------------------"
echo "Local RDMC Directory: $LOCAL_RDMC_DIR"
echo "Node Configuration File: $NODE_CONFIG_FILE"
echo ""
echo ""

debug "- LOCAL_RDMC_DIR:${LOCAL_RDMC_DIR} \n" \
    "- NODE_CONFIG_FILE:${NODE_CONFIG_FILE} \n" \
    "- NUM_EXPERIMENTS:${NUM_EXPERIMENTS} \n" \
    "- WORKER_INTER_WAIT_TIME:${WORKER_INTER_WAIT_TIME} \n" \
    "- WORKER_COMM_ESTABLISH_WAIT_TIME:${WORKER_COMM_ESTABLISH_WAIT_TIME} \n" \
    "- CONTROLLER_WAIT_TIME:${CONTROLLER_WAIT_TIME} \n" \
    "- CLIENT_WAIT_TIME:${CLIENT_WAIT_TIME} \n" \
    "- TOTAL_WORKERS:${TOTAL_WORKERS} \n" \
    "- TARGET_REMOTE_DIR:${TARGET_REMOTE_DIR}  \n" \
    "- REMOTE_APP_DIR:${REMOTE_APP_DIR}  \n" \
    "- WORKER_DIR:${WORKER_DIR}  \n" \
    "- CTRL_DIR:${CTRL_DIR}  \n" \
    "- CLIENT_DIR:${CLIENT_DIR}  \n" \
    "- WORKER_FILE:${WORKER_FILE}  \n" \
    "- CTRL_FILE:${CTRL_FILE}  \n" \
    "- CLIENT_FILE:${CLIENT_FILE} \n" \
    "- LOG_PATTERN_1:${LOG_PATTERN_1}  \n" \
    "- LOG_PATTERN_2:${LOG_PATTERN_2}  \n"

echo "-----------------------------------------"
echo "Configure experiment and worker wait times..."
echo "-----------------------------------------"
echo "Experiment Wait Time: $EXPERIMENT_WAIT_TIME seconds"
echo "Worker Inter-Wait Time: $WORKER_INTER_WAIT_TIME seconds"
echo "Worker Comm Establish Wait Time: $WORKER_COMM_ESTABLISH_WAIT_TIME seconds"
echo "Controller Wait Time: $CONTROLLER_WAIT_TIME seconds"
echo "Client Wait Time: $CLIENT_WAIT_TIME seconds"
echo ""
echo ""

echo "-----------------------------------------"
echo "Extract number of experiments and workers from config..."
echo "-----------------------------------------"
echo "Number of Experiments: $NUM_EXPERIMENTS"
echo "Total Workers: $TOTAL_WORKERS, Scaling Factor: $MUL_FACTOR"
echo ""
echo ""

echo "-----------------------------------------"
echo "Set resource platform and SSH environment variables..."
echo "-----------------------------------------"
echo "Resource Platform: $RESOURCE_PLATFORM"
echo ""
echo ""

echo "---------------------------------------------------------------------------------------------"
echo "Starting experiment automation process. This automation process includes the following steps:"
echo "---------------------------------------------------------------------------------------------"
echo "1. Configure both RDMC-GDR and application. App includes: a) fast-gpu-scaling (FGS) and b) gpu-fast-scaling (GFS)"
echo "2. Launch application and RDMC instances on each node"
echo "3. Cleanup ports used by upper layer applications and RDMC-GDR layer"
echo "4. Collect worker's broadcasting and RDMC-GDR log files from each node to local machine (disable push to RDMC-GDR repo)"
echo ""
echo ""

echo "-----------------------------------------"
echo "Fetch hostname list from SLURM cluster..."
echo "-----------------------------------------"
echo "Hostnames fetched: ${NODE_HOSTNAMES[@]}"
echo ""
echo ""

# echo "---------------------------------"
# echo "Build hostname array mapping..."
# echo "---------------------------------"



# Function to configure the cluster by executing the configuration script on each node
echo "-----------------------------------------"
echo "Set logging configuration"
echo "-----------------------------------------"
echo "Log Level: $LOG_LEVEL, Log File Prefix: $LOG_FILE_PREFIX"
echo ""
echo ""

echo "-----------------------------------------"
echo "Extract target application from config..."
echo "-----------------------------------------"
echo "Target Application: $TARGET_APP"
if [[ "$TARGET_APP" != "fgs" && "$TARGET_APP" != "gfs" ]]; then
    echo "derecho_node.cfg error. target_app should be either 'fgs' or 'gfs'. Exiting..."
    exit 1
fi
echo ""
echo ""

declare -A hostnames
# Clear or create the node_config_file
: >"$NODE_CONFIG_FILE"

if [[ -z "$NODE_HOSTNAMES" ]]; then
    echo "NODE_HOSTNAMES is not set or is empty."
    exit 1
fi
echo ""
echo ""

echo "------------------------------"
echo "Configuring node id and ip..."
echo "------------------------------"
workers_addr=()
# Iterate through node IDs and hostnames, writing IP addresses and node IDs to the config file
for node_id in "${!NODE_HOSTNAMES[@]}"; do
    # Map node_id to hostname
    hostnames["$node_id"]="${NODE_HOSTNAMES[$node_id]}"
    # hostnames["$node_id"]="${hostname}"
    
    if [[ "$node_id" -eq 0 ]]; then
        srun_options="--nodes=1 --nodelist=${NODE_HOSTNAMES[1]} --export=ALL --exclusive"
        node_ip=$(srun $srun_options bash -c "getent hosts ${NODE_HOSTNAMES[0]} | awk '{print \$1}'")
    else
        node_ip=$(getent hosts "${hostnames[$node_id]}" | awk '{print $1}')
    fi

    workers_addr+=("${node_ip}:${APP_PORT}")
    # Write node_id and corresponding IP in the format "node_id,node_ip" to node_config_file
    if [[ -n "$node_ip" ]]; then
        echo "$node_id,$node_ip" >>"$NODE_CONFIG_FILE"
        echo "Mapped Node ID $node_id to IP $node_ip and written to $NODE_CONFIG_FILE"
    else
        echo "Could not resolve IP for hostname ${hostnames[$node_id]}"
    fi
done
workers_addr_str=$(
    IFS=','
    echo "${workers_addr[*]}"
)
echo ""
echo ""

# workers_addr=()
# # Iterate through node IDs and hostnames, writing IP addresses and node IDs to the config file
# for node_id in "${!NODE_HOSTNAMES[@]}"; do
#     # Map node_id to hostname
#     hostnames["$node_id"]="${NODE_HOSTNAMES[$node_id]}"

#     # Retrieve the IP address for the corresponding hostname
#     # node_ip=$(getent hosts "${hostnames[$node_id]}" | awk '{print $1}')
#     # Rui: Replacing the above getent hosts with grep /etc/hosts to get ip as getent hosts has issue with 1st host.
#     node_ip=$(grep -E "^.*[[:space:]]${host}[[:space:]]" /etc/hosts | awk '{print $1}' | head -n1)
#     workers_addr+=("${node_ip}:${APP_PORT}")

#     echo "hostname:${hostnames[$node_id]}, node_id:$node_id, node_ip:$node_ip"

#     # Write node_id and corresponding IP in the format "node_id,node_ip" to node.cfg
#     if [[ -n "$node_ip" ]]; then
#         echo "$node_id,$node_ip" >>"$NODE_CONFIG_FILE"
#         echo "Mapped Node ID $node_id to IP $node_ip and written to $NODE_CONFIG_FILE"
#     else
#         echo "Could not resolve IP for hostname ${hostnames[$node_id]}"
#     fi
# done
# workers_addr_str=$(
#     IFS=','
#     echo "${workers_addr[*]}"
# )

export WORKERS_ADDR=$workers_addr_str
export TOTAL_WORKERS=${#hostnames[@]}
export TOTAL_GPU_NUM=$(($TOTAL_WORKERS * $GPU_PER_NODE))
echo "-------------------------------"
echo "Listing configuration values..."
echo "-------------------------------"
echo "WORKERS_ADDR: $WORKERS_ADDR"
echo "TOTAL_WORKERS: $TOTAL_WORKERS"
echo "TOTAL_GPU_NUM: $TOTAL_GPU_NUM"
echo ""
echo ""

echo "-----------------------------------------"
echo "Set control and client node identifiers..."
echo "-----------------------------------------"
export CTRL_ID=$(grep -v '^#' "$NODE_CONFIG_FILE" | head -n 1 | cut -d',' -f1)
# export CTRL_ID=$(grep -v '^#' "$NODE_CONFIG_FILE" | sed -n '2p' | cut -d',' -f1)
export CTRL_HOSTNAME=${hostnames[$CTRL_ID]}
export CONTACT_IP=$(getent hosts $CTRL_HOSTNAME | awk '{print $1}')
echo "Control Node ID: $CTRL_ID, Hostname: $CTRL_HOSTNAME"
echo "Contact IP: $CONTACT_IP"
echo ""
echo ""

# Function to run a remote command on a node
run_remote_command() {
    node_id=$1
    node_hostname=$2
    remote_command=$3
    run_as_background=$4

    debug "run_remote_command():\n" \
        "- node_id=${node_id}\n" \
        "- node_hostname=${node_hostname}\n" \
        "- remote_command=${remote_command}\n" \
        "- run_as_background=${run_as_background}\n"

    srun_options="--nodes=1 --nodelist=${node_hostname} --export=ALL --verbose --exclusive"

    if [ "$run_as_background" = "1" ]; then
        srun $srun_options /bin/bash -c "$remote_command" &
        echo "srun $srun_options /bin/bash -c '$remote_command' &"
    else
        srun $srun_options /bin/bash -c "$remote_command"
    fi
}


# Loop over nodes and execute node_setup.sh
for node_id in "${!hostnames[@]}"; do
    node_hostname=${hostnames[$node_id]}
    node_dir="${BASE_DIR}/slurm_node${node_id}"
    echo "Executing node_setup.sh on node ${node_hostname} (ID: ${node_id})"

    # Export node-specific variables
    export NODE_ID="$node_id"
    export NODE_HOSTNAME="$node_hostname"
    export NODE_DIR="$node_dir"

    echo "Exported node-specific variables: NODE_ID=${NODE_ID}, NODE_HOSTNAME=${NODE_HOSTNAME}, NODE_DIR=${NODE_DIR}"

    # Use srun to execute node_setup.sh on the node, passing the environment variables
    remote_command="bash ${LOCAL_SLURM_DIR}/test/node_setup.sh"
    # remote_command="fi_info --hmem cuda"
    
    detach=1

    # Execute the command on the node with environment variables exported
    run_remote_command "$node_id" "$node_hostname" "$remote_command" "$detach"

    # remote_command=ifconfig
    # run_remote_command "$node_id" "$node_hostname" "$remote_command" "$detach"

    # remote_command=ibv_devices
    # run_remote_command "$node_id" "$node_hostname" "$remote_command" "$detach"
    
done

echo "-------------------------------------------------"
echo "Wait for all background processes to complete..."
echo "-------------------------------------------------"
# Wait for all background processes to complete
wait
echo "All background processes completed. automate_experiment.slurm exit successfully."