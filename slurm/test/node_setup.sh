#!/bin/bash

# node_setup.sh

# debug the env variables passed to the script
debug() {
    echo "DEBUG: $@"
}


# module load gcc/11.4.0
# module load cuda/12.4.1

module load gcc/13.1.0
module load nvhpc-hpcx-cuda12/23.11

node_dir=$NODE_DIR
node_id=$NODE_ID
node_hostname=$NODE_HOSTNAME
workers_addr=$WORKERS_ADDR
total_workers=$TOTAL_WORKERS
node_src_dir=$node_dir/src
rdmc_dir=$node_src_dir/RDMC-GDR
fgs_dir=$node_src_dir/fast-gpu-scaling #yuchen implementation
gfs_dir=$node_src_dir/gpu-fast-scaling #chaobo implementation
gfs_script_dir=$gfs_dir/test_bed_local/scripts
log_level=$LOG_LEVEL

# define command to activate conda
conda_activate="source $node_dir/utils/miniconda3/bin/activate base"

ip_addr=$(hostname -I | awk '{print $1}')

# Function to set up the environment
setup_environment() {
    echo "Setting up environment on node ${NODE_HOSTNAME} (ID: ${NODE_ID})"
    # Export environment variables
    export PATH="$node_dir/utils/bin:/usr/local/bin:$PATH"
    export PYTHONPATH="$node_dir/src/gpu-fast-scaling/src:$node_dir/src/RDMC-GDR"
    export LD_LIBRARY_PATH="/usr/local/lib:/usr/local/cuda/targets/x86_64-linux/lib:$LD_LIBRARY_PATH"
    $conda_activate
    echo "Python version: $(which python)"
}

configure_rdmc() {
    
    cd $rdmc_dir 

    if ! $(cp $LOCAL_RDMC_DIR/node.cfg .); then
        echo "Error. Failed to copy $local_rdmc_dir/node.cfg, check if exist."
        exit 1
    fi

    ######### derecho_node.cfg#########
    echo "Start configuring derecho_node.cfg"
    if ! $(cp src/conf/derecho_node-sample.cfg -f .); then
        echo "Error. Failed to copy src/conf/derecho_node-sample.cfg, check if exist."
        exit 1
    fi

    ip_addr=$(hostname -I | awk '{print $1}')
    ib_device=$(ibv_devinfo | grep hca_id: | awk 'NR==1 {print $2}')
    cuda_device_id=0
    # automation
    sed -i "s#^node_dir = .*#node_dir = ${node_dir}#" derecho_node-sample.cfg

    # application
    sed -i "s/^device_id = .*/device_id = ${cuda_device_id}/" derecho_node-sample.cfg
    echo "workers_addr=${workers_addr}" 
    sed -i "s/^workers_addr = .*/workers_addr = ${workers_addr}/" derecho_node-sample.cfg
    sed -i "s/^default_log_level = .*/default_log_level = ${LOG_LEVEL}/" derecho_node-sample.cfg

    # p2p 
    sed -i "s/^my_id = .*/my_id = ${node_id}/" derecho_node-sample.cfg
    sed -i "s/^my_ip = .*/my_ip = ${ip_addr} /" derecho_node-sample.cfg
    sed -i "s/^contact_ip = .*/contact_ip = ${contact_ip}/" derecho_node-sample.cfg
    sed -i "s/^total_p2p_nodes = .*/total_p2p_nodes = ${total_workers}/" derecho_node-sample.cfg
    
    #rdma
    sed -i "s/^provider = .*/provider = verbs/" derecho_node-sample.cfg
    sed -i "s/^domain = .*/domain = ${ib_device}/" derecho_node-sample.cfg
    sed -i "s/^tx_depth = .*/tx_depth = 4096/" derecho_node-sample.cfg
    sed -i "s/^rx_depth = .*/rx_depth = 4096/" derecho_node-sample.cfg
    
    if ! $(mv derecho_node-sample.cfg derecho_node.cfg); then
        echo "Error. Failed to rename derecho_node-sample.cfg to derecho_node.cfg, check exist."
        exit 1
    fi

    echo "Finish configuring derecho_node.cfg."

    ######### derecho.cfg#########
    echo "Start confguring logging vars in derecho.cfg"
    if ! $(cp src/conf/derecho-sample.cfg .); then
        echo "Error. Failed to copy src/conf/derecho-sample.cfg, check if exist."
        exit 1
    fi

    #log vars for RDMC-GDR

    sed -i "s/^default_log_name = .*/default_log_name = derecho_debug_${my_id}/" derecho-sample.cfg
    sed -i "s/^default_log_level = .*/default_log_level = ${LOG_LEVEL}/" derecho-sample.cfg

    if ! $(mv derecho-sample.cfg derecho.cfg); then
        echo "Error. Failed to rename derecho-sample.cfg to derecho.cfg, check exist."
        exit 1
    fi
    echo ""
    echo ""
}


configure_fgs() {
    ########## fgs configuration ##########
    cd $fgs_dir

    echo ""

    local fgs_server_dir=${fgs_dir}/test_bed_local/serve/server
    local fgs_controller_dir=${fgs_dir}/test_bed_local/serve/controller
    cd ${fgs_server_dir}

    echo "Copying configure files {derecho_node.cfg, derecho.cfg} into ${fgs_working_dir} ..."
    
    if ! $(cp ${rdmc_dir}/derecho*.cfg .); then
        echo "Fail to copy ${rdmc_dir}/derecho*.cfg to ${fgs_woriking_dir}"
        exit 1
    fi   
 
    if [[ ! -f "derecho_node.cfg" || ! -f "derecho.cfg" ]]; then
        echo "Fail to configure application. derecho_node.cfg or derecho.cfg not found in ${fgs_working_dir}"
        exit 1
    fi

}

configure_gfs() {
    ########## gfs configuration ##########
    cd $gfs_dir

    echo ""

    local gfs_server_dir=${gfs_dir}/test_bed_local/serve/server
    local gfs_controller_dir=${gfs_dir}/test_bed_local/serve/controller
    cd ${gfs_server_dir}

    echo "Copying configure files {derecho_node.cfg, derecho.cfg} into ${gfs_working_dir} ..."
    
    if ! $(cp ${rdmc_dir}/derecho*.cfg .); then
        echo "Fail to copy ${rdmc_dir}/derecho*.cfg to ${gfs_woriking_dir}"
        exit 1
    fi   
 
    if [[ ! -f "derecho_node.cfg" || ! -f "derecho.cfg" ]]; then
        echo "Fail to configure application. derecho_node.cfg or derecho.cfg not found in ${gfs_working_dir}"
        exit 1
    fi


    echo "Copying node.cfg into ${gfs_dir}/test_bed_rdma/serve ..." 
    if ! $(cp ${rdmc_dir}/node.cfg .); then
        echo "Fail to copy node.cfg into ${gfs_dir}/test_bed_rdma/serve"
        exit 1
    fi

    cd ${gfs_controller_dir}
    sed -i "s#^total_node_num = .*#total_node_num = ${total_workers}#" evaluation.cfg
    sed -i "s#^total_gpu_num = .*#total_gpu_num = ${GPU_PER_NODE}#" evaluation.cfg
    sed -i "s#^root_path = .*#root_path = ${node_src_dir}#" evaluation.cfg
    sed -i "s#^is_rdma = .*#is_rdma = True#" evaluation.cfg
    sed -i "s#model_name = .*#model_name = ${MODEL_NAME}#" evaluation.cfg

    ## vars related to app already configured in configure_rdmc. just copy file into the app.
    echo "Finishi configuring gfs."
    echo ""
    echo ""
}



configure_cluster() {
    echo "Configuring cluster on node ${NODE_HOSTNAME} (ID: ${NODE_ID})"
    configure_rdmc
    # check the target app is fgs or gfs
    if [[ "$TARGET_APP" == "fgs" ]]; then
        configure_fgs
    elif [[ "$TARGET_APP" == "gfs" ]]; then
        configure_gfs
    else
        echo "Error. Invalid target app: $TARGET_APP"
        exit 1
    fi

    echo "Node ${NODE_HOSTNAME} (ID: ${NODE_ID}) configured successfully."
    echo ""
}

# Function to clean up old processes and ports
cleanup_cluster_port() {
    ports=($APP_PORT $P2P_PORT $VIEW_PORT 2000 5000 5001 5002 5003 5004 7000 8000 8050 9000 7048 7049 7017 7016 7064 29500)
    echo "Cleaning up ports on ${NODE_HOSTNAME} (ID: ${NODE_ID})"
    if ! command -v lsof &> /dev/null; then
        echo "Error: lsof command not found. Please install lsof first."
        return 1
    fi
    for port in "${ports[@]}"; do
        pid=$(lsof -i:$port -t 2>/dev/null)
        echo "Checking process on port $port"
        echo "PID: $pid"
        if [ ! -z "$pid" ]; then
            if kill -9 "$pid" &> /dev/null; then
                echo "Successfully killed process $pid on port $port."
            else
                echo "Failed to kill process $pid on port $port. Check permissions."
            fi
        else
            echo "No process found on port $port"
        fi
    done
}


start_workers() {
    echo "Starting workers on node ${NODE_HOSTNAME} (ID: ${NODE_ID})"
    # Change directory to the script directory
    cd $gfs_script_dir
    # cd $gfs_dir/test_bed_local/serve/server
    nohup bash $WORKER_FILE &> "worker_${NODE_ID}.log" &
    # bash $WORKER_FILE &> "worker_${NODE_ID}.log"
}

# Function to start ctrl
start_ctrl() {
    echo "Starting controller on node ${NODE_HOSTNAME} (ID: ${NODE_ID})"
    # Change directory to the script directory
    cd $gfs_script_dir
    nohup bash $CTRL_FILE &> "ctrl_${NODE_ID}.log" &
    # bash $CTRL_FILE &> "ctrl_${NODE_ID}.log" &
}

# Function to start client
start_client() {
    echo "Starting client on node ${NODE_HOSTNAME} (ID: ${NODE_ID})"
    # Change directory to the script directory
    cd $gfs_script_dir
    bash $START_FILE &> "start_${NODE_ID}.log" &
    # nohup bash $START_FILE &> "start_${NODE_ID}.log" &
    sleep $WORKER_COMM_ESTABLISH_WAIT_TIME

    # nohup bash $CLIENT_FILE &> "client_${NODE_ID}.log" &
    bash $CLIENT_FILE &> "client_${NODE_ID}.log" &
}

start_gfs() {
    echo "Starting GFS on node ${NODE_HOSTNAME} (ID: ${NODE_ID})"
    echo "Starting application with total workers:${TOTAL_WORKERS}, inter wait time:${WORKER_INTER_WAIT_TIME}, worker establishment wait time:${WORKER_COMM_ESTABLISH_WAIT_TIME}, controller wait time:${CONTROLLER_WAIT_TIME}, client wait time:${CLIENT_WAIT_TIME}"
    # Start workers in the background
    start_workers

    echo "Waiting for workers to establish communication..."
    # sleep $WORKER_COMM_ESTABLISH_WAIT_TIME

    if [[ $NODE_ID == $CTRL_ID ]]; then
        echo "Node ${NODE_HOSTNAME} (ID: ${NODE_ID}) is not a worker."
        echo "Node ${NODE_HOSTNAME} (ID: ${NODE_ID}) is controller."

        # Start controller in the background
        echo "Starting controller on node ${NODE_HOSTNAME} (ID: ${NODE_ID})"
        start_ctrl

        echo "Waiting for controller to establish communication..."
        sleep $CONTROLLER_WAIT_TIME

        # Start client in the background
        start_client
        
        echo "Waiting for client to establish communication..."
        sleep $CLIENT_WAIT_TIME
    else
        echo "Node ${NODE_HOSTNAME} (ID: ${NODE_ID}) is worker."
    fi


}

# Function to collect logs
collect_logs() {
    echo "Collecting logs on node ${NODE_HOSTNAME} (ID: ${NODE_ID})"
}


# Main process
echo "-------------------------------------------"
echo "Configure cluster for application startup..."
echo "-------------------------------------------"
configure_cluster
echo "-------------------------------------------"
echo "Setup application for application startup..."
echo "-------------------------------------------"
setup_environment
fi_info --hmem cuda
# echo "-----------------------------------------"
# echo "Start cleanup process on cluster ports..."
# echo "-----------------------------------------"
# cleanup_cluster_port
# echo "-----------------------------------------"
# echo "Start application on the cluster nodes..."
# echo "-----------------------------------------"
# start_gfs
# echo "-----------------------------------------"
# echo "Collect logs from the cluster nodes..."
# echo "-----------------------------------------"
# collect_logs

# echo "Node ${NODE_HOSTNAME} (ID: ${NODE_ID}) setup completed successfully."
echo "-------------------------------------------------"
echo "Wait for all background processes to complete..."
echo "-------------------------------------------------"
wait

echo "All background processes completed. Node ${NODE_HOSTNAME} (ID: ${NODE_ID}) setup completed successfully."
