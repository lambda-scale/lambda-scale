#!/bin/bash
#SBATCH --job-name=perftest
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=4
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=gpu

echo "--------------------------------------------------"
echo "perftest.slurm is running on hostname:$(hostname)"
echo "--------------------------------------------------"

# Expand SLURM_NODELIST into individual nodes
expanded_nodes=$(scontrol show hostname "${SLURM_NODELIST}")
expanded_nodes_array=(${expanded_nodes})

# Check if we have at least two nodes
if [ "${#expanded_nodes_array[@]}" -lt 2 ]; then
    echo "Error: At least two nodes are required to run ib_read_bw."
    exit 1
fi

# Assign the first node for the server and the second node for the client
server_node="${expanded_nodes_array[0]}"
client_node="${expanded_nodes_array[1]}"

echo "Server Node: $server_node"
echo "Client Node: $client_node"

# Extract the server's InfiniBand IP
server_ip=$(srun --nodelist="$client_node" getent hosts "${server_node}" | awk '!/^(127\.|::1)/ {print $1}' | head -n 1)
client_ip=$(srun --nodelist="$server_node" getent hosts "${client_node}" | awk '!/^(127\.|::1)/ {print $1}' | head -n 1)

if [ -z "$server_ip" ]; then
    echo "Error: Unable to fetch InfiniBand IP for the server node ($server_node)."
    exit 1
fi

echo "Server IP: $server_ip"
echo "Client IP: $client_ip"


server_ib_device=$(srun --nodelist="$server_node" ibv_devinfo | grep hca_id: | awk 'NR==1 {print $2}')
client_ib_device=$(srun --nodelist="$client_node" ibv_devinfo | grep hca_id: | awk 'NR==1 {print $2}')

if [ -z "$server_ib_device" ] || [ -z "$client_ib_device" ]; then
    echo "Error: Unable to fetch IB device on one or both nodes."
    exit 1
fi

qp_num=4
output_dir=/scratch/qgh4hm/chaobo/slurm_nodes/workspace/RDMC-GDR/slurm/baseline/logs/12_2

echo "Server IB Device: $server_ib_device"
echo "Client IB Device: $client_ib_device"
echo "Queue pair number: $qp_num"
#launch server
echo "Launching ib_read_bw server on $server_node with IB Device: $server_ib_device"
# srun --verbose --nodelist="$server_node" --exclusive --output=server_output.log ib_read_bw &
srun --nodelist="$server_node" \
    --exclusive --ntasks=1 \
    --nodes=1 --output=${output_dir}/server_output.log \
    LD_PRELOAD=/scratch/qgh4hm/3rd-lib/lib/libpci.so.3.13.0 /scratch/qgh4hm/3rd-lib/bin/ib_read_bw \
    --connection=RC --qp=$qp_num --report_gbits --use_cuda=0 & 

server_pid=$!
# Allow the server to start
sleep 5

#launch client
echo "Launching ib_read_bw client on $client_node connecting to server $server_ip with IB Device: $client_ib_device"
# # Launch ib_read_bw client on the second node
# srun --verbose --nodelist="$client_node" --exclusive --output=client_output.log ib_read_bw "$server_ip" -d "$client_ib_device"
srun --nodelist="$client_node" \
    --exclusive --ntasks=1 \
    --nodes=1 --output=${output_dir}/client_output.log \
    LD_PRELOAD=/scratch/qgh4hm/3rd-lib/lib/libpci.so.3.13.0 /scratch/qgh4hm/3rd-lib/bin/ib_read_bw \
    "$server_ip"  --connection=RC --qp=$qp_num --report_gbits --use_cuda=0

# Wait for all background processes to complete
wait $server_pid
echo "ib_read_bw test completed successfully."