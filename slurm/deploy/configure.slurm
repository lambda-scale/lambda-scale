#!/bin/bash

# Set PYTHONPATH
export PYTHONPATH=$node_dir/src/fast-gpu-scaling/src:$node_dir/src/RDMC-GDR:$PYTHONPATH

# Update code repositories (if needed)
# cd "$remote_rdmc_dir" && git config pull.rebase true && git checkout . && git pull
# cd "/root/fast-gpu-scaling" && git config pull.rebase true && git checkout . && git pull
# cd "/root/gpu-fast-scaling" && git config pull.rebase true && git checkout . && git pull

# Run configuration script

echo -e "Executing $(basename "$0") with: \n" \
"- my_id=${my_id} \n" \
"- num_node=${total_workers} \n" \
"- workers_addr=${workers_addr} \n" \
"- view_port=${view_port} \n" \
"- p2p_port=${p2p_port} \n" \
"- log_level=${log_level} \n" \
"- log_file=${log_file} \n" \
"- contact_ip=${contact_ip}" \
"- node_dir=${node_dir} \n"  \
"- local_rdmc_dir=${local_rdmc_dir} \n"
echo ""

node_src_dir=$node_dir/src
rdmc_dir=$node_src_dir/RDMC-GDR
app1_dir=$node_src_dir/fast-gpu-scaling #yuchen implementation
app2_dir=$node_src_dir/gpu-fast-scaling #chaobo implementation

ip_addr=$(hostname -I | awk '{print $1}')


configure_rdmc() {
    
    cd $rdmc_dir 

    if ! $(cp $local_rdmc_dir/node.cfg .); then
        echo "Error. Failed to copy $local_rdmc_dir/node.cfg, check if exist."
        exit 1
    fi

    ######### derecho_node.cfg#########
    echo "Start configuring derecho_node.cfg"
    if ! $(cp src/conf/derecho_node-sample.cfg .); then
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
    sed -i "s/^default_log_level = .*/default_log_level = ${log_level}/" derecho_node-sample.cfg

    # p2p 
    sed -i "s/^my_id = .*/my_id = ${my_id}/" derecho_node-sample.cfg
    sed -i "s/^my_ip = .*/my_ip = ${ip_addr} /" derecho_node-sample.cfg
    sed -i "s/^contact_ip = .*/contact_ip = ${contact_ip}/" derecho_node-sample.cfg
    sed -i "s/^total_p2p_nodes = .*/total_p2p_nodes = ${num_node}/" derecho_node-sample.cfg
    
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
    sed -i "s/^default_log_level = .*/default_log_level = ${log_level}/" derecho-sample.cfg

    if ! $(mv derecho-sample.cfg derecho.cfg); then
        echo "Error. Failed to rename derecho-sample.cfg to derecho.cfg, check exist."
        exit 1
    fi
    echo ""
    echo ""
}

configure_app1() {
    ########## app1 configuration ##########
    cd ${app1_dir}/src/worker

    echo "Copying configure files {derecho_node.cfg, derecho.cfg} into ${app1_dir}/src/worker"
    
    if ! $(cp ${rdmc_dir}/derecho*.cfg .); then
        echo "Fail to copy ${rdmc_dir}/derecho*.cfg to ${app1_dir}/src/worker"
        exit 1
    fi
    
    if [[ ! -f "derecho_node.cfg" || ! -f "derecho.cfg" ]]; then
        echo "Fail configure application. derecho_node.cfg not found in ${app1_dir}"
        exit 1
    fi

    ## change log file prefix to fgs
    #sed -i "s/^log_file_prefix = .*/log_file_prefix = fgs/" derecho_node.cfg

        
    ## vars related to app already configured in configure_rdmc. just copy file into the app.
    echo "Finish configuring ${app1_dir}"
    echo ""
    echo ""
}

configure_app2() {
    ########## app2 configuration ##########
    cd $app2_dir

    echo ""

    local app2_working_dir=${app2_dir}/test_bed_local/serve/server
    local app2_controller_dir=${app2_dir}/test_bed_local/serve/controller
    cd ${app2_working_dir}

    echo "Copying configure files {derecho_node.cfg, derecho.cfg} into ${app2_working_dir} ..."
    
    if ! $(cp ${rdmc_dir}/derecho*.cfg .); then
        echo "Fail to copy ${rdmc_dir}/derecho*.cfg to ${app2_woriking_dir}"
        exit 1
    fi   
 
    if [[ ! -f "derecho_node.cfg" || ! -f "derecho.cfg" ]]; then
        echo "Fail to configure application. derecho_node.cfg or derecho.cfg not found in ${app2_working_dir}"
        exit 1
    fi


    echo "Copying node.cfg into ${app2_dir}/test_bed_rdma/serve ..." 
    if ! $(cp ${rdmc_dir}/node.cfg .); then
        echo "Fail to copy node.cfg into ${app2_dir}/test_bed_rdma/serve"
        exit 1
    fi

    cd ${app2_controller_dir}
    sed -i "s#^total_node_num = .*#total_node_num = ${num_node}#" evaluation.cfg
    sed -i "s#^total_gpu_num = .*#total_gpu_num = ${num_node}#" evaluation.cfg
    sed -i "s#^root_path = .*#root_path = ${node_src_dir}#" evaluation.cfg
    sed -i "s#^is_rdma = .*#is_rdma = True#" evaluation.cfg

    ## vars related to app already configured in configure_rdmc. just copy file into the app.
    echo "Finishi configuring app2."
    echo ""
    echo ""
}


configure_node(){
    echo "=> Configuring RDMC-GDR"
    configure_rdmc

    # echo "=> Configuring FGS (yuchen) application"
    # configure_app1

    echo "=> Configuring GFS (chaobo) application"
    configure_app2

}

configure_node

