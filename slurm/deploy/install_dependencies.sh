#!/bin/bash

# ------------------------------------------------------------------
# Script Name: install_dependencies_slurm.sh
# Description: This script installs MLNX_OFED_LINUX, pciutils,
#              perftest with GPU Direct support, and RDMC-GDR
#              dependencies across all allocated SLURM nodes.
# ------------------------------------------------------------------

# Enable or disable debug output
DEBUG=true # Set to "true" to enable debug output, "false" to disable
module load cuda12.2
module load cuda12.2/toolkit/12.2.2


# Function to print debug messages
debug() {
    if [ "$DEBUG" = "true" ]; then
        echo -e "$@" >&2
    fi
}

# Base directory for node-specific installations
NODE_BASE_DIR=$(sed -n 's/^node_base_dir[[:space:]]*=[[:space:]]*\(.*\)/\1/p' derecho_node.cfg | sed 's/[[:space:]]*$//')
LOCAL_WORK_DIR="${NODE_BASE_DIR}/workspace"



# TODO
# Function to install pciutils from source
# install_pciutils() {
#     local node_install_dir=$1
#     echo "----------------------------------"
#     echo "Installing pciutils from source on $node_install_dir..."
#     echo "----------------------------------"
#     cd "$LOCAL_WORK_DIR/pciutils" || { echo "pciutils directory not found in $LOCAL_WORK_DIR"; exit 1; }

#     # Configure with installation prefix
#     ./configure --prefix="$node_install_dir/utils"
#     make
#     make install
# }

# TODO
# Function to install perftest with GPU Direct support
# install_perftest() {
#     local node_install_dir=$1
#     echo "----------------------------------------------"
#     echo "Installing perftest with GPU Direct support on $node_install_dir..."
#     echo "----------------------------------------------"
#     cd "$LOCAL_WORK_DIR/perftest" || { echo "perftest directory not found in $LOCAL_WORK_DIR"; exit 1; }

#     ./autogen.sh
#     ./configure --prefix="$node_install_dir/utils" CUDA_H_PATH=/usr/local/cuda/include/cuda.h
#     make -j
#     make install
# }

# Function to install RDMC-GDR dependencies
install_rdmc_gdr_dependencies() {
    local node_install_dir=$1
    echo "-----------------------------------"
    echo "Installing RDMC-GDR dependencies on $node_install_dir..."
    echo "-----------------------------------"
    cd "$LOCAL_WORK_DIR/RDMC-GDR" || { echo "RDMC-GDR directory not found in $LOCAL_WORK_DIR"; exit 1; }

    # Ensure the installation scripts are executable
    chmod +x ./slurm/prerequisites/*.sh

    # Run installation scripts with specified installation prefix
    ./slurm/prerequisites/install-json.sh --prefix="$node_install_dir"
    ./slurm/prerequisites/install-mutils.sh --prefix="$node_install_dir"
    ./slurm/prerequisites/install-mutils-containers.sh --prefix="$node_install_dir"
    ./slurm/prerequisites/install-libfabric.sh --prefix="$node_install_dir" --local-work-dir="$LOCAL_WORK_DIR" --cuda-home-dir="$CUDA_HOME"
    ./slurm/prerequisites/install-spdlog.sh --prefix="$node_install_dir"
}

# Function to build RDMC-GDR library
build_rdmc_gdr() {
    local node_dir=$1
    local node_install_dir=$2
    echo "---------------------"
    echo "Building RDMC-GDR lib on $node_dir..."
    echo "---------------------"
    mkdir -p "$node_dir/src/"
    cd "$node_dir/src" || { echo "Failed to change directory to $node_dir/src"; exit 1; }
    local target_dir="RDMC-GDR"
    if [ -d "$target_dir" ]; then
        echo "Directory $target_dir already exists. Removing it to re-clone."
        rm -rf "$target_dir"
    fi

    git clone --branch ibverbs-wip git@github.com:ruiyang00/RDMC-GDR.git
    # cd RDMC-GDR
    # git reset f3d662544723bf8ed929f182186a229c5f77fc7f --hard
    cd "$node_dir/src/RDMC-GDR" || { echo "RDMC-GDR directory not found in $node_dir"; exit 1; }

    # Ensure the build script is executable
    chmod +x ./build.sh

    # Build RDMC-GDR in Debug mode with installation prefix
    # DERECHO_INSTALL_PREFIX="$node_install_dir" ./build.sh Debug USE_VERBS_API
}

install_conda() {
    local node_install_dir=$1
    echo "-----------------------------------"
    echo "Installing conda on $node_install_dir..."
    echo "-----------------------------------"
    cd "$node_install_dir" || { echo "Failed to change directory to $LOCAL_WORK_DIR"; exit 1; }
    mkdir -p $node_install_dir/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $node_install_dir/miniconda3/miniconda.sh
    bash $node_install_dir/miniconda3/miniconda.sh -b -u -p $node_install_dir/miniconda3
    rm $node_install_dir/miniconda3/miniconda.sh

    source $node_install_dir/miniconda3/bin/activate base
}

build_gfs() {
    local node_dir=$1
    local node_install_dir=$2
    echo "---------------------"
    echo "Building GFS lib on $node_dir..."
    echo "---------------------"

    mkdir -p "$node_dir/src/"
    cd "$node_dir/src" || { echo "Failed to change directory to $node_dir/src"; exit 1; }
    # local target_dir="gpu-fast-scaling"
    # if [ -d "$target_dir" ]; then
    #     echo "Directory $target_dir already exists. Removing it to re-clone."
    #     rm -rf "$target_dir"
    # fi
    export GIT_SSH_COMMAND="ssh -i $LOCAL_WORK_DIR/RDMC-GDR/dockerfiles/id_rsa"
    # git clone --branch test_bed_local https://github.com/MincYu/gpu-fast-scaling.git
    # git clone --branch test_bed_local git@github.com:MincYu/gpu-fast-scaling.git
    cd "$node_dir/src/gpu-fast-scaling" || { echo "GFS directory not found in $node_dir"; exit 1; }
    
    #
    # #Rui TODO: uncomment below lines back
    # git add .
    # git stash
    # git checkout test_bed_local
    # git fetch origin
    # git reset --hard origin/test_bed_local
    # git pull

    echo "------------------------------"
    echo "installing python3 packages..."
    echo "------------------------------"

    echo "source ${node_install_dir}/miniconda3/bin/activate"
    source ${node_install_dir}/miniconda3/bin/activate base

    # pip3 install torch==2.3.0
    # pip3 install --no-binary=:all: Pillow==9.2.0 \
    #     --global-option=build_ext \
    #     --global-option="-I/scratch/infattllm/fgscaling/3rd-lib/softwares/include" \
    #     --global-option="-L/scratch/infattllm/fgscaling/3rd-lib/softwares/lib"

    # pip3 install -r requirements.txt

    # if [[ $? -ne 0 ]]; then
    #     echo "pip3 install failed, exiting."
    #     exit 1
    # fi

    module load gcc/13.1.0

    # Ensure the build script is executable
    echo "change directory to $node_dir/src/gpu-fast-scaling/test_bed_local/scripts"
    cd $node_dir/src/gpu-fast-scaling/test_bed_local/scripts

    # export HF_TOKEN="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"

    # huggingface-cli login --token $HF_TOKEN

    echo "prepare to execute compile.sh"
    if ! bash compile.sh; then
        echo "Fail to execute compile.sh ..."
        exit 1
    fi

    # echo "prepare to execute pre_load_model.sh ..."
    # if ! bash pre_load_model.sh; then
    #     echo "Fail to execute pre_load_model.sh ..."
    #     exit 1
    # fi

}

# Aggregate all installation steps
install_all_dependencies() {
    local node_install_dir=$1
    install_rdmc_gdr_dependencies "$node_install_dir"
}

# Fetch the list of node hostnames allocated by SLURM
node_hostnames=($(scontrol show hostname "$SLURM_NODELIST"))
debug "Allocated nodes: ${node_hostnames[@]}"

# Iterate through each node and execute installation commands
# for index in "0 1 2 3 4 5 6 7 8 9 10 11${!node_hostnames[@]}"; do
# for index in 0 1 2 3 4 5 6 7 8 9 10 11; do
for index in 0; do
    node="${node_hostnames[$index]}"
    seq=$((index)) # Sequence number starts at 0
    node_dir="${NODE_BASE_DIR}/slurm_node${seq}"
    node_install_dir="${NODE_BASE_DIR}/slurm_node${seq}/utils"

    echo "--------------------------------------------------------------"
    echo "Starting installation on node: $node"
    echo "--------------------------------------------------------------"

    # Define node-specific installation directory
    echo "Node-specific installation directory: $node_install_dir"
    
    # Create the node-specific installation directory if it doesn't exist
    mkdir -p "$node_install_dir"

    # Directly call the installation function without using srun
    # install_all_dependencies $node_install_dir
    # install_conda $node_install_dir
    build_gfs $node_dir $node_install_dir

    # build_rdmc_gdr $node_dir $node_install_dir

done

# echo "--------------------------------------------------------------"
# echo "Installation of all dependencies completed on all nodes."
# echo "--------------------------------------------------------------"
