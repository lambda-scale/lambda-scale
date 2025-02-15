#!/bin/bash
set -eu

# Default installation prefix
INSTALL_PREFIX="/usr/local"

# Function to display usage information
usage() {
    echo "Usage: $0 [--prefix=<installation_prefix>]"
    echo "Options:"
    echo "  --prefix=<installation_prefix>   Set the installation prefix (default: /usr/local)"
    echo "  -h, --help                      Display this help message"
    exit 1
}

# Parse command-line arguments
for arg in "$@"; do
    case $arg in
        --prefix=*)
            INSTALL_PREFIX="${arg#*=}"
            shift
            ;;
        --local-work-dir=*)
            LOCAL_WORK_DIR="${arg#*=}"
            shift
            ;;
        --cuda-home-dir=*)
            CUDA_HOME_DIR="${arg#*=}"
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $arg"
            usage
            ;;
    esac
done

INSTALL_PREFIX="${INSTALL_PREFIX}"

echo "Using INSTALL_PREFIX=${INSTALL_PREFIX}"
echo "Using LOCAL_WORK_DIR=${LOCAL_WORK_DIR}"


# Temporary directory for the installation process
export TMPDIR="${INSTALL_PREFIX}/tmp/libfabric"
mkdir -p "${TMPDIR}"
WORKPATH="${TMPDIR}"

# Set library paths based on the installation prefix
export LIBRARY_PATH="${INSTALL_PREFIX}/lib:${INSTALL_PREFIX}/lib64"
export LD_LIBRARY_PATH="${LIBRARY_PATH}"

# Navigate to the temporary working directory
cd "${WORKPATH}"

# Define the target directory for cloning
TARGET_DIR="libfabric"

# Check if the target directory exists
if [ -d "${TARGET_DIR}" ]; then
    echo "Directory ${TARGET_DIR} exists. Removing it to overwrite."
    rm -rf "${TARGET_DIR}"
fi

# Clone the libfabric repository
git clone --branch v1.12.1 https://github.com/ofiwg/libfabric.git "${TARGET_DIR}"
cd "${TARGET_DIR}"

# Apply the necessary patch
PATCH_PATH="${LOCAL_WORK_DIR}/RDMC-GDR/slurm/prerequisites/libfabric.patch"
git apply "${PATCH_PATH}"

# Prepare the build system
libtoolize
./autogen.sh

# Configure the build with specified options
./configure --prefix="${INSTALL_PREFIX}" --disable-memhooks-monitor --disable-spinlock --with-cuda="${CUDA_HOME_DIR}"

# Compile the project using all available CPU cores
make -j

# Install the compiled binaries
if make install; then
    rm -rf "${WORKPATH}"
    echo "libfabric has been successfully installed to ${INSTALL_PREFIX}"
else
    echo "Installation failed. Please check the logs for details."
    exit 1
fi