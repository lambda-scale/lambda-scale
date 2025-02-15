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
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $arg"
            usage
            ;;
    esac
done

BASE_INSTALL_PREFIX="${INSTALL_PREFIX}"

INSTALL_PREFIX="${INSTALL_PREFIX}"

echo "Using INSTALL_PREFIX=${INSTALL_PREFIX}"

# Temporary directory for the installation process
export TMPDIR="${INSTALL_PREFIX}/tmp/mutils"
mkdir -p "${TMPDIR}"
WORKPATH="${TMPDIR}"

# Set library paths based on the installation prefix
export LIBRARY_PATH="${INSTALL_PREFIX}/lib:${INSTALL_PREFIX}/lib64"
export LD_LIBRARY_PATH="${LIBRARY_PATH}"

# Navigate to the temporary working directory
cd "${WORKPATH}"

# Define the target directory for cloning
TARGET_DIR="mutils"

# Check if the target directory exists
if [ -d "${TARGET_DIR}" ]; then
    echo "Directory ${TARGET_DIR} exists. Removing it to overwrite."
    rm -rf "${TARGET_DIR}"
fi

# Clone the mutils repository
git clone https://github.com/mpmilano/mutils.git "${TARGET_DIR}"
cd "${TARGET_DIR}"

# Checkout a specific commit to ensure consistency
git checkout 81e16d898d1f58a2ce4e58253d4b397783866357

# Create and navigate to the build directory
mkdir build
cd build

# Configure the build with CMake
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="${BASE_INSTALL_PREFIX}" ..

# Install the compiled binaries
if make install; then
    rm -rf "${WORKPATH}"
    echo "mutils has been successfully installed to ${INSTALL_PREFIX}"
else
    echo "Installation failed. Please check the logs for details."
    exit 1
fi