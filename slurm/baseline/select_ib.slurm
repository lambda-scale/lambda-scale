#!/bin/bash
#SBATCH --job-name=find_best_nic_per_gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --partition=normal
#SBATCH --account=infattllm 

echo "DEBUG: Starting script..."

# Parse CUDA_VISIBLE_DEVICES
IFS=',' read -r -a GPU_INDICES <<< "${CUDA_VISIBLE_DEVICES}"
echo "DEBUG: GPU_INDICES: ${GPU_INDICES[@]}"

RAW_TOPO_OUTPUT=$(nvidia-smi topo -m)
if [[ $? -ne 0 ]]; then
    echo "ERROR: Failed to run 'nvidia-smi topo -m'."
    exit 1
fi

# Remove ANSI escape sequences
TOPO_OUTPUT=$(echo "$RAW_TOPO_OUTPUT" | sed 's/\x1B\[[0-9;]*m//g')

echo "DEBUG: Full topology output:"
echo "$TOPO_OUTPUT"
echo "DEBUG: End of topology output."

# Extract the header line
HEADER_LINE=$(echo "$TOPO_OUTPUT" | head -n1)
echo "DEBUG: Original HEADER_LINE: '$HEADER_LINE'"

# Replace multi-word column headers in HEADER_LINE
HEADER_LINE=$(echo "$HEADER_LINE" \
    | sed 's/CPU Affinity/CPU_Affinity/g' \
    | sed 's/NUMA Affinity/NUMA_Affinity/g' \
    | sed 's/GPU NUMA ID/GPU_NUMA_ID/g')

# Normalize whitespace in HEADER_LINE
HEADER_LINE=$(echo "$HEADER_LINE" | tr -s '[:space:]' ' ' | sed 's/^ *//;s/ *$//')

echo "DEBUG: CLEAN_HEADER after replacements and trimming: '$HEADER_LINE'"

IFS=' ' read -r -a ALL_COLUMNS <<< "$HEADER_LINE"
echo "DEBUG: ALL_COLUMNS length: ${#ALL_COLUMNS[@]}"
for ((i=0; i<${#ALL_COLUMNS[@]}; i++)); do
    echo "DEBUG: ALL_COLUMNS[$i] = ${ALL_COLUMNS[$i]}"
done

# Identify columns that are GPUs or NICs
DESIRED_COLS=()
for ((i=0; i<${#ALL_COLUMNS[@]}; i++)); do
    if [[ "${ALL_COLUMNS[i]}" =~ ^GPU[0-9]+$ || "${ALL_COLUMNS[i]}" =~ ^NIC[0-9]+$ ]]; then
        DESIRED_COLS+=($i)
        echo "DEBUG: Keeping column $i (${ALL_COLUMNS[i]})"
    fi
done

echo "DEBUG: DESIRED_COLS: ${DESIRED_COLS[@]}"

if [[ ${#DESIRED_COLS[@]} -eq 0 ]]; then
    echo "ERROR: No GPU or NIC columns found in the header."
    exit 1
fi

FILTERED_COLUMNS=()
for idx in "${DESIRED_COLS[@]}"; do
    FILTERED_COLUMNS+=("${ALL_COLUMNS[$idx]}")
done

FILTERED_HEADER="$(printf "%s " "${FILTERED_COLUMNS[@]}" | sed 's/ *$//')"
echo "DEBUG: FILTERED_HEADER: '$FILTERED_HEADER'"

# Function to replace multi-word column names in any line
function normalize_line {
    local line="$1"
    line=$(echo "$line" \
        | sed 's/CPU Affinity/CPU_Affinity/g' \
        | sed 's/NUMA Affinity/NUMA_Affinity/g' \
        | sed 's/GPU NUMA ID/GPU_NUMA_ID/g')
    line=$(echo "$line" | tr -s '[:space:]' ' ' | sed 's/^ *//;s/ *$//')
    echo "$line"
}

FILTERED_TOPO=""
while IFS= read -r line; do
    NORM_LINE=$(normalize_line "$line")
    echo "DEBUG: Processing line: '$NORM_LINE'"

    if [[ "$NORM_LINE" == "$HEADER_LINE" ]]; then
        # This is the header line
        LINE_ARRAY=($NORM_LINE)
        FILTERED_LINE="#"
        for col_idx in "${DESIRED_COLS[@]}"; do
            [ $col_idx -lt ${#LINE_ARRAY[@]} ] && FILTERED_LINE+="${LINE_ARRAY[$col_idx]} " || FILTERED_LINE+="N/A "
        done
        FILTERED_LINE=$(echo "$FILTERED_LINE" | sed 's/ *$//')
        echo "DEBUG: Filtered header line: '$FILTERED_LINE'"
        FILTERED_TOPO+="$FILTERED_LINE\n"
        continue
    fi

    # Device line check
    if [[ "$NORM_LINE" =~ ^(GPU[0-9]+|NIC[0-9]+)([[:space:]]|$) ]]; then
        DEVICE_NAME=${BASH_REMATCH[1]}
        echo "DEBUG: This line appears to be a device line for $DEVICE_NAME."
        LINE_ARRAY=($NORM_LINE)
        echo "DEBUG: LINE_ARRAY length: ${#LINE_ARRAY[@]}"
        for ((ai=0; ai<${#LINE_ARRAY[@]}; ai++)); do
            echo "DEBUG: LINE_ARRAY[$ai] = ${LINE_ARRAY[$ai]}"
        done

        # Construct filtered line:
        # First add the device name, then the desired columns
        FILTERED_LINE="$DEVICE_NAME"
        for col_idx in "${DESIRED_COLS[@]}"; do
            adjusted_idx=$((col_idx + 1))
            echo "DEBUG: For column ${ALL_COLUMNS[$col_idx]} (col_idx=$col_idx), adjusted_idx=$adjusted_idx"
            if (( adjusted_idx < ${#LINE_ARRAY[@]} )); then
                FILTERED_LINE+=" ${LINE_ARRAY[$adjusted_idx]}"
            else
                FILTERED_LINE+=" N/A"
            fi
        done
        echo "DEBUG: FILTERED_LINE for device $DEVICE_NAME: '$FILTERED_LINE'"
        FILTERED_TOPO+="$FILTERED_LINE\n"
    else
        # Non-device lines remain as is
        FILTERED_TOPO+="$line\n"
    fi
done <<< "$TOPO_OUTPUT"

echo "DEBUG: FILTERED_TOPO:"
echo -e "$FILTERED_TOPO"
echo "DEBUG: End of FILTERED_TOPO."

# Extract NIC Legend lines
LEGEND_LINES=$(echo -e "$FILTERED_TOPO" | sed -n '/NIC Legend:/,$p' | grep -E "NIC[0-9]:")
echo "DEBUG: NIC Legend lines:"
echo "$LEGEND_LINES"

declare -A NIC_MAP
while IFS= read -r line; do
    line=$(echo "$line" | sed 's/^[ \t]*//')
    if [[ $line =~ ^NIC([0-9]+):[[:space:]]+([a-zA-Z0-9_]+) ]]; then
        index=${BASH_REMATCH[1]}
        nic_name=${BASH_REMATCH[2]}
        NIC_MAP["NIC$index"]="$nic_name"
        echo "DEBUG: Parsed NIC$index -> $nic_name"
    fi
done <<< "$LEGEND_LINES"

echo "DEBUG: NIC_MAP content:"
for key in "${!NIC_MAP[@]}"; do
    echo "DEBUG: $key -> ${NIC_MAP[$key]}"
done

# Strip '#' if present in header line for devices
CLEAN_HEADER=$(echo "$FILTERED_HEADER" | sed 's/^#//')
if [ -z "$CLEAN_HEADER" ]; then
    CLEAN_HEADER="$FILTERED_HEADER"
fi

IFS=' ' read -r -a DEVICES <<< "$CLEAN_HEADER"
echo "DEBUG: DEVICES length: ${#DEVICES[@]}"
for ((idx=0; idx<${#DEVICES[@]}; idx++)); do
    echo "DEBUG: DEVICES[$idx] -> '${DEVICES[$idx]}'"
done

function topo_cost() {
    local conn=$1
    # Print debug info to stderr so it doesn't pollute the numeric output
    >&2 echo "DEBUG: topo_cost called with conn='$conn'"
    if [[ $conn =~ NV[0-9]+ ]]; then
        echo 0
    elif [[ $conn == "PIX" ]]; then
        echo 1
    elif [[ $conn == "PXB" ]]; then
        echo 2
    elif [[ $conn == "PHB" ]]; then
        echo 3
    elif [[ $conn == "SYS" ]]; then
        echo 4
    elif [[ $conn == "X" ]]; then
        echo 99
    else
        echo 99
    fi
}

echo "DEBUG: Starting NIC selection for each GPU..."

declare -A BEST_NIC_PER_GPU

for GPU_ID in "${GPU_INDICES[@]}"; do
    GPU_NAME="GPU${GPU_ID}"
    echo "DEBUG: Processing $GPU_NAME..."

    GPU_LINE=$(echo -e "$FILTERED_TOPO" | grep "^$GPU_NAME ")
    if [[ -z "$GPU_LINE" ]]; then
        echo "ERROR: Unable to find line for $GPU_NAME in filtered topology."
        echo "DEBUG: Current FILTERED_TOPO:"
        echo -e "$FILTERED_TOPO"
        continue
    fi

    echo "DEBUG: Found GPU_LINE: '$GPU_LINE'"
    IFS=' ' read -r -a GPU_LINE_ENTRIES <<< "$GPU_LINE"
    echo "DEBUG: GPU_LINE_ENTRIES length: ${#GPU_LINE_ENTRIES[@]}"
    for ((idx=0; idx<${#GPU_LINE_ENTRIES[@]}; idx++)); do
        echo "DEBUG: GPU_LINE_ENTRIES[$idx] -> '${GPU_LINE_ENTRIES[$idx]}'"
    done

    BEST_NIC=""
    BEST_COST=99

    for ((i=0; i<${#DEVICES[@]}; i++)); do
        DEVICE_NAME=${DEVICES[$i]}
        DATA_INDEX=$((i+1))
        if (( DATA_INDEX >= ${#GPU_LINE_ENTRIES[@]} )); then
            echo "DEBUG: DATA_INDEX=$DATA_INDEX out of range for GPU_LINE_ENTRIES"
            continue
        fi
        CONN_TYPE=${GPU_LINE_ENTRIES[$DATA_INDEX]}
        echo "DEBUG: Checking column i=$i DEVICE_NAME='$DEVICE_NAME' with CONN_TYPE='$CONN_TYPE'"

        if [[ "$DEVICE_NAME" =~ ^NIC[0-9]+$ ]]; then
            NIC_NAME=${NIC_MAP[$DEVICE_NAME]}
            if [[ -z "$NIC_NAME" ]]; then
                idx_num=$(echo $DEVICE_NAME | sed 's/NIC//')
                NIC_NAME="mlx5_$idx_num"
                echo "DEBUG: $DEVICE_NAME not found in NIC_MAP. Using fallback name $NIC_NAME."
            fi

            # Capture only numeric output of topo_cost
            COST=$(topo_cost "$CONN_TYPE" 2>&1 | tail -n1)
            echo "DEBUG: After topo_cost, COST='$COST' for $DEVICE_NAME ($NIC_NAME) with CONN_TYPE='$CONN_TYPE'"

            # Check if COST is numeric
            if ! [[ "$COST" =~ ^[0-9]+$ ]]; then
                echo "DEBUG: COST is not a numeric value. COST='$COST'"
                COST=99
            fi

            echo "DEBUG: Final interpreted COST=$COST for NIC=$NIC_NAME"
            if (( COST < BEST_COST )); then
                echo "DEBUG: Updating best NIC for $GPU_NAME: $NIC_NAME (cost $COST) from previous cost $BEST_COST"
                BEST_COST=$COST
                BEST_NIC=$NIC_NAME
            fi
        fi
    done

    if [[ -z "$BEST_NIC" ]]; then
        echo "ERROR: No suitable NIC found for $GPU_NAME."
        echo "Please select a NIC manually."
    else
        echo "DEBUG: Selected NIC for $GPU_NAME: $BEST_NIC with cost $BEST_COST"
        BEST_NIC_PER_GPU[$GPU_NAME]=$BEST_NIC
    fi
done

echo "INFO: Final results:"
for gpu in "${!BEST_NIC_PER_GPU[@]}"; do
    echo "$gpu -> ${BEST_NIC_PER_GPU[$gpu]}"
done

