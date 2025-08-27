#!/bin/bash

# Set the CPU affinity of the script to exclude cores 0 and 4
# Assuming system has at least 8 cores (0-7); using 1-3,5-7
taskset -cp 1-3,5-7 $$ >/dev/null
if [ $? -ne 0 ]; then
    echo "Error: Failed to set CPU affinity for the script (cores 1-3,5-7)"
    exit 1
fi

# Check if the executable exists
EXEC="./mat_mul/tiling"
if [ ! -x "$EXEC" ]; then
    echo "Error: Executable $EXEC not found or not executable"
    exit 1
fi

# Output CSV file
OUTPUT_FILE="perf_results.csv"

# Matrix sizes to test
MATRIX_SIZES=(2 4 8 16 32 64 128 256 512 1024 2048)

# Write CSV header
echo "Matrix size,L1-dcache-loads,L1-dcache-load-misses,Execution Time(ms)" > "$OUTPUT_FILE"

# Function to parse perf stat output and program output
parse_output() {
    local output="$1"
    local matrix_size="$2"
    local run_number="$3"

    # Extract L1-dcache-loads and L1-dcache-load-misses
    loads=$(echo "$output" | grep "L1-dcache-loads" | awk '{print $1}' | tr -d ',')
    misses=$(echo "$output" | grep "L1-dcache-load-misses" | awk '{print $1}' | tr -d ',')
    # Extract execution time from program output (e.g., "Tiling matrix multiplication took 63941 ms to execute")
    exec_time=$(echo "$output" | grep "Tiling matrix multiplication took" | awk '{print $5}')

    # Validate extracted values
    if [ -z "$loads" ] || [ -z "$misses" ] || [ -z "$exec_time" ]; then
        echo "Warning: Could not parse output for Matrix size $matrix_size, run $run_number"
        return 1
    fi

    echo "$loads $misses $exec_time"
    return 0
}

# Loop over each matrix size
for size in "${MATRIX_SIZES[@]}"; do
    echo "Testing matrix size $size..."

    # Arrays to store results for averaging
    declare -a loads_array
    declare -a misses_array
    declare -a exec_time_array
    valid_runs=0

    # Run the command 10 times
    for ((i=1; i<=10; i++)); do
        # Run perf stat with the program pinned to cores 0,4
        output=$(perf stat -e L1-dcache-loads,L1-dcache-load-misses taskset -c 0,4 "$EXEC" "$size" 2>&1)
        if [ $? -ne 0 ]; then
            echo "Warning: Run $i for matrix size $size failed"
            continue
        fi

        # Parse the output
        result=$(parse_output "$output" "$size" "$i")
        if [ $? -eq 0 ]; then
            read curr_loads curr_misses curr_exec_time <<< "$result"
            loads_array[$valid_runs]=$curr_loads
            misses_array[$valid_runs]=$curr_misses
            exec_time_array[$valid_runs]=$curr_exec_time
            ((valid_runs++))
        fi
    done

    # Calculate averages
    if [ $valid_runs -eq 0 ]; then
        echo "Error: No valid runs for matrix size $size"
        continue
    fi

    loads_sum=0
    misses_sum=0
    exec_time_sum=0
    for ((i=0; i<valid_runs; i++)); do
        loads_sum=$((loads_sum + loads_array[i]))
        misses_sum=$((misses_sum + misses_array[i]))
        exec_time_sum=$(echo "$exec_time_sum + ${exec_time_array[i]}" | bc)
    done

    avg_loads=$((loads_sum / valid_runs))
    avg_misses=$((misses_sum / valid_runs))
    avg_exec_time=$(echo "scale=3; $exec_time_sum / $valid_runs" | bc)

    # Append to CSV
    echo "$size,$avg_loads,$avg_misses,$avg_exec_time" >> "$OUTPUT_FILE"
done

echo "Results written to $OUTPUT_FILE"