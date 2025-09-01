#!/bin/bash

# Set CPU affinity (exclude cores 0 and 4)
taskset -cp 1-3,5-7 $$ >/dev/null
if [ $? -ne 0 ]; then
    echo "Error: Failed to set CPU affinity for the script (cores 1-3,5-7)"
    exit 1
fi

# Check if executable exists
EXEC="./mat_mul/tiling"
if [ ! -x "$EXEC" ]; then
    echo "Error: Executable $EXEC not found or not executable"
    exit 1
fi

# Output files
RAW_FILE="raw_results.csv"
AVG_FILE="avg_results.csv"

# Matrix sizes
MATRIX_SIZES=(32 64 128 256 512 831 1024 2048)

# Prepare output files
> "$RAW_FILE"
echo "Matrix size,Avg L1-dcache-loads,Avg L1-dcache-load-misses,Avg Execution Time(ms),Avg Instructions,Avg MPKI" > "$AVG_FILE"

# Parse perf output
parse_output() {
    local output="$1"
    loads=$(echo "$output" | grep "L1-dcache-loads" | awk '{print $1}' | tr -d ',')
    misses=$(echo "$output" | grep "L1-dcache-load-misses" | awk '{print $1}' | tr -d ',')
    instructions=$(echo "$output" | grep "instructions" | awk '{print $1}' | tr -d ',')
    exec_time=$(echo "$output" | grep "Tiling matrix multiplication took" | awk '{print $5}')

    if [ -z "$loads" ] || [ -z "$misses" ] || [ -z "$instructions" ] || [ -z "$exec_time" ]; then
        return 1
    fi
    echo "$loads $misses $instructions $exec_time"
    return 0
}

# Main loop
for size in "${MATRIX_SIZES[@]}"; do
    echo "" >> "$RAW_FILE"
    echo "# Matrix size $size" >> "$RAW_FILE"
    echo "Matrix size,L1-dcache-loads,L1-dcache-load-misses,Execution Time(ms),Instructions,MPKI" >> "$RAW_FILE"

    loads_sum=0
    misses_sum=0
    instructions_sum=0
    exec_time_sum=0
    mpki_sum=0
    valid_runs=0

    for ((i=1; i<=10; i++)); do
        output=$(perf stat -e L1-dcache-loads,L1-dcache-load-misses,instructions \
                 taskset -c 0,4 "$EXEC" "$size" 2>&1)

        result=$(parse_output "$output")
        if [ $? -eq 0 ]; then
            read l m instr t <<< "$result"

            # Compute MPKI per run
            if [ "$instr" -gt 0 ]; then
                mpki=$(echo "scale=3; ($m * 1000) / $instr" | bc)
            else
                mpki=0
            fi

            echo "$size,$l,$m,$t,$instr,$mpki" >> "$RAW_FILE"

            loads_sum=$((loads_sum + l))
            misses_sum=$((misses_sum + m))
            instructions_sum=$((instructions_sum + instr))
            exec_time_sum=$(echo "$exec_time_sum + $t" | bc)
            mpki_sum=$(echo "$mpki_sum + $mpki" | bc)
            ((valid_runs++))
        fi
    done

    if [ $valid_runs -eq 0 ]; then
        echo "❌ No valid runs for matrix size $size"
        continue
    fi

    avg_loads=$((loads_sum / valid_runs))
    avg_misses=$((misses_sum / valid_runs))
    avg_instructions=$((instructions_sum / valid_runs))
    avg_time=$(echo "scale=3; $exec_time_sum / $valid_runs" | bc)
    avg_mpki=$(echo "scale=3; $mpki_sum / $valid_runs" | bc)

    echo "$size,Avg,$avg_loads,$avg_misses,$avg_time,$avg_instructions,$avg_mpki" >> "$RAW_FILE"
    echo "$size,$avg_loads,$avg_misses,$avg_time,$avg_instructions,$avg_mpki" >> "$AVG_FILE"

    # Print live progress
    echo "✅ Matrix size $size done → Avg Loads=$avg_loads, Avg Misses=$avg_misses, Time=$avg_time ms, Instructions=$avg_instructions, MPKI=$avg_mpki"
done

echo ""
echo "Results written to:"
echo "  - $RAW_FILE (raw runs + averages per block, with instructions and MPKI per run)"
echo "  - $AVG_FILE (averages only)"