#!/bin/bash

# Configuration for logging
LOG_ROOT_DIR="remesh_logs_full_config"
mkdir -p "$LOG_ROOT_DIR"

# Define specific directories for 2D and 3D logs
LOG_DIR_2D="${LOG_ROOT_DIR}/remesh_stats_2D"
LOG_DIR_3D="${LOG_ROOT_DIR}/remesh_stats_3D"

if [ -d "$LOG_DIR_2D" ]; then
    echo "Nettoyage des fichiers 'costToto' dans le répertoire: $LOG_DIR_2D"
    find "${LOG_DIR_2D}" -type f -name "*costToto*" -delete
else
    echo "Création du répertoire: $LOG_DIR_2D"
    mkdir -p "$LOG_DIR_2D"
fi
if [ -d "$LOG_DIR_3D" ]; then
    echo "Nettoyage des fichiers 'costToto' dans le répertoire: $LOG_DIR_3D"
    find "${LOG_DIR_3D}" -type f -name "*costToto*" -delete
else
    echo "Création du répertoire: $LOG_DIR_3D"
    mkdir -p "$LOG_DIR_3D"
fi

# Define the example names for your 2D and 3D remeshing applications
EXAMPLE_2D_NAME="remeshing_res_2d"
EXAMPLE_3D_NAME="remeshing_res_3d"

# Define the number of repetitions for each simulation
NUM_REPETITIONS=5

# Define the different parameters to test
SPLITS_2D=()
SPLITS_3D=(4 5 6)
METRIC_TYPES=("iso")
COST_ESTIMATORS=("Toto" "Nocost")
PARTITIONERS=("MetisRecursive")
OPTIONS=("true")

echo "Starting ALL remeshing tests (2D and 3D) with all configurations..."
echo "Each simulation will be run $NUM_REPETITIONS times."
echo "Logs will be saved in: $LOG_DIR_2D and $LOG_DIR_3D"
echo "----------------------------------------------------"


## 2D Remeshing Tests

echo "--- Starting 2D remeshing tests ---"
for splits in "${SPLITS_2D[@]}"; do
    for metric_type in "${METRIC_TYPES[@]}"; do
        for cost_estimator in "${COST_ESTIMATORS[@]}"; do
            for partitioner in "${PARTITIONERS[@]}"; do
                for option in "${OPTIONS[@]}"; do
                    for i in $(seq 1 $NUM_REPETITIONS); do
                        echo "Running 2D test (Repetition $i/$NUM_REPETITIONS) with:"
                        echo "  num_splits     = $splits"
                        echo "  metric_type    = $metric_type"
                        echo "  cost_estimator = $cost_estimator"
                        echo "  partitioner    = $partitioner"
                        echo "  option         = $option"

                        LOG_FILE="${LOG_DIR_2D}/splits${splits}_metric${metric_type}_cost${cost_estimator}_part${partitioner}_opt${option}_rep${i}.txt"

                        cargo run --release --features="metis" --example "$EXAMPLE_2D_NAME" -- \
                            --num-splits "$splits" \
                            --metric-type "$metric_type" \
                            --cost-estimator "$cost_estimator" \
                            --partitionner "$partitioner" \
                            --option "$option" \
                            > "$LOG_FILE" 2>&1

                        if [ $? -eq 0 ]; then
                            echo "2D test (Repetition $i) completed successfully. Log saved to: $LOG_FILE"
                        else
                            echo "2D test (Repetition $i) FAILED. Check log file for details: $LOG_FILE"
                        fi
                        echo "----------------------------------------------------"
                    done
                done
            done
        done
    done
done

echo "--- 2D remeshing tests completed. ---"
echo ""

## 3D Remeshing Tests

echo "--- Starting 3D remeshing tests ---"
for splits in "${SPLITS_3D[@]}"; do
    for metric_type in "${METRIC_TYPES[@]}"; do
        for cost_estimator in "${COST_ESTIMATORS[@]}"; do
            for partitioner in "${PARTITIONERS[@]}"; do
                for option in "${OPTIONS[@]}"; do
                    for i in $(seq 1 $NUM_REPETITIONS); do
                        echo "Running 3D test (Repetition $i/$NUM_REPETITIONS) with:"
                        echo "  num_splits     = $splits"
                        echo "  metric_type    = $metric_type"
                        echo "  cost_estimator = $cost_estimator"
                        echo "  partitioner    = $partitioner"
                        echo "  option         = $option"

                        LOG_FILE="${LOG_DIR_3D}/splits${splits}_metric${metric_type}_cost${cost_estimator}_part${partitioner}_opt${option}_rep${i}.txt"

                        cargo run --release --features="metis" --example "$EXAMPLE_3D_NAME" -- \
                            --num-splits "$splits" \
                            --metric-type "$metric_type" \
                            --cost-estimator "$cost_estimator" \
                            --partitionner "$partitioner" \
                            --option "$option" \
                            > "$LOG_FILE" 2>&1

                        if [ $? -eq 0 ]; then
                            echo "3D test (Repetition $i) completed successfully. Log saved to: $LOG_FILE"
                        else
                            echo "3D test (Repetition $i) FAILED. Check log file for details: $LOG_FILE"
                        fi
                        echo "----------------------------------------------------"
                    done
                done
            done
        done
    done
done

echo "--- 3D remeshing tests completed. ---"
echo ""
echo "All remeshing tests completed. Logs are in the '$LOG_ROOT_DIR/remesh_stats_2D' and '$LOG_ROOT_DIR/remesh_stats_3D' directories."
python Rmeshing_data_analysis.py