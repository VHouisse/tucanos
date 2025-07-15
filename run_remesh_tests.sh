#!/bin/bash

# Configuration for logging
LOG_ROOT_DIR="remesh_logs_full_config" # Renommé pour plus de clarté
mkdir -p "$LOG_ROOT_DIR"

# Définir les répertoires spécifiques pour les logs 2D et 3D
LOG_DIR_2D="${LOG_ROOT_DIR}/remesh_stats_2D"
LOG_DIR_3D="${LOG_ROOT_DIR}/remesh_stats_3D"

# Créer les répertoires pour les logs 2D et 3D
mkdir -p "$LOG_DIR_2D"
mkdir -p "$LOG_DIR_3D"

# Define the example names for your 2D and 3D remeshing applications
# IMPORTANT: These should match the names in your Cargo.toml for the examples.
# For example, if your 2D file is `examples/remeshing_2d_example.rs`, use "remeshing_2d_example"
# And if your 3D file is `examples/remeshing_3d_example.rs`, use "remeshing_3d_example"
EXAMPLE_2D_NAME="remeshing_res_2d" # Assuming you named your 2D example this way
EXAMPLE_3D_NAME="remeshing_res_3d" # Assuming you named your 3D example this way

# Define the different parameters to test
SPLITS_2D=(5 6)   # Number of initial splits for 2D mesh
SPLITS_3D=(3 4)   # Number of initial splits for 3D mesh
METRIC_TYPES=("iso" "aniso") # Metric types: isotropic and anisotropic
COST_ESTIMATORS=("Nocost" "Toto") # Cost estimators

# Base partitioners
PARTITIONERS=("HilbertBallPartitionner" "BFSPartitionner" "BFSWRPartitionner")

echo "Starting ALL remeshing tests (2D and 3D) with all configurations..."
echo "Logs will be saved in: $LOG_DIR_2D and $LOG_DIR_3D"
echo "----------------------------------------------------"


## 2D Remeshing Tests

echo "--- Starting 2D remeshing tests ---"
for splits in "${SPLITS_2D[@]}"; do
    for metric_type in "${METRIC_TYPES[@]}"; do
        for cost_estimator in "${COST_ESTIMATORS[@]}"; do
            for partitioner in "${PARTITIONERS[@]}"; do
                echo "Running 2D test with:"
                echo "  num_splits     = $splits"
                echo "  metric_type    = $metric_type"
                echo "  cost_estimator = $cost_estimator"
                echo "  partitioner    = $partitioner"

                # **CORRECTION ICI : suppression du "_2D" redondant dans le nom de fichier**
                LOG_FILE="${LOG_DIR_2D}/splits${splits}_metric${metric_type}_cost${cost_estimator}_part${partitioner}.txt"

                # Run the Rust program for 2D
                cargo run --release --example "$EXAMPLE_2D_NAME" -- \
                    --num-splits "$splits" \
                    --metric-type "$metric_type" \
                    --cost-estimator "$cost_estimator" \
                    --partitionner "$partitioner" \
                    > "$LOG_FILE" 2>&1

                # Check the exit status of the previous command
                if [ $? -eq 0 ]; then
                    echo "2D test completed successfully. Log saved to: $LOG_FILE"
                else
                    echo "2D test FAILED. Check log file for details: $LOG_FILE"
                fi
                echo "----------------------------------------------------"
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
                echo "Running 3D test with:"
                echo "  num_splits     = $splits"
                echo "  metric_type    = $metric_type"
                echo "  cost_estimator = $cost_estimator"
                echo "  partitioner    = $partitioner"

                # **CORRECTION ICI : suppression du "_3D" redondant dans le nom de fichier**
                LOG_FILE="${LOG_DIR_3D}/splits${splits}_metric${metric_type}_cost${cost_estimator}_part${partitioner}.txt"

                # Run the Rust program for 3D
                cargo run --release --example "$EXAMPLE_3D_NAME" -- \
                    --num-splits "$splits" \
                    --metric-type "$metric_type" \
                    --cost-estimator "$cost_estimator" \
                    --partitionner "$partitioner" \
                    > "$LOG_FILE" 2>&1

                # Check the exit status of the previous command
                if [ $? -eq 0 ]; then
                    echo "3D test completed successfully. Log saved to: $LOG_FILE"
                else
                    echo "3D test FAILED. Check log file for details: $LOG_FILE"
                fi
                echo "----------------------------------------------------"
            done
        done
    done
done

echo "--- 3D remeshing tests completed. ---"
echo ""
echo "All remeshing tests completed. Logs are in the '$LOG_ROOT_DIR/remesh_stats_2D' and '$LOG_ROOT_DIR/remesh_stats_3D' directories."