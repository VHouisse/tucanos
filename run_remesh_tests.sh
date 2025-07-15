
LOG_DIR="remesh_logs"
mkdir -p "$LOG_DIR"

LOG_BASE_NAME="remesh_stats"


METRIC_TYPE="iso"


EXAMPLE_2D_NAME="remeshing_res_2d" 
EXAMPLE_3D_NAME="remeshing_res_3d" 

SPLITS_2D=(5 6)
SPLITS_3D=(3 4)

echo "Starting 2D remeshing tests..."

for splits in "${SPLITS_2D[@]}"; do
    echo "Running 2D test with num_splits = $splits..."
    LOG_FILE="${LOG_DIR}/${LOG_BASE_NAME}_2d_splits_${splits}_metric_${METRIC_TYPE}.txt"

    cargo run --release --example "$EXAMPLE_2D_NAME" -- --num-splits  "$splits" --metric-type "$METRIC_TYPE" > "$LOG_FILE" 2>&1

    if [ $? -eq 0 ]; then
        echo "2D test with $splits splits completed successfully. Log saved to: $LOG_FILE"
    else
        echo "2D test with $splits splits FAILED. Check log file for details: $LOG_FILE"
    fi
    echo "" t
done

echo "Starting 3D remeshing tests..."

for splits in "${SPLITS_3D[@]}"; do
    echo "Running 3D test with num_splits = $splits..."

    LOG_FILE="${LOG_DIR}/${LOG_BASE_NAME}_3d_splits_${splits}_metric_${METRIC_TYPE}.txt"

  
    cargo run --release --example "$EXAMPLE_3D_NAME" -- --num-splits  "$splits" --metric-type "$METRIC_TYPE" > "$LOG_FILE" 2>&1

    if [ $? -eq 0 ]; then
        echo "3D test with $splits splits completed successfully. Log saved to: $LOG_FILE"
    else
        echo "3D test with $splits splits FAILED. Check log file for details: $LOG_FILE"
    fi
    echo "" 
done

echo "All remeshing tests completed. Logs are in the '$LOG_DIR' directory."