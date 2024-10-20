#!/bin/zsh

# Set the project directory to the current directory
PROJECT_DIR=$HOME/PROJECTS/cusplibrary/
TEST_DIR=$PROJECT_DIR/testing

# Create the output directory if it doesn't exist
output_dir="${TEST_DIR}/device_cuda_debug"
mkdir -p "$output_dir"

# Loop through all .cu files in the testing folder
for cu_file in ${TEST_DIR}/*.cu; do
    # Extract the filename without the extension
    filename=$(basename -- "$cu_file")
    filename="${filename%.*}"
    
    # Compile the .cu file
    echo "Compiling ${cu_file} to ${output_dir}/${filename}.o"
    echo "==================================================="
    nvcc -o "${output_dir}/${filename}.o" -c -arch=sm_50 \
        -Xcompiler -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA \
        -Xcompiler -DCUSP_DEVICE_BLAS_SYSTEM=CUSP_DEVICE_BLAS_CUBLAS \
        -Xcompiler -DCUSP_HOST_BLAS_SYSTEM=CUSP_HOST_BLAS_GENERIC \
        -Xcompiler -g -Xcompiler -DTHRUST_DEBUG -Xcompiler -Wall \
        -I "${PROJECT_DIR}" -I /opt/cuda/include -I "${TEST_DIR}" \
        "$cu_file"

    # Check if the compilation failed
    if [ $? -ne 0 ]; then
        echo "Compilation failed for ${cu_file}. Exiting..."
        break
    fi
done