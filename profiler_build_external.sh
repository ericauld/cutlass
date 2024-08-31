cd /scratch/ericauld/cut_build \
&& cmake ~/cut* -DCUTLASS_NVCC_ARCHS=90a \
    -DCUTLASS_ENABLE_TESTS=OFF \
    -DCUTLASS_UNITY_BUILD_ENABLED=ON \
    -DCUTLASS_LIBRARY_OPERATIONS=gemm \
    -DCUTLASS_LIBRARY_KERNELS="cutlass3x*bf16*" \
    -DCUTLASS_ENABLE_CUBLAS=ON  \
&& make cutlass_profiler -j64
