// ---------------------------------------------------------------------------
// kernels.cu – stub CUDA kernels
// ---------------------------------------------------------------------------
#include <cuda_runtime.h>

// Example kernel – replace with real GPU work
__global__ void hello_kernel(int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // TODO: GPU work
    }
}
