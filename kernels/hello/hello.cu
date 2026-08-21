// Phase 0 sanity check: compile → launch → retrieve on the local GPU.
#include <cstdio>
#include <cuda_runtime.h>

__global__ void hello_from_thread() {
    printf("Hello from GPU thread %d (block %d)\n",
           threadIdx.x, blockIdx.x);
}

int main() {
    hello_from_thread<<<1, 4>>>();
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("Host: kernel finished OK\n");
    return 0;
}
