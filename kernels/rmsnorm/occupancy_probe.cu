// Occupancy probe for Phase 4 (no Nsight counters required).
// Uses cudaOccupancyMaxActiveBlocksPerMultiprocessor for the same launch
// configs as the RMSNorm kernels (256 threads, dynamic smem = cols*4).
//
// Build: scripts/build_occupancy_probe.ps1
// Run:   kernels/rmsnorm/occupancy_probe.exe

#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _err = (call);                                             \
        if (_err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(_err));                                 \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

static constexpr int kThreads = 256;

// Minimal stand-ins with the same resource signature as production kernels:
// dynamic smem float row + static partials (~1 KiB) + inv_rms.
__global__ void probe_smem_like(float* out, const float* in, int cols) {
    extern __shared__ float srow[];
    __shared__ float spartial[kThreads];
    __shared__ float sinv;
    int tid = threadIdx.x;
    float acc = 0.0f;
    for (int c = tid; c < cols; c += blockDim.x) {
        float v = in[c];
        srow[c] = v;
        acc += v * v;
    }
    spartial[tid] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            spartial[tid] += spartial[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        sinv = rsqrtf(spartial[0] / static_cast<float>(cols) + 1e-6f);
    }
    __syncthreads();
    float inv = sinv;
    for (int c = tid; c < cols; c += blockDim.x) {
        out[c] = srow[c] * inv;
    }
}

__global__ void probe_fused_reread(float* out, const float* in, int cols) {
    __shared__ float spartial[kThreads];
    __shared__ float sinv;
    int tid = threadIdx.x;
    float acc = 0.0f;
    for (int c = tid; c < cols; c += blockDim.x) {
        float v = in[c];
        acc += v * v;
    }
    spartial[tid] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            spartial[tid] += spartial[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        sinv = rsqrtf(spartial[0] / static_cast<float>(cols) + 1e-6f);
    }
    __syncthreads();
    float inv = sinv;
    for (int c = tid; c < cols; c += blockDim.x) {
        out[c] = in[c] * inv;
    }
}

static void print_device() {
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp p{};
    CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
    int smem_per_sm = 0;
    int max_thr_sm = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(
        &smem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev));
    CUDA_CHECK(cudaDeviceGetAttribute(
        &max_thr_sm, cudaDevAttrMaxThreadsPerMultiProcessor, dev));

    std::printf("DEVICE %s\n", p.name);
    std::printf("CC %d.%d\n", p.major, p.minor);
    std::printf("SMs %d\n", p.multiProcessorCount);
    std::printf("SMEM_PER_SM_BYTES %d\n", smem_per_sm);
    std::printf("MAX_THREADS_PER_SM %d\n", max_thr_sm);
    std::printf("WARP_SIZE %d\n", p.warpSize);
}

static void report_occupancy(const char* label, const void* kernel, int cols,
                             size_t dyn_smem) {
    int blocks = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks, kernel, kThreads, dyn_smem));
    int threads = blocks * kThreads;
    int warps = threads / 32;
    std::printf("OCC %s cols=%d dyn_smem=%zu blocks_per_sm=%d threads_per_sm=%d "
                "warps_per_sm=%d\n",
                label, cols, dyn_smem, blocks, threads, warps);
}

int main() {
    print_device();

    const int widths[] = {1024, 2048, 4096, 8192};
    for (int cols : widths) {
        size_t dyn = static_cast<size_t>(cols) * sizeof(float);
        report_occupancy("smem_like",
                         reinterpret_cast<const void*>(probe_smem_like), cols,
                         dyn);
    }
    // Fused re-read: no dynamic smem (static only) — occupancy should stay high.
    report_occupancy("fused_reread",
                     reinterpret_cast<const void*>(probe_fused_reread), 4096,
                     /*dyn_smem=*/0);
    report_occupancy("fused_reread",
                     reinterpret_cast<const void*>(probe_fused_reread), 8192,
                     /*dyn_smem=*/0);

    // Theory without static smem vs occupancy API (includes __shared__ partials).
    std::printf("THEORY dynamic-only sm_75: cols=4096 -> 4 blocks; cols=8192 -> 2 blocks (2.0x). Static spartial+sinv reduces this further - see OCC lines.\n");
    std::printf("Phase3 smem BW 122.3/45.8 = 2.67x; OCC thread ratio at 4096/8192 is the "
                "API prediction to compare against that measurement.\n");
    return 0;
}
