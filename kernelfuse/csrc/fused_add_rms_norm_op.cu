// PyTorch binding: in-place fused_add_rms_norm (vLLM 0.8.5 semantics).
// Compiled as .cu so CUDAExtension uses nvcc (not MSVC cl on Windows).
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "add_rmsnorm_kernels.cuh"

namespace kernelfuse {

void fused_add_rms_norm(
    torch::Tensor& input,
    torch::Tensor& residual,
    torch::Tensor& weight,
    double epsilon) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA");
    TORCH_CHECK(residual.is_cuda(), "residual must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16, "input must be bf16");
    TORCH_CHECK(residual.scalar_type() == torch::kBFloat16, "residual must be bf16");
    TORCH_CHECK(weight.scalar_type() == torch::kBFloat16, "weight must be bf16");
    TORCH_CHECK(input.sizes() == residual.sizes(), "input/residual shape mismatch");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(residual.is_contiguous(), "residual must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");

    const int hidden_size = static_cast<int>(input.size(-1));
    const int num_tokens = static_cast<int>(input.numel() / hidden_size);
    TORCH_CHECK(static_cast<int>(weight.numel()) == hidden_size, "weight size mismatch");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    auto* in_ptr = reinterpret_cast<__nv_bfloat16*>(input.data_ptr<at::BFloat16>());
    auto* res_ptr = reinterpret_cast<__nv_bfloat16*>(residual.data_ptr<at::BFloat16>());
    auto* w_ptr = reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<at::BFloat16>());

    launch_fused_add_rms_norm_bf16(
        in_ptr, res_ptr, w_ptr, num_tokens, hidden_size, static_cast<float>(epsilon), stream);
}

}  // namespace kernelfuse

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "fused_add_rms_norm",
        &kernelfuse::fused_add_rms_norm,
        "In-place fused residual add + RMSNorm (bf16, vLLM semantics)");
}
