#include "util.h"

#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/cuda/Reduce.cuh>
#include <torch/types.h>

void SampleOut(const torch::Tensor probs, torch::Tensor out, const uint32_t cuda_device)
{
#ifndef NDEBUG
    MUSTRATEGO_CHECK_CUDA_DTYPE(probs, (c10::DeviceIndex)cuda_device, torch::kFloat32, "Invalid tensor");
    MUSTRATEGO_CHECK_CUDA_DTYPE(out, (c10::DeviceIndex)cuda_device, torch::kInt32, "Invalid tensor");
    MUSTRATEGO_CHECK_DISTRIBUTION(probs);
#endif

    const torch::Tensor q = torch::empty({out.numel(), probs.numel()}, torch::TensorOptions()
                                                                           .device(torch::Device(torch::kCUDA, cuda_device))
                                                                           .dtype(torch::kFloat32))
                                .exponential_(1)
                                .reciprocal_() *
                            probs.unsqueeze(0);
    auto iter =
        at::meta::make_reduction(q, out.flatten(), /*dims=*/{1}, /*keepdim=*/false, torch::kFloat32);
    at::native::gpu_reduce_kernel<float, int32_t>(
        iter,
        at::native::ArgMaxOps<float>{},
        thrust::pair<float, int32_t>(-static_cast<float>(INFINITY), 0));
}