#pragma once

#include <c10/core/TensorOptions.h>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <torch/torch.h>
#include <unistd.h>

#if defined(NDEBUG)
#define MUSTRATEGO_COMPILE_MODE "RELEASE (NDEBUG)"
#elif defined(DEBUG)
#define MUSTRATEGO_COMPILE_MODE "DEBUG"
#else
#define MUSTRATEGO_COMPILE_MODE "RELEASE (ASSERTS)"
#endif

typedef unsigned __int128 uint128_t;

inline std::string UInt128ToString(uint128_t n)
{
    std::string out;
    out.reserve(20);
    while (n)
    {
        out += "0123456789"[n % 10];
        n /= 10;
    }
    std::reverse(out.begin(), out.end());
    return out;
}

namespace
{
    void _PrintMsSinceEpoch()
    {
        long int ms; // Milliseconds
        time_t s;    // Seconds
        struct timespec spec;

        clock_gettime(CLOCK_REALTIME, &spec);

        s = spec.tv_sec;
        ms = round(spec.tv_nsec / 1.0e6); // Convert nanoseconds to milliseconds
        if (ms > 999)
            ++s, ms = 0;

        printf("[µstratego:%" PRIdMAX ".%03ld", (intmax_t)s, ms);
    }
} // namespace

#ifndef __FILE_NAME__
#define __FILE_NAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#endif

#define MUSTRATEGO_LOG(...)                                             \
    {                                                                   \
        _PrintMsSinceEpoch();                                           \
        printf("|>%s] [%s:%03d] ",                                      \
               isatty(fileno(stdout)) ? "\033[34mINFO\033[0m" : "INFO", \
               __FILE_NAME__,                                           \
               __LINE__);                                               \
        printf(__VA_ARGS__);                                            \
        printf("\n");                                                   \
    }

#define MUSTRATEGO_DEBUG(...)                                             \
    {                                                                     \
        _PrintMsSinceEpoch();                                             \
        printf("|%s] [%s:%03d] ",                                         \
               isatty(fileno(stdout)) ? "\033[37mDEBUG\033[0m" : "DEBUG", \
               __FILE_NAME__,                                             \
               __LINE__);                                                 \
        printf(__VA_ARGS__);                                              \
        printf("\n");                                                     \
    }

#define MUSTRATEGO_WARN(...)                                                   \
    {                                                                          \
        _PrintMsSinceEpoch();                                                  \
        printf("|>%s] [%s:%03d] ",                                             \
               isatty(fileno(stdout)) ? "\033[1m\033[33mWARN\033[0m" : "WARN", \
               __FILE_NAME__,                                                  \
               __LINE__);                                                      \
        printf(__VA_ARGS__);                                                   \
        printf("\n");                                                          \
    }

#define MUSTRATEGO_FATAL(...)                                                    \
    {                                                                            \
        _PrintMsSinceEpoch();                                                    \
        printf("|%s] [%s:%03d] ",                                                \
               isatty(fileno(stdout)) ? "\033[1m\033[31mFATAL\033[0m" : "FATAL", \
               __FILE_NAME__,                                                    \
               __LINE__);                                                        \
        printf(__VA_ARGS__);                                                     \
        printf("\n");                                                            \
        std::abort();                                                            \
    }

#define MUSTRATEGO_CUDA_CHECK(OP)                                                     \
    {                                                                                 \
        cudaError_t result = (OP);                                                    \
        if (result != cudaSuccess)                                                    \
            MUSTRATEGO_FATAL("CUDA Runtime Error: %s\n", cudaGetErrorString(result)); \
    }

#define MUSTRATEGO_CHECK(CONDITION, ...)  \
    {                                     \
        if (!(CONDITION))                 \
            MUSTRATEGO_FATAL(__VA_ARGS__) \
    }

#define MUSTRATEGO_CHECK_IS_CUDA(X, DEVICE, ...) \
    MUSTRATEGO_CHECK(X.device().is_cuda() && X.device().index() == DEVICE, __VA_ARGS__)

#define MUSTRATEGO_CHECK_IS_DTYPE(X, TYPE, ...) \
    MUSTRATEGO_CHECK(X.dtype() == TYPE, __VA_ARGS__)

#define MUSTRATEGO_CHECK_CUDA_DTYPE(X, DEVICE, TYPE, ...) \
    MUSTRATEGO_CHECK_IS_CUDA(X, DEVICE, __VA_ARGS__)      \
    MUSTRATEGO_CHECK_IS_DTYPE(X, TYPE, __VA_ARGS__)

#define MUSTRATEGO_CHECK_DISTRIBUTION(X)                                                              \
    {                                                                                                 \
        MUSTRATEGO_CHECK(X.dim() == 1, "Distribution is not 1-dimensional");                          \
        const bool is_valid = (X.max() < INFINITY).item<bool>() && (X.min() >= 0).item<bool>();       \
        MUSTRATEGO_CHECK(is_valid, "Probability tensor contains either `inf`, `nan` or element < 0"); \
        const bool zero_prob = (X.sum() <= 0).item<bool>();                                           \
        MUSTRATEGO_CHECK(!zero_prob, "Invalid probability distribution (sum of probabilities <= 0)"); \
    }

#define MUSTRATEGO_CREATE_CUDA_TENSOR(VAR, DEVICE, DTYPE, ...)                                                                   \
    {                                                                                                                            \
        torch::TensorOptions options = torch::TensorOptions().device(torch::kCUDA, DEVICE).dtype(DTYPE).layout(torch::kStrided); \
        VAR = torch::empty(__VA_ARGS__, options);                                                                                \
    }

#define MUSTRATEGO_WRAP_CUDA_TENSOR(PTR, DEVICE, DTYPE, ...) \
    torch::from_blob(PTR, /* sizes */ __VA_ARGS__, torch::TensorOptions().device(torch::kCUDA, DEVICE).dtype(DTYPE).layout(torch::kStrided))

inline uint32_t ceil(uint64_t a, uint64_t b)
{
    return (a + b - 1) / b;
}

#if defined(MUSTRATEGO_USE_BFLOAT16)
constexpr const auto MUSTRATEGO_FLOAT_TORCH_DTYPE = torch::kBFloat16;
#elif defined(MUSTRATEGO_USE_FLOAT16)
constexpr const auto MUSTRATEGO_FLOAT_TORCH_DTYPE = torch::kFloat16;
#else
constexpr const auto MUSTRATEGO_FLOAT_TORCH_DTYPE = torch::kFloat32;
#endif

using MUSTRATEGO_FLOAT_CUDA_DTYPE = typename c10::impl::ScalarTypeToCPPType<MUSTRATEGO_FLOAT_TORCH_DTYPE>::type;

/// Fills up `out` with independent samples from the given distribution `probs`.
/// Both tensors are required to live on the given `cuda_device`.
///
/// The algorithm picks s = argmax( p / q ) where q ~ Exp(1), similarly to what `torch` does.
void SampleOut(const torch::Tensor probs, torch::Tensor out, const uint32_t cuda_device = 0);