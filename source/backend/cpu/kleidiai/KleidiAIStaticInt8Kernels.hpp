//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MNN_KLEIDIAI_STATIC_INT8_KERNELS_HPP
#define MNN_KLEIDIAI_STATIC_INT8_KERNELS_HPP

#ifdef MNN_KLEIDIAI_STATIC_INT8
#include <cstddef>
#include <cstdint>
#include <limits>
#include "kai/kai_common.h"
#include "kai/ukernels/matmul/kai_matmul.h"
#include "kai/ukernels/matmul/kai_matmul_pack_rhs.h"

namespace MNN {
namespace KleidiAIStaticInt8 {

// Tensor byte lengths and all intermediate packed extents must fit the arena's int dimensions.
inline bool product(size_t a, size_t b, size_t& result) {
    const size_t limit = std::numeric_limits<int>::max();
    if (a > limit || b > limit || (b != 0 && a > limit / b)) {
        return false;
    }
    result = a * b;
    return true;
}
inline bool roundUp(size_t value, size_t alignment, size_t& result) {
    const size_t limit = std::numeric_limits<int>::max();
    if (alignment == 0 || alignment > limit || value > limit - (alignment - 1)) {
        return false;
    }
    result = (value + alignment - 1) / alignment * alignment;
    return true;
}

enum class Family { W2, W4, W8, Legacy, IndirectSme2 };

// One immutable RHS layout per family, shared by its vector/matrix kernels and executor clones.
struct Kernels {
    Family family = Family::W4;
    size_t mr = 0, nr = 0;
    kai_matmul_uker_api gemm = {}, dot = {};
    kai_matmul_pack_rhs_uker_api rhs = {};

    explicit Kernels(Family selected);
    bool indirect() const;
    bool modern() const;
    bool hasDot() const;
    size_t rhsBytes(size_t n, size_t k, size_t kernelSize) const;
    bool lhsBytes(size_t rows, size_t k, size_t kernelSize, size_t& bytes) const;
    void packRhs(size_t n, size_t k, size_t kernelSize, const void* weights,
                 const int32_t* bias, const float* scale, void* packed) const;
    void packLhs(size_t rows, size_t k, size_t kernelSize, const int8_t* input,
                 const void* const* pointers, const void* padding, void* packed) const;
    void run(bool useDot, size_t rows, size_t n, size_t k, size_t kernelSize,
             const void* lhs, const void* rhsPacked, void* output, int32_t min, int32_t max) const;
};

} // namespace KleidiAIStaticInt8
} // namespace MNN
#endif
#endif