//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MNN_KLEIDIAI_COMMON_HPP
#define MNN_KLEIDIAI_COMMON_HPP

#include <MNN/ErrorCode.hpp>
#ifdef MNN_KLEIDIAI_STATIC_INT8
#include "Tensor_generated.h"
#endif

namespace MNN {
class Tensor;
struct CoreFunctions;

namespace KleidiAICommon {
// Floating KAI staging only; unsupported tensor contracts retain the CPU converter.
ErrorCode convertLayout(const Tensor* input, const Tensor* output, const CoreFunctions* core = nullptr,
                        int threadId = 0, int threadCount = 1);

#ifdef MNN_KLEIDIAI_STATIC_INT8
// INT8 pack4 staging policy; NC4 destinations not aligned to a four-byte cell stay serial.
int recommendedThreadCount(const void* dst, MNN_DATA_FORMAT to, int batch, int area, int channels,
                           int maxThreads);
// Direction-aware policy; the destination-only overload retains its conservative behavior.
int recommendedThreadCount(const void* dst, MNN_DATA_FORMAT from, MNN_DATA_FORMAT to,
                           int batch, int area, int channels, int maxThreads);

// Non-overlapping KAI staging: ARM64 INT8 pack4 NC4/NCHW <-> NHWC, including scalar tails.
// NC4 padding is byte zero; other contracts retain the CPU converter fallback.
// All workers [0, threadCount) must call; optimized workers own disjoint 128-byte destination lines.
ErrorCode convertLayout(const void* src, void* dst, MNN_DATA_FORMAT from, MNN_DATA_FORMAT to,
                        int batch, int area, int channels, int bytes, const CoreFunctions* core,
                        int threadId = 0, int threadCount = 1);
#endif
} // namespace KleidiAICommon
} // namespace MNN

#endif