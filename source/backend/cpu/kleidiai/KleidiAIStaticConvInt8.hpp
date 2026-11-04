//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MNN_KLEIDIAI_STATIC_CONV_INT8_HPP
#define MNN_KLEIDIAI_STATIC_CONV_INT8_HPP

#include "backend/cpu/CPUConvolution.hpp"

namespace MNN {

// Static W2/W4/W8: immutable packed RHS is shared; all activation scratch is executor-local.
class KleidiAIStaticConvInt8 : public CPUConvolution {
public:
    static Execution* create(Backend* backend, const Op* op,
                             const std::shared_ptr<ConvolutionCommon::Int8Common>& quant,
                             const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs);
    ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
    ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
    bool onClone(Backend* backend, const Op* op, Execution** dst) override;

private:
    struct Resource;
    KleidiAIStaticConvInt8(Backend* backend, const Op* op, std::shared_ptr<const Resource> resource);
    std::shared_ptr<const Resource> mResource;
    std::shared_ptr<Tensor> mInput, mOutput, mPackedLhs, mPointers, mPadding;
    size_t mRows = 0;
    size_t mInputRows = 0;
    size_t mPackedRowBytes = 0;
    size_t mRowsPerThread = 0;
    int mThreads = 1;
    bool mUseDot = false;
};

} // namespace MNN
#endif