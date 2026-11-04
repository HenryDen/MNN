//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "KleidiAIStaticConvInt8.hpp"

#ifdef MNN_KLEIDIAI_STATIC_INT8
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include "KleidiAIStaticInt8Kernels.hpp"
#include "KleidiAICommon.hpp"
#include "core/Concurrency.h"
#include "core/TensorUtils.hpp"

namespace MNN {
namespace {
using namespace KleidiAIStaticInt8;

bool isInt8Tensor(const Tensor* tensor) {
    return tensor->getType() == halide_type_of<int8_t>() || CPUBackend::getDataType(tensor) == DataType_DT_INT8;
}

bool supportedBackend(Backend* backend) {
    auto cpu = static_cast<CPUBackend*>(backend);
    return cpu->functions()->supportSME2 && cpu->getRuntime()->hint().enableKleidiAI &&
           cpu->getRuntime()->hint().useCachedMmap == 0;
}

bool supportedFormat(const Tensor* tensor) {
    const auto format = TensorUtils::getDescribe(tensor)->dimensionFormat;
    return format == MNN_DATA_FORMAT_NC4HW4 || format == MNN_DATA_FORMAT_NCHW || format == MNN_DATA_FORMAT_NHWC;
}

bool tensorRows(const Tensor* tensor, size_t& rows) {
    return tensor->batch() > 0 && tensor->height() > 0 && tensor->width() > 0 &&
           product(size_t(tensor->height()), size_t(tensor->width()), rows) &&
           product(rows, size_t(tensor->batch()), rows);
}

bool tensorStorageFits(const Tensor* tensor, size_t rows, int pack) {
    size_t channels = tensor->channel(), bytes;
    if (TensorUtils::getDescribe(tensor)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4 &&
        (pack <= 0 || !roundUp(channels, size_t(pack), channels))) {
        return false;
    }
    return product(rows, channels, bytes);
}
} // namespace

struct KleidiAIStaticConvInt8::Resource {
    explicit Resource(Family family) : kernels(family) {
    }
    std::shared_ptr<Tensor> weight;
    KleidiAIStaticInt8::Kernels kernels;
    size_t k = 0, n = 0, kernelSize = 0;
    int32_t clampMin = -128, clampMax = 127;
    int8_t inputZeroPoint = 0;
};

Execution* KleidiAIStaticConvInt8::create(Backend* backend, const Op* op,
                                        const std::shared_ptr<ConvolutionCommon::Int8Common>& quant,
                                        const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    // Module pre-rearrangement uses float placeholders; explicit ConvInt8 pins the eventual runtime type.
    const bool explicitInt8 = op->type() == OpType_ConvInt8;
    if (!supportedBackend(backend) || inputs.size() != 1 || outputs.size() != 1 ||
        (!explicitInt8 && (!isInt8Tensor(inputs[0]) || !isInt8Tensor(outputs[0]))) ||
        !supportedFormat(inputs[0]) || !supportedFormat(outputs[0])) {
        return nullptr;
    }
    auto conv = op->main_as_Convolution2D();
    auto common = conv->common();
    auto sq = conv->symmetricQuan();
    auto iq = conv->quanParameter();
    const int k = common->inputCount(), n = common->outputCount();
    if (k <= 0 || n <= 0 || common->kernelX() <= 0 || common->kernelY() <= 0 ||
        common->strideX() <= 0 || common->strideY() <= 0 || common->dilateX() <= 0 || common->dilateY() <= 0 ||
        common->group() != 1 || common->relu6() || (common->pads() && common->pads()->size() == 1) ||
        !sq || sq->nbits() != 8 || sq->outputDataType() != DataType_DT_INT8 ||
        !sq->bias() || sq->bias()->size() != n || !sq->scale() || sq->scale()->size() != n ||
        (conv->bias() && conv->bias()->size() != 0) ||
        (iq && (iq->scaleIn() != 0.0f || iq->scaleOut() != 0.0f)) ||
        (quant && (quant->asymmetric || (quant->alphaSize != 0 && quant->alphaSize != n)))) {
        return nullptr;
    }
    size_t kernelSize, logicalK, count;
    if (!product(size_t(common->kernelX()), size_t(common->kernelY()), kernelSize) ||
        !product(size_t(k), kernelSize, logicalK) || !product(logicalK, size_t(n), count)) {
        return nullptr;
    }
    const bool packed = quant && quant->canUseInt4;
    const int8_t* weights = quant ? quant->weight.get() : (sq->weight() ? sq->weight()->data() : nullptr);
    const size_t weightSize = quant ? quant->weight.size() : (sq->weight() ? sq->weight()->size() : 0);
    if (!weights || weightSize < (packed ? count / 2 + count % 2 : count)) {
        return nullptr;
    }
    const auto valueAt = [weights, packed](size_t index) {
        const uint8_t byte = static_cast<uint8_t>(weights[packed ? index / 2 : index]);
        return packed ? int((byte >> (index % 2 == 0 ? 4 : 0)) & 15) - 8 : int(weights[index]);
    };
    int minWeight = 127, maxWeight = -128;
    for (size_t i = 0; i < count; ++i) {
        const int value = valueAt(i);
        minWeight = std::min(minWeight, value);
        maxWeight = std::max(maxWeight, value);
    }
    bool general = kernelSize != 1 || common->strideX() != 1 || common->strideY() != 1 ||
                   common->dilateX() != 1 || common->dilateY() != 1 || common->padX() != 0 || common->padY() != 0;
    if (common->pads()) {
        for (auto pad : *common->pads()) {
            general = general || pad != 0;
        }
    }
    Family family;
    if (general) {
        family = Family::IndirectSme2;
    } else if (minWeight >= -2 && maxWeight <= 1) {
        family = Family::W2;
    } else if (minWeight >= -8 && maxWeight <= 7) {
        family = Family::W4;
    } else {
        // Static layout heuristic, not a measured performance claim: legacy for N<=64, new otherwise.
        family = n <= 64 ? Family::Legacy : Family::W8;
    }
    if (sq->clampMin() < -128 || sq->clampMax() > 127) {
        return nullptr;
    }
    const int magnitude = std::max(-minWeight, maxWeight);
    for (int c = 0; c < n; ++c) {
        const float scale = sq->scale()->Get(c);
        if (quant && quant->alphaSize != 0 && quant->getAlphaFloat()[c] != scale) {
            return nullptr;
        }
        const int64_t bound = int64_t(logicalK) * 128 * magnitude + std::abs(int64_t(sq->bias()->Get(c)));
        if (!std::isfinite(scale) || scale < 0.0f || bound > std::numeric_limits<int32_t>::max() ||
            double(bound) * scale > std::numeric_limits<int32_t>::max()) {
            return nullptr;
        }
    }

    std::shared_ptr<Resource> resource(new Resource(family));
    resource->k = k;
    resource->n = n;
    resource->kernelSize = kernelSize;
    resource->inputZeroPoint = sq->zeroPoint();
    resource->clampMin = common->relu() ? std::max(int(sq->clampMin()), int(sq->outputZeroPoint())) : sq->clampMin();
    resource->clampMax = sq->clampMax();
    const size_t bytes = resource->kernels.rhsBytes(n, k, kernelSize);
    if (resource->clampMin > resource->clampMax || bytes == 0) {
        return nullptr;
    }
    const int bits = family == Family::W2 ? 2 : (family == Family::W4 ? 4 : 8);
    size_t rowBytes = size_t(k) / (8 / bits) + (size_t(k) % (8 / bits) != 0), sourceBytes;
    if (!product(rowBytes, size_t(n), sourceBytes) || !product(sourceBytes, kernelSize, sourceBytes)) {
        return nullptr;
    }
    std::vector<uint8_t> reordered(sourceBytes, 0);
    for (size_t c = 0; c < size_t(n); ++c) {
        for (size_t i = 0; i < size_t(k); ++i) {
            for (size_t p = 0; p < kernelSize; ++p) {
                const int value = valueAt((c * k + i) * kernelSize + p);
                if (bits == 8) {
                    // O,I,KH,KW -> kernel-position,I,N; each kernel position pads I independently.
                    reordered[(p * k + i) * n + c] = static_cast<uint8_t>(value);
                } else {
                    // IDST is high-first q+8. KAI W4 is low-first signed; QSU2 is low-first q+2.
                    const uint8_t code = bits == 2 ? uint8_t(value + 2) : uint8_t(value & 15);
                    reordered[c * rowBytes + i / (8 / bits)] |= code << ((i % (8 / bits)) * bits);
                }
            }
        }
    }
    resource->weight.reset(Tensor::createDevice<uint8_t>({int(bytes)}));
    if (!backend->onAcquireBuffer(resource->weight.get(), Backend::STATIC)) {
        return nullptr;
    }
    resource->kernels.packRhs(n, k, kernelSize, reordered.data(), sq->bias()->data(), sq->scale()->data(),
                              resource->weight->host<uint8_t>());
    return new KleidiAIStaticConvInt8(backend, op, resource);
}

KleidiAIStaticConvInt8::KleidiAIStaticConvInt8(Backend* backend, const Op* op,
                                            std::shared_ptr<const Resource> resource)
    : CPUConvolution(op->main_as_Convolution2D()->common(), backend), mResource(std::move(resource)) {
}

bool KleidiAIStaticConvInt8::onClone(Backend* backend, const Op* op, Execution** dst) {
    if (dst == nullptr) {
        return true;
    }
    if (!supportedBackend(backend)) {
        return false;
    }
    const Kernels current(mResource->kernels.family);
    if (current.mr != mResource->kernels.mr || current.nr != mResource->kernels.nr) {
        return false;
    }
    *dst = new KleidiAIStaticConvInt8(backend, op, mResource);
    return true;
}

ErrorCode KleidiAIStaticConvInt8::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input = inputs[0];
    auto output = outputs[0];
    const auto& res = *mResource;
    if (!supportedBackend(backend()) || !isInt8Tensor(input) || !isInt8Tensor(output) ||
        input->dimensions() != 4 || output->dimensions() != 4 || !supportedFormat(input) || !supportedFormat(output) ||
        input->channel() != res.k || output->channel() != res.n || input->batch() != output->batch()) {
        return NOT_SUPPORT;
    }
    const int pack = static_cast<CPUBackend*>(backend())->functions()->pack;
    if (!tensorRows(input, mInputRows) || !tensorRows(output, mRows) ||
        !tensorStorageFits(input, mInputRows, pack) || !tensorStorageFits(output, mRows, pack)) {
        return INPUT_DATA_ERROR;
    }
    // Use wide geometry arithmetic rather than overflowing the generic int padding helper.
    int64_t padX = mCommon->padX(), padY = mCommon->padY();
    if (mCommon->padMode() == PadMode_SAME) {
        padX = (int64_t(output->width() - 1) * mCommon->strideX() +
                int64_t(mCommon->kernelX() - 1) * mCommon->dilateX() + 1 - input->width());
        padY = (int64_t(output->height() - 1) * mCommon->strideY() +
                int64_t(mCommon->kernelY() - 1) * mCommon->dilateY() + 1 - input->height());
        padX = std::max(int64_t(0), padX) / 2;
        padY = std::max(int64_t(0), padY) / 2;
    } else if (mCommon->pads() && mCommon->pads()->size() >= 2) {
        padX = mCommon->pads()->Get(1);
        padY = mCommon->pads()->Get(0);
    }
    if (padX < std::numeric_limits<int>::min() || padX > std::numeric_limits<int>::max() ||
        padY < std::numeric_limits<int>::min() || padY > std::numeric_limits<int>::max()) {
        return NOT_SUPPORT;
    }
    mPadX = int(padX);
    mPadY = int(padY);
    if (!res.kernels.indirect() && (mInputRows != mRows || input->height() != output->height() ||
                                   input->width() != output->width() || mPadX != 0 || mPadY != 0)) {
        return NOT_SUPPORT;
    }
    // DOT is a whole-operation choice, never selected merely because a worker has a one-row tail.
    mUseDot = res.kernels.hasDot() && mRows == 1;
    size_t alignedRows, alignedK;
    if (!roundUp(mRows, res.kernels.mr, alignedRows) || !roundUp(res.k, 4, alignedK) ||
        !product(alignedK, res.kernelSize, mPackedRowBytes)) {
        return OUT_OF_MEMORY;
    }
    const size_t tiles = alignedRows / res.kernels.mr;
    mThreads = int(std::min(size_t(std::max(1, static_cast<CPUBackend*>(backend())->threadNumber())), tiles));
    const size_t tilesPerThread = tiles / mThreads + (tiles % mThreads != 0);
    size_t scheduledRows;
    if (!product(tilesPerThread, res.kernels.mr, mRowsPerThread) ||
        !product(size_t(mThreads), mRowsPerThread, scheduledRows)) {
        return OUT_OF_MEMORY;
    }
    size_t sizes[5] = {0, 0, 0, 1, 1};
    if (!product(mInputRows, res.k, sizes[0]) || !product(mRows, res.n, sizes[1]) ||
        !res.kernels.lhsBytes(mRows, res.k, res.kernelSize, sizes[2]) ||
        (res.kernels.indirect() && (!product(alignedRows, res.kernelSize, sizes[3]) ||
                                   !product(sizes[3], sizeof(void*), sizes[3])))) {
        return OUT_OF_MEMORY;
    }
    sizes[4] = res.kernels.indirect() ? res.k : 1;
    std::shared_ptr<Tensor>* buffers[] = {&mInput, &mOutput, &mPackedLhs, &mPointers, &mPadding};
    for (int i = 0; i < 5; ++i) {
        buffers[i]->reset(Tensor::createDevice<uint8_t>({int(sizes[i])}));
        if (!backend()->onAcquireBuffer(buffers[i]->get(), Backend::DYNAMIC)) {
            for (int j = 0; j < i; ++j) {
                backend()->onReleaseBuffer(buffers[j]->get(), Backend::DYNAMIC);
            }
            return OUT_OF_MEMORY;
        }
    }
    // Acquire together so the planner cannot alias live input/output/packed/pointer/padding scratch.
    for (auto buffer : buffers) {
        backend()->onReleaseBuffer(buffer->get(), Backend::DYNAMIC);
    }
    return NO_ERROR;
}

ErrorCode KleidiAIStaticConvInt8::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input = inputs[0];
    auto output = outputs[0];
    const auto& res = *mResource;
    auto core = static_cast<CPUBackend*>(backend())->functions();
    const auto inputFormat = TensorUtils::getDescribe(input)->dimensionFormat;
    const auto outputFormat = TensorUtils::getDescribe(output)->dimensionFormat;
    auto lhs = mInput->host<int8_t>();
    auto dst = mOutput->host<int8_t>();
    auto convert = [&](const void* src, void* dest, MNN_DATA_FORMAT from, MNN_DATA_FORMAT to,
                       const Tensor* tensor) -> ErrorCode {
        const int area = tensor->height() * tensor->width();
        const int threads = KleidiAICommon::recommendedThreadCount(
            dest, from, to, tensor->batch(), area, tensor->channel(),
            static_cast<CPUBackend*>(backend())->threadNumber());
        if (threads == 1) {
            return KleidiAICommon::convertLayout(src, dest, from, to, tensor->batch(), area, tensor->channel(), 1, core);
        }
        std::atomic<int> result(NO_ERROR);
        // GCD blocks capture by value; capture a pointer to the non-copyable atomic.
        auto error = &result;
        MNN_CONCURRENCY_BEGIN(tId, threads) {
            const auto code = KleidiAICommon::convertLayout(src, dest, from, to, tensor->batch(), area,
                                                           tensor->channel(), 1, core, tId, threads);
            if (code != NO_ERROR) {
                error->store(code, std::memory_order_relaxed);
            }
        } MNN_CONCURRENCY_END();
        return static_cast<ErrorCode>(result.load(std::memory_order_relaxed));
    };
    // Always stage both sides: graph input/output may alias, including NHWC views.
    if (inputFormat == MNN_DATA_FORMAT_NHWC) {
        std::memcpy(lhs, input->host<int8_t>(), mInputRows * res.k);
    } else {
        auto code = convert(input->host<int8_t>(), lhs, inputFormat, MNN_DATA_FORMAT_NHWC, input);
        if (code != NO_ERROR) {
            return code;
        }
    }
    if (res.kernels.indirect()) {
        // Quantized real zero is the serialized input zero point, not necessarily byte zero.
        std::memset(mPadding->host<int8_t>(), res.inputZeroPoint, res.k);
    }
    MNN_CONCURRENCY_BEGIN(tId, mThreads) {
        const size_t start = size_t(tId) * mRowsPerThread;
        if (start < mRows) {
            const size_t rows = std::min(mRowsPerThread, mRows - start);
            auto packed = mPackedLhs->host<uint8_t>() + start * mPackedRowBytes;
            const void* const* table = nullptr;
            if (res.kernels.indirect()) {
                auto pointers = mPointers->host<const void*>() + start * res.kernelSize;
                const size_t mr = res.kernels.mr;
                const size_t outputPlane = size_t(output->height()) * output->width();
                for (size_t tile = 0; tile < rows; tile += mr) {
                    for (size_t p = 0; p < res.kernelSize; ++p) {
                        for (size_t r = 0; r < mr; ++r) {
                            const size_t row = start + tile + r;
                            const void* ptr = mPadding->host<int8_t>();
                            if (tile + r < rows) {
                                const size_t batch = row / outputPlane;
                                const size_t position = row % outputPlane;
                                const int64_t y = int64_t(position / output->width()) * mCommon->strideY() - mPadY +
                                                  int64_t(p / mCommon->kernelX()) * mCommon->dilateY();
                                const int64_t x = int64_t(position % output->width()) * mCommon->strideX() - mPadX +
                                                  int64_t(p % mCommon->kernelX()) * mCommon->dilateX();
                                if (y >= 0 && y < input->height() && x >= 0 && x < input->width()) {
                                    ptr = lhs + ((batch * input->height() + size_t(y)) * input->width() + size_t(x)) * res.k;
                                }
                            }
                            pointers[tile * res.kernelSize + p * mr + r] = ptr;
                        }
                    }
                }
                table = pointers;
            }
            const void* kernelLhs = lhs;
            if (!mUseDot) {
                res.kernels.packLhs(rows, res.k, res.kernelSize,
                                    res.kernels.indirect() ? lhs : lhs + start * res.k,
                                    table, mPadding->host<int8_t>(), packed);
                kernelLhs = packed;
            }
            res.kernels.run(mUseDot, rows, res.n, res.k, res.kernelSize, kernelLhs,
                            res.weight->host<uint8_t>(), dst + start * res.n, res.clampMin, res.clampMax);
        }
    } MNN_CONCURRENCY_END();
    if (outputFormat == MNN_DATA_FORMAT_NHWC) {
        std::memcpy(output->host<int8_t>(), dst, mRows * res.n);
        return NO_ERROR;
    }
    return convert(dst, output->host<int8_t>(), MNN_DATA_FORMAT_NHWC, outputFormat, output);
}
} // namespace MNN
#endif // MNN_KLEIDIAI_STATIC_INT8