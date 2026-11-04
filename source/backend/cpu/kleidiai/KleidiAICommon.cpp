//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "KleidiAICommon.hpp"
#include "backend/cpu/CPUTensorConvert.hpp"
#include "core/TensorUtils.hpp"
#include <algorithm>
#include <cstdint>
#include <cstring>

#if defined(MNN_USE_NEON) || (defined(MNN_KLEIDIAI_STATIC_INT8) && defined(__aarch64__))
#include <arm_neon.h>
#endif

namespace MNN {
namespace KleidiAICommon {
namespace {
#if defined(MNN_USE_NEON)
bool isFloatTensor(const Tensor* tensor) {
    const auto type = tensor->getType();
    return type.code == halide_type_float && type.bits == 32 && type.lanes == 1;
}

bool isFloatingStaging(const Tensor* packed, const Tensor* nhwc, int bytes) {
    const auto type = nhwc->getType();
    // Pointwise and dynamic-W4 executors describe their FP16 NHWC scratch as int16.
    return isFloatTensor(packed) && (isFloatTensor(nhwc) ||
        (bytes == 2 && type.code == halide_type_int && type.bits == 16 && type.lanes == 1));
}

void transposeCells(const uint8_t* source, uint8_t* destination, size_t sourceStride,
                    size_t destinationStride, int rows, int columns) {
    // Preserve the channel order inside each opaque 16-byte cell.
    for (int rowBlock = 0; rowBlock < rows; rowBlock += 32) {
        for (int columnBlock = 0; columnBlock < columns; columnBlock += 32) {
            const int rowEnd = std::min(rowBlock + 32, rows);
            const int columnEnd = std::min(columnBlock + 32, columns);
            for (int row = rowBlock; row < rowEnd; ++row) {
                for (int column = columnBlock; column < columnEnd; ++column) {
                    const auto cell = vld1q_u8(source + size_t(row) * sourceStride + size_t(column) * 16);
                    vst1q_u8(destination + size_t(column) * destinationStride + size_t(row) * 16, cell);
                }
            }
        }
    }
}

void convertPacked(const uint8_t* source, uint8_t* destination, bool unpack, int plane, int channels,
                   int bytes, int pack, int threadId, int threadCount) {
    const int step = (plane + threadCount - 1) / threadCount;
    const int start = threadId * step;
    const int end = std::min(start + step, plane);
    if (end <= start) {
        return;
    }
    const int channelBlocks = channels / pack;
    const int tailChannels = channels % pack;
    const size_t nhwcStride = size_t(channels) * bytes;
    // NC blocks span the full batch/spatial plane, not this worker's slice.
    const size_t packedStride = size_t(plane) * 16;
    if (unpack) {
        transposeCells(source + size_t(start) * 16, destination + size_t(start) * nhwcStride,
                       packedStride, nhwcStride, channelBlocks, end - start);
    } else {
        transposeCells(source + size_t(start) * nhwcStride, destination + size_t(start) * 16,
                       nhwcStride, packedStride, end - start, channelBlocks);
    }
    if (tailChannels) {
        const size_t tailBytes = size_t(tailChannels) * bytes;
        for (int row = start; row < end; ++row) {
            const size_t packedOffset = size_t(channelBlocks) * packedStride + size_t(row) * 16;
            const size_t nhwcOffset = size_t(row) * nhwcStride + size_t(channelBlocks) * 16;
            if (unpack) {
                std::memcpy(destination + nhwcOffset, source + packedOffset, tailBytes);
            } else {
                std::memcpy(destination + packedOffset, source + nhwcOffset, tailBytes);
                std::memset(destination + packedOffset + tailBytes, 0, 16 - tailBytes);
            }
        }
    }
}
#endif

#ifdef MNN_KLEIDIAI_STATIC_INT8
#if defined(__aarch64__)
constexpr size_t kNc4TileCells = 32;
constexpr size_t kPlanarTileBytes = 8;
constexpr size_t kLayoutCacheLine = 128;

size_t spatialLayoutGroup(const void* dst, MNN_DATA_FORMAT to, int batch, int area) {
    const size_t offset = reinterpret_cast<uintptr_t>(dst) % kLayoutCacheLine;
    if (to == MNN_DATA_FORMAT_NC4HW4) {
        return offset % 4 == 0 ? kNc4TileCells : 0;
    }
    return to == MNN_DATA_FORMAT_NCHW && area >= 1024 ?
        kLayoutCacheLine : 0;
}

// Interior boundaries are absolute 128-byte lines (also separating 64-byte lines).
void layoutRange(size_t items, size_t group, size_t offset, int threadId, int threadCount,
                 size_t& start, size_t& end) {
    const size_t groups = (items + offset + group - 1) / group;
    const size_t step = (groups + threadCount - 1) / threadCount * group;
    start = threadId == 0 ? 0 : std::min(size_t(threadId) * step - offset, items);
    end = std::min(size_t(threadId + 1) * step - offset, items);
}

void transpose4ByteCellsNeon(const uint8_t* src, uint8_t* dst, size_t srcStrideBytes,
                            size_t dstStrideBytes, size_t rows, size_t cols) {
    for (size_t col = 0; col < cols; col += 4) {
        for (size_t row = 0; row < rows; row += 4) {
            const uint8_t* srcBlock = src + row * srcStrideBytes + col * 4;
            uint8_t* dstBlock = dst + col * dstStrideBytes + row * 4;
            const uint32x4_t row0 = vreinterpretq_u32_u8(vld1q_u8(srcBlock));
            const uint32x4_t row1 = vreinterpretq_u32_u8(vld1q_u8(srcBlock + srcStrideBytes));
            const uint32x4_t row2 = vreinterpretq_u32_u8(vld1q_u8(srcBlock + 2 * srcStrideBytes));
            const uint32x4_t row3 = vreinterpretq_u32_u8(vld1q_u8(srcBlock + 3 * srcStrideBytes));
            const uint32x4x2_t pair01 = vtrnq_u32(row0, row1);
            const uint32x4x2_t pair23 = vtrnq_u32(row2, row3);
            const uint64x2_t lo0 = vreinterpretq_u64_u32(pair01.val[0]);
            const uint64x2_t lo1 = vreinterpretq_u64_u32(pair23.val[0]);
            const uint64x2_t hi0 = vreinterpretq_u64_u32(pair01.val[1]);
            const uint64x2_t hi1 = vreinterpretq_u64_u32(pair23.val[1]);
            vst1q_u8(dstBlock, vreinterpretq_u8_u64(vtrn1q_u64(lo0, lo1)));
            vst1q_u8(dstBlock + dstStrideBytes, vreinterpretq_u8_u64(vtrn1q_u64(hi0, hi1)));
            vst1q_u8(dstBlock + 2 * dstStrideBytes, vreinterpretq_u8_u64(vtrn2q_u64(lo0, lo1)));
            vst1q_u8(dstBlock + 3 * dstStrideBytes, vreinterpretq_u8_u64(vtrn2q_u64(hi0, hi1)));
        }
    }
}

void convertNc4Nhwc(const void* source, void* dest, size_t spatialSize, size_t channels, bool nc4ToNhwc,
                    size_t start, size_t end) {
    // NC4 is [ceil(C/4)][B][area][4]; B*area is contiguous within each four-byte cell plane.
    const size_t rows = nc4ToNhwc ? channels / 4 : end - start;
    const size_t cols = nc4ToNhwc ? end - start : channels / 4;
    const size_t srcStrideBytes = nc4ToNhwc ? spatialSize * 4 : channels;
    const size_t dstStrideBytes = nc4ToNhwc ? channels : spatialSize * 4;
    const auto src = static_cast<const uint8_t*>(source) + start * (nc4ToNhwc ? 4 : channels);
    auto dst = static_cast<uint8_t*>(dest) + start * (nc4ToNhwc ? channels : 4);
    for (size_t rowBase = 0; rowBase < rows; rowBase += kNc4TileCells) {
        for (size_t colBase = 0; colBase < cols; colBase += kNc4TileCells) {
            const size_t tileRows = std::min(kNc4TileCells, rows - rowBase);
            const size_t tileCols = std::min(kNc4TileCells, cols - colBase);
            const size_t vectorRows = tileRows / 4 * 4;
            const size_t vectorCols = tileCols / 4 * 4;
            transpose4ByteCellsNeon(src + rowBase * srcStrideBytes + colBase * 4,
                                   dst + colBase * dstStrideBytes + rowBase * 4,
                                   srcStrideBytes, dstStrideBytes, vectorRows, vectorCols);
            for (size_t col = 0; col < tileCols; ++col) {
                for (size_t row = col < vectorCols ? vectorRows : 0; row < tileRows; ++row) {
                    std::memcpy(dst + (colBase + col) * dstStrideBytes + (rowBase + row) * 4,
                                src + (rowBase + row) * srcStrideBytes + (colBase + col) * 4, 4);
                }
            }
        }
    }
    const size_t alignedChannels = channels / 4 * 4;
    const size_t tailChannels = channels % 4;
    if (tailChannels != 0) {
        // NHWC rows have no padding; only the last NC4 plane has unused lanes, which are byte zero.
        for (size_t spatialIndex = 0; spatialIndex < end - start; ++spatialIndex) {
            if (nc4ToNhwc) {
                std::memcpy(dst + spatialIndex * channels + alignedChannels,
                            src + alignedChannels * spatialSize + spatialIndex * 4, tailChannels);
            } else {
                auto dstTail = dst + alignedChannels * spatialSize + spatialIndex * 4;
                std::memcpy(dstTail, src + spatialIndex * channels + alignedChannels, tailChannels);
                std::memset(dstTail + tailChannels, 0, 4 - tailChannels);
            }
        }
    }
}

void transposePlanarBytes(const uint8_t* src, uint8_t* dst, size_t rows, size_t cols,
                          size_t srcStride, size_t dstStride) {
    const size_t vectorRows = rows / kPlanarTileBytes * kPlanarTileBytes;
    const size_t vectorCols = cols / kPlanarTileBytes * kPlanarTileBytes;
    for (size_t row = 0; row < vectorRows; row += kPlanarTileBytes) {
        for (size_t col = 0; col < vectorCols; col += kPlanarTileBytes) {
            const auto srcBlock = src + row * srcStride + col;
            auto dstBlock = dst + col * dstStride + row;
            const uint8x8x2_t pair01 = vtrn_u8(vld1_u8(srcBlock), vld1_u8(srcBlock + srcStride));
            const uint8x8x2_t pair23 = vtrn_u8(vld1_u8(srcBlock + 2 * srcStride), vld1_u8(srcBlock + 3 * srcStride));
            const uint8x8x2_t pair45 = vtrn_u8(vld1_u8(srcBlock + 4 * srcStride), vld1_u8(srcBlock + 5 * srcStride));
            const uint8x8x2_t pair67 = vtrn_u8(vld1_u8(srcBlock + 6 * srcStride), vld1_u8(srcBlock + 7 * srcStride));
            for (size_t i = 0; i < 2; ++i) {
                const uint16x4x2_t rows03 = vtrn_u16(vreinterpret_u16_u8(pair01.val[i]),
                                                  vreinterpret_u16_u8(pair23.val[i]));
                const uint16x4x2_t rows47 = vtrn_u16(vreinterpret_u16_u8(pair45.val[i]),
                                                  vreinterpret_u16_u8(pair67.val[i]));
                for (size_t j = 0; j < 2; ++j) {
                    const uint32x2x2_t out = vtrn_u32(vreinterpret_u32_u16(rows03.val[j]),
                                                    vreinterpret_u32_u16(rows47.val[j]));
                    vst1_u8(dstBlock + (i + 2 * j) * dstStride, vreinterpret_u8_u32(out.val[0]));
                    vst1_u8(dstBlock + (i + 2 * j + 4) * dstStride, vreinterpret_u8_u32(out.val[1]));
                }
            }
        }
    }
    for (size_t row = 0; row < rows; ++row) {
        for (size_t col = row < vectorRows ? vectorCols : 0; col < cols; ++col) {
            dst[col * dstStride + row] = src[row * srcStride + col];
        }
    }
}

// Transpose the common rectangle directly; only boundary tiles need filtered stores.
template <size_t Cell>
void transposeWindow(const uint8_t* src, uint8_t* dst, size_t rows, size_t cols,
                     size_t srcStride, size_t dstStride, const size_t* begin, const size_t* end) {
    constexpr size_t tile = Cell == 1 ? 8 : 4;
    size_t first = rows, last = 0, innerFirst = 0, innerLast = rows;
    for (size_t c = 0; c < cols; ++c) {
        if (begin[c] < end[c]) {
            first = std::min(first, begin[c]);
            last = std::max(last, end[c]);
        }
        innerFirst = std::max(innerFirst, begin[c]);
        innerLast = std::min(innerLast, end[c]);
    }
    if (last <= first) {
        return;
    }
    innerFirst = std::min(last, std::max(first, innerFirst));
    innerLast = std::max(innerFirst, std::min(last, innerLast));
    auto transpose = [&](const uint8_t* input, uint8_t* output, size_t count, size_t stride) {
        if (Cell == 1) {
            transposePlanarBytes(input, output, count, cols, srcStride, stride);
        } else {
            const size_t nr = count / 4 * 4, nc = cols / 4 * 4;
            transpose4ByteCellsNeon(input, output, srcStride, stride, nr, nc);
            for (size_t c = 0; c < cols; ++c) {
                for (size_t r = c < nc ? nr : 0; r < count; ++r) {
                    std::memcpy(output + c * stride + r * Cell, input + r * srcStride + c * Cell, Cell);
                }
            }
        }
    };
    if (innerLast > innerFirst) {
        transpose(src + innerFirst * srcStride, dst + innerFirst * Cell, innerLast - innerFirst, dstStride);
    }
    for (int side = 0; side < 2; ++side) {
        const size_t low = side == 0 ? first : innerLast;
        const size_t high = side == 0 ? std::min(innerFirst, last) : last;
        for (size_t r = low; r < high; r += tile) {
            const size_t count = std::min(tile, high - r);
            if (Cell == 1 && count == 8 && cols == 8) {
                const auto input = src + r * srcStride;
                const auto pair01 = vtrn_u8(vld1_u8(input), vld1_u8(input + srcStride));
                const auto pair23 = vtrn_u8(vld1_u8(input + 2 * srcStride), vld1_u8(input + 3 * srcStride));
                const auto pair45 = vtrn_u8(vld1_u8(input + 4 * srcStride), vld1_u8(input + 5 * srcStride));
                const auto pair67 = vtrn_u8(vld1_u8(input + 6 * srcStride), vld1_u8(input + 7 * srcStride));
                auto store = [&](size_t c, uint8x8_t value) {
                    const size_t a = std::max(r, begin[c]), z = std::min(r + 8, end[c]);
                    if (a == r && z == r + 8) {
                        vst1_u8(dst + c * dstStride + r, value);
                    } else if (a < z) {
                        uint8_t lanes[8];
                        vst1_u8(lanes, value);
                        std::memcpy(dst + c * dstStride + a, lanes + (a - r), z - a);
                    }
                };
                for (size_t i = 0; i < 2; ++i) {
                    const auto rows03 = vtrn_u16(vreinterpret_u16_u8(pair01.val[i]), vreinterpret_u16_u8(pair23.val[i]));
                    const auto rows47 = vtrn_u16(vreinterpret_u16_u8(pair45.val[i]), vreinterpret_u16_u8(pair67.val[i]));
                    for (size_t j = 0; j < 2; ++j) {
                        const auto out = vtrn_u32(vreinterpret_u32_u16(rows03.val[j]), vreinterpret_u32_u16(rows47.val[j]));
                        store(i + 2 * j, vreinterpret_u8_u32(out.val[0]));
                        store(i + 2 * j + 4, vreinterpret_u8_u32(out.val[1]));
                    }
                }
                continue;
            }
            uint8_t buffer[tile * tile * Cell];
            transpose(src + r * srcStride, buffer, count, tile * Cell);
            for (size_t c = 0; c < cols; ++c) {
                const size_t a = std::max(r, begin[c]), z = std::min(r + count, end[c]);
                if (a < z) {
                    std::memcpy(dst + c * dstStride + a * Cell,
                                buffer + c * tile * Cell + (a - r) * Cell, (z - a) * Cell);
                }
            }
        }
    }
}

// Separate full plane lines from head/tail lines that can straddle the next plane.
void planeRange(const uint8_t* dst, size_t bytes, int tid, int threads, int part, size_t& first, size_t& last) {
    const size_t head = std::min(bytes, (kLayoutCacheLine - uintptr_t(dst) % kLayoutCacheLine) % kLayoutCacheLine);
    const size_t tail = head + (bytes - head) / kLayoutCacheLine * kLayoutCacheLine;
    first = last = 0;
    if (part == 0) {
        layoutRange(tail - head, kLayoutCacheLine, 0, tid, threads, first, last);
        first += head;
        last += head;
    } else if (tid == 0) {
        first = part == 1 ? 0 : tail;
        last = part == 1 ? head : bytes;
    }
}

template <size_t Cell>
void convertSpatialOwned(const uint8_t* src, uint8_t* dst, size_t rows, size_t cols,
                         size_t srcStride, int tid, int threads, size_t edgeFirst = 0, size_t edgeLast = 0) {
    constexpr size_t tile = Cell == 1 ? 8 : 4;
    const size_t stride = rows * Cell;
    size_t commonFirst = 0, commonLast = rows;
    if (Cell == 1) {
        for (size_t c = 0; c < cols; ++c) {
            size_t a, z;
            planeRange(dst + c * stride, stride, tid, threads, 0, a, z);
            commonFirst = std::max(commonFirst, a);
            commonLast = std::min(commonLast, z);
        }
        commonLast = std::max(commonFirst, commonLast);
        if (commonFirst < commonLast) {
            transposePlanarBytes(src + commonFirst * srcStride, dst + commonFirst,
                                 commonLast - commonFirst, cols, srcStride, stride);
        }
    }
    for (int part = 0; part < (Cell == 1 || tid == 0 ? 3 : 1); ++part) {
        const bool ownedEdge = Cell == 1 && part != 0;
        const size_t firstCol = ownedEdge ? edgeFirst / stride / tile * tile : 0;
        const size_t lastCol = ownedEdge ? (edgeLast + stride - 1) / stride : cols;
        for (size_t c = firstCol; c < lastCol; c += tile) {
            const size_t count = std::min(tile, cols - c);
            size_t begin[tile], end[tile];
            for (size_t i = 0; i < count; ++i) {
                planeRange(dst + (c + i) * stride, stride, ownedEdge ? 0 : tid, threads, part, begin[i], end[i]);
                if (ownedEdge) {
                    const size_t base = (c + i) * stride;
                    const size_t a = edgeFirst > base ? std::min(stride, edgeFirst - base) : 0;
                    const size_t z = edgeLast > base ? std::min(stride, edgeLast - base) : 0;
                    begin[i] = std::max(begin[i], a);
                    end[i] = std::max(begin[i], std::min(end[i], z));
                }
                begin[i] /= Cell;
                end[i] /= Cell;
            }
            if (Cell == 1 && part == 0 && commonFirst < commonLast) {
                size_t edgeBegin[tile], edgeEnd[tile];
                for (size_t i = 0; i < count; ++i) {
                    edgeBegin[i] = commonLast;
                    edgeEnd[i] = end[i];
                    end[i] = commonFirst;
                }
                transposeWindow<Cell>(src + c * Cell, dst + c * stride, rows, count, srcStride, stride,
                                      edgeBegin, edgeEnd);
            }
            transposeWindow<Cell>(src + c * Cell, dst + c * stride, rows, count, srcStride, stride, begin, end);
        }
    }
}

void convertPlanarOwned(const uint8_t* src, uint8_t* dst, size_t rows, size_t cols,
                        size_t srcStride, size_t first, size_t last) {
    if (first >= last) {
        return;
    }
    const size_t innerFirst = ((first + rows - 1) / rows + 7) / 8 * 8;
    const size_t innerLast = last / rows / 8 * 8;
    if (innerFirst < innerLast) {
        transposePlanarBytes(src + innerFirst, dst + innerFirst * rows, rows,
                             innerLast - innerFirst, srcStride, rows);
    }
    for (size_t c = first / rows / 8 * 8; c <= (last - 1) / rows; c += 8) {
        if (c >= innerFirst && c < innerLast) {
            c = innerLast - 8;
            continue;
        }
        const size_t count = std::min(size_t(8), cols - c);
        size_t begin[8], end[8];
        for (size_t i = 0; i < count; ++i) {
            const size_t base = (c + i) * rows;
            begin[i] = first > base ? std::min(rows, first - base) : 0;
            end[i] = last > base ? std::min(rows, last - base) : 0;
        }
        transposeWindow<1>(src + c, dst + c * rows, rows, count, srcStride, rows, begin, end);
    }
}

// Short NC4 planes are grouped in output order; only the two global worker boundaries are filtered.
void convertNc4Owned(const uint8_t* src, uint8_t* dst, size_t rows, size_t channels,
                     size_t first, size_t last) {
    const size_t stride = rows * 4, planes = channels / 4;
    if (first >= last) {
        return;
    }
    const size_t innerFirst = ((first + stride - 1) / stride + 3) / 4 * 4;
    const size_t innerLast = std::min(planes, last / stride) / 4 * 4;
    for (size_t r = 0; r < rows; r += kNc4TileCells) {
        for (size_t c = innerFirst; c < innerLast; c += kNc4TileCells) {
            const size_t nr = std::min(kNc4TileCells, rows - r);
            const size_t nc = std::min(kNc4TileCells, innerLast - c);
            transpose4ByteCellsNeon(src + r * channels + c * 4, dst + c * stride + r * 4,
                                   channels, stride, nr / 4 * 4, nc);
            for (size_t i = 0; i < nc; ++i) {
                for (size_t j = nr / 4 * 4; j < nr; ++j) {
                    std::memcpy(dst + (c + i) * stride + (r + j) * 4,
                                src + (r + j) * channels + (c + i) * 4, 4);
                }
            }
        }
    }
    for (size_t c = first / stride / 4 * 4; c < planes && c <= (last - 1) / stride; c += 4) {
        if (c >= innerFirst && c < innerLast) {
            c = innerLast - 4;
            continue;
        }
        const size_t count = std::min(size_t(4), planes - c);
        size_t begin[4], end[4];
        for (size_t i = 0; i < count; ++i) {
            const size_t base = (c + i) * stride;
            begin[i] = (first > base ? std::min(stride, first - base) : 0) / 4;
            end[i] = (last > base ? std::min(stride, last - base) : 0) / 4;
        }
        transposeWindow<4>(src + c * 4, dst + c * stride, rows, count, channels, stride, begin, end);
    }
    if (channels % 4 && last > planes * stride) {
        const size_t a = first > planes * stride ? (first - planes * stride) / 4 : 0;
        const size_t z = (last - planes * stride) / 4;
        for (size_t r = a; r < z; ++r) {
            auto output = dst + planes * stride + r * 4;
            std::memcpy(output, src + r * channels + planes * 4, channels % 4);
            std::memset(output + channels % 4, 0, 4 - channels % 4);
        }
    }
}

void convertLayoutFragment(const uint8_t* src, uint8_t* dst, MNN_DATA_FORMAT from, MNN_DATA_FORMAT to,
                           size_t batch, size_t area, size_t channels, size_t start, size_t end) {
    if (start >= end) {
        return;
    }
    if (from == MNN_DATA_FORMAT_NC4HW4) {
        // Each fragment is contained within one NHWC row.
        const size_t row = start / channels;
        size_t i = start;
        for (; i < end && (i - row * channels) % 4; ++i) {
            const size_t c = i - row * channels;
            dst[i] = src[(c / 4 * batch * area + row) * 4 + c % 4];
        }
        for (; i + 4 <= end; i += 4) {
            const size_t c = i - row * channels;
            std::memcpy(dst + i, src + (c / 4 * batch * area + row) * 4, 4);
        }
        for (; i < end; ++i) {
            const size_t c = i - row * channels;
            dst[i] = src[(c / 4 * batch * area + row) * 4 + c % 4];
        }
        return;
    }
    const size_t batchBytes = area * channels, b = start / batchBytes;
    const bool unpack = to == MNN_DATA_FORMAT_NHWC;
    convertPlanarOwned(src + b * batchBytes, dst + b * batchBytes, unpack ? channels : area,
                       unpack ? area : channels, unpack ? area : channels,
                       start - b * batchBytes, end - b * batchBytes);
}
#endif
#endif
} // namespace

ErrorCode convertLayout(const Tensor* input, const Tensor* output, const CoreFunctions* core,
                        int threadId, int threadCount) {
    if (core == nullptr) {
        core = MNNGetCoreFunctions();
    }
#if defined(MNN_USE_NEON)
    const auto inputDesc = TensorUtils::getDescribe(input);
    const auto outputDesc = TensorUtils::getDescribe(output);
    const auto from = inputDesc->dimensionFormat;
    const auto to = outputDesc->dimensionFormat;
    const bool unpack = from == MNN_DATA_FORMAT_NC4HW4 && to == MNN_DATA_FORMAT_NHWC;
    const bool pack = from == MNN_DATA_FORMAT_NHWC && to == MNN_DATA_FORMAT_NC4HW4;
    if (core && input->dimensions() > 1 && (unpack || pack) &&
        ((core->bytes == 4 && core->pack == 4) || (core->bytes == 2 && core->pack == 8)) &&
        !(inputDesc->quantAttr && inputDesc->applyQuant) &&
        !(outputDesc->quantAttr && outputDesc->applyQuant) &&
        isFloatingStaging(unpack ? input : output, unpack ? output : input, core->bytes)) {
        const auto shape = CPUTensorConverter::splitDimensions(input->buffer(), from);
        const int batch = std::get<0>(shape);
        const int area = std::get<1>(shape);
        const int channels = std::get<2>(shape);
        if (batch > 0 && area > 0 && channels > 0) {
            convertPacked(input->host<uint8_t>(), output->host<uint8_t>(), unpack, batch * area, channels,
                          core->bytes, core->pack, threadId, threadCount);
            return NO_ERROR;
        }
    }
#endif
    return CPUTensorConverter::convert(input, output, core, threadId, threadCount);
}

#ifdef MNN_KLEIDIAI_STATIC_INT8
int recommendedThreadCount(const void* dst, MNN_DATA_FORMAT to, int batch, int area, int channels,
                           int maxThreads) {
#if defined(__aarch64__)
    if (batch > 0 && area > 0 && channels > 0 && maxThreads > 1) {
        const size_t group = spatialLayoutGroup(dst, to, batch, area);
        if (to != MNN_DATA_FORMAT_NC4HW4 || group != 0) {
            const size_t usefulBytes = size_t(batch) * area * channels;
            const size_t items = group != 0 ? size_t(batch) * area / group : usefulBytes / kLayoutCacheLine;
            // Amortize pool dispatch with useful bytes, not padded storage or just row count.
            const size_t minimumWork = to == MNN_DATA_FORMAT_NHWC ?
                std::max(size_t(16 * 1024), size_t(channels) * kNc4TileCells) : size_t(16 * 1024);
            const size_t workers = usefulBytes / minimumWork;
            return int(std::max(size_t(1), std::min(size_t(maxThreads), std::min(items, workers))));
        }
    }
#endif
    return 1;
}

int recommendedThreadCount(const void* dst, MNN_DATA_FORMAT from, MNN_DATA_FORMAT to,
                           int batch, int area, int channels, int maxThreads) {
    int workers = recommendedThreadCount(dst, to, batch, area, channels, maxThreads);
#if defined(__aarch64__)
    if (batch > 0 && area > 0 && channels > 0 && maxThreads > 1) {
        const size_t spatial = size_t(batch) * area;
        const size_t useful = spatial * channels;
        // Channel tails make small unpack worth splitting; aligned cells retain the base work threshold.
        if (from == MNN_DATA_FORMAT_NC4HW4 && to == MNN_DATA_FORMAT_NHWC &&
            channels >= 16 && channels % 4 != 0 && spatial >= 128 && useful < 32 * 1024) {
            workers = int(std::min(size_t(maxThreads), std::min(spatial / kNc4TileCells, useful / 2048)));
        }
        // Unequal planar alignment adds masked strips at every spatial boundary. Keep at least
        // four full lines per worker; regularly aligned large planes can still use the whole pool.
        if (to == MNN_DATA_FORMAT_NCHW && area >= 1024 && area % kLayoutCacheLine != 0) {
            workers = std::min(workers, int(std::max(size_t(1), spatial / (4 * kLayoutCacheLine))));
        }
        int balanced = 1;
        while (balanced <= workers / 2) {
            balanced *= 2;
        }
        return balanced;
    }
#endif
    return workers;
}

ErrorCode convertLayout(const void* src, void* dst, MNN_DATA_FORMAT from, MNN_DATA_FORMAT to,
                        int batch, int area, int channels, int bytes, const CoreFunctions* core,
                        int threadId, int threadCount) {
#if defined(__aarch64__)
    if (bytes == 1 && core->pack == 4 && batch > 0 && area > 0 && channels > 0 &&
        ((from == MNN_DATA_FORMAT_NHWC && (to == MNN_DATA_FORMAT_NC4HW4 || to == MNN_DATA_FORMAT_NCHW)) ||
         (to == MNN_DATA_FORMAT_NHWC && (from == MNN_DATA_FORMAT_NC4HW4 || from == MNN_DATA_FORMAT_NCHW)))) {
        const size_t spatialSize = size_t(batch) * area;
        const size_t group = spatialLayoutGroup(dst, to, batch, area);
        size_t start = 0, end = spatialSize;
        if (threadCount > 1 && group != 0) {
            if (to == MNN_DATA_FORMAT_NC4HW4 && reinterpret_cast<uintptr_t>(dst) % kLayoutCacheLine == 0 &&
                spatialSize % kNc4TileCells == 0) {
                layoutRange(spatialSize, kNc4TileCells, 0, threadId, threadCount, start, end);
                if (start < end) {
                    convertNc4Nhwc(src, dst, spatialSize, channels, false, start, end);
                }
                return NO_ERROR;
            }
            if (to == MNN_DATA_FORMAT_NC4HW4 && spatialSize <= 256 && channels >= 64) {
                layoutRange(spatialSize * ((size_t(channels) + 3) / 4) * 4, kLayoutCacheLine,
                            reinterpret_cast<uintptr_t>(dst) % kLayoutCacheLine,
                            threadId, threadCount, start, end);
                convertNc4Owned(static_cast<const uint8_t*>(src), static_cast<uint8_t*>(dst),
                                spatialSize, channels, start, end);
                return NO_ERROR;
            }
            if (to == MNN_DATA_FORMAT_NCHW) {
                const size_t batchBytes = size_t(area) * channels;
                size_t edgeFirst, edgeLast;
                layoutRange(spatialSize * channels, kLayoutCacheLine,
                            reinterpret_cast<uintptr_t>(dst) % kLayoutCacheLine,
                            threadId, threadCount, edgeFirst, edgeLast);
                for (int b = 0; b < batch; ++b) {
                    const size_t base = b * batchBytes;
                    const size_t a = edgeFirst > base ? std::min(batchBytes, edgeFirst - base) : 0;
                    const size_t z = edgeLast > base ? std::min(batchBytes, edgeLast - base) : 0;
                    convertSpatialOwned<1>(static_cast<const uint8_t*>(src) + b * batchBytes,
                                           static_cast<uint8_t*>(dst) + b * batchBytes,
                                           area, channels, channels, threadId, threadCount, a, z);
                }
            } else {
                convertSpatialOwned<4>(static_cast<const uint8_t*>(src), static_cast<uint8_t*>(dst),
                                       spatialSize, channels / 4, channels, threadId, threadCount);
                if (channels % 4) {
                    auto tail = static_cast<uint8_t*>(dst) + channels / 4 * spatialSize * 4;
                    for (int part = 0; part < (threadId == 0 ? 3 : 1); ++part) {
                        size_t a, z;
                        planeRange(tail, spatialSize * 4, threadId, threadCount, part, a, z);
                        for (size_t r = a / 4; r < z / 4; ++r) {
                            std::memcpy(tail + r * 4,
                                        static_cast<const uint8_t*>(src) + r * channels + channels / 4 * 4,
                                        channels % 4);
                            std::memset(tail + r * 4 + channels % 4, 0, 4 - channels % 4);
                        }
                    }
                }
            }
            return NO_ERROR;
        }
        const bool linear = threadCount > 1 && to != MNN_DATA_FORMAT_NC4HW4 && group == 0;
        if (linear) {
            size_t first, last;
            layoutRange(spatialSize * channels, kLayoutCacheLine,
                        reinterpret_cast<uintptr_t>(dst) % kLayoutCacheLine, threadId, threadCount, first, last);
            if (from == MNN_DATA_FORMAT_NCHW || to == MNN_DATA_FORMAT_NCHW) {
                const size_t batchBytes = size_t(area) * channels;
                for (size_t b = first / batchBytes; first < last && b <= (last - 1) / batchBytes; ++b) {
                    const size_t a = std::max(first, b * batchBytes) - b * batchBytes;
                    const size_t z = std::min(last, (b + 1) * batchBytes) - b * batchBytes;
                    const bool unpack = to == MNN_DATA_FORMAT_NHWC;
                    convertPlanarOwned(static_cast<const uint8_t*>(src) + b * batchBytes,
                                       static_cast<uint8_t*>(dst) + b * batchBytes,
                                       unpack ? channels : area, unpack ? area : channels,
                                       unpack ? area : channels, a, z);
                }
                return NO_ERROR;
            }
            const size_t stride = to == MNN_DATA_FORMAT_NHWC ? channels : area;
            const size_t prefixEnd = std::min(last, (first + stride - 1) / stride * stride);
            convertLayoutFragment(static_cast<const uint8_t*>(src), static_cast<uint8_t*>(dst), from, to,
                                  batch, area, channels, first, prefixEnd);
            const size_t suffixStart = std::max(prefixEnd, last / stride * stride);
            convertLayoutFragment(static_cast<const uint8_t*>(src), static_cast<uint8_t*>(dst), from, to,
                                  batch, area, channels, suffixStart, last);
            start = prefixEnd / stride;
            end = suffixStart / stride;
        } else if (threadId != 0) {
            return NO_ERROR;
        }
        if (start >= end) {
            return NO_ERROR;
        }
        if ((from == MNN_DATA_FORMAT_NC4HW4 && to == MNN_DATA_FORMAT_NHWC) ||
            (from == MNN_DATA_FORMAT_NHWC && to == MNN_DATA_FORMAT_NC4HW4)) {
            convertNc4Nhwc(src, dst, spatialSize, channels, from == MNN_DATA_FORMAT_NC4HW4, start, end);
            return NO_ERROR;
        }
        if ((from == MNN_DATA_FORMAT_NCHW && to == MNN_DATA_FORMAT_NHWC) ||
            (from == MNN_DATA_FORMAT_NHWC && to == MNN_DATA_FORMAT_NCHW)) {
            const size_t batchBytes = size_t(area) * channels;
            const bool planarToNhwc = from == MNN_DATA_FORMAT_NCHW;
            for (size_t b = start / area; b <= (end - 1) / area; ++b) {
                const size_t offset = std::max(start, b * area) - b * area;
                const size_t count = std::min(end, (b + 1) * area) - b * area - offset;
                transposePlanarBytes(static_cast<const uint8_t*>(src) + b * batchBytes +
                                         offset * (planarToNhwc ? 1 : channels),
                                     static_cast<uint8_t*>(dst) + b * batchBytes +
                                         offset * (planarToNhwc ? channels : 1),
                                     planarToNhwc ? channels : count, planarToNhwc ? count : channels,
                                     planarToNhwc ? area : channels, planarToNhwc ? channels : area);
            }
            return NO_ERROR;
        }
    }
#endif
    return CPUTensorConverter::convert(src, dst, from, to, batch, area, channels, bytes, core, threadId, threadCount);
}
#endif
} // namespace KleidiAICommon
} // namespace MNN