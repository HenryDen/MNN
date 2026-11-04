//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "KleidiAIStaticInt8Kernels.hpp"

#ifdef MNN_KLEIDIAI_STATIC_INT8
#include "kai/ukernels/matmul/kai_matmul_pack_lhs.h"
#include "kai/ukernels/matmul/pack/kai_lhs_pack_x8p2vlx4_x8_sme.h"
#include "kai/ukernels/matmul/pack/kai_lhs_imatmul_pack_x8p2vlx4_x8p_sme.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme.h"
#include "kai/ukernels/matmul/pack/kai_rhs_imatmul_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme.h"
#include "kai/ukernels/matmul/matmul_clamp_qai8_qai8_qsi8cxp/kai_matmul_clamp_qai8_qai8_qsi8cxp2vlx4sb_1x16vl_sme2_dot.h"
#include "kai/ukernels/matmul/matmul_clamp_qai8_qai8p_qsi8cxp/kai_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa.h"
#include "kai/ukernels/matmul/imatmul_clamp_qai8_qai8p_qsi8cxp/kai_imatmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa.h"

namespace MNN {
namespace KleidiAIStaticInt8 {

Kernels::Kernels(Family selected) : family(selected) {
    switch (family) {
        case Family::W2:
            gemm = kai_matmul_clamp_qai8_qai8p8vsx4_qsu2cxp16vsx4sf32bi32_8vsx16vs_sme2_mopa();
            dot = kai_matmul_clamp_qai8_qai8_qsu2cxp16vsx4sf32bi32_1x64vs_sme2_dot();
            rhs = kai_matmul_pack_rhs_nxk_qsu2cxp16vsx4sf32bi32_qsu2cx_f32_i32_sme();
            break;
        case Family::W4:
            gemm = kai_matmul_clamp_qai8_qai8p8vsx4_qsi4cxp8vsx4sf32bi32_8vsx8vs_sme2_mopa();
            dot = kai_matmul_clamp_qai8_qai8_qsi4cxp8vsx4sf32bi32_1x64vs_sme2_dot();
            rhs = kai_matmul_pack_rhs_nxk_qsi4cxp8vsx4sf32bi32_qsi4cx_f32_i32_sme();
            break;
        case Family::W8:
            gemm = kai_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4bi32sf32_8vsx8vs_sme2_mopa();
            dot = kai_matmul_clamp_qai8_qai8_qsi8cxp4vsx4bi32sf32_1x32vs_sme2_dot();
            rhs = kai_matmul_pack_rhs_kxn_qsi8cxp4vsx4bi32sf32_qsi8_i32_f32_sme();
            break;
        default:
            mr = kai_get_mr_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa();
            nr = kai_get_nr_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa();
            return;
    }
    const kai_matmul_uker_config config = {};
    const kai_matmul_pack_rhs_uker_config packConfig = {};
    mr = gemm.get_step(&config).m;
    nr = rhs.get_step(&packConfig).n;
}

bool Kernels::indirect() const {
    return family == Family::IndirectSme2;
}
bool Kernels::modern() const {
    return gemm.run != nullptr;
}
bool Kernels::hasDot() const {
    return modern() || family == Family::Legacy;
}

size_t Kernels::rhsBytes(size_t n, size_t k, size_t kernelSize) const {
    size_t alignedK, alignedN, payload, bytes;
    const size_t divisor = family == Family::W2 ? 4 : (family == Family::W4 ? 2 : 1);
    if (!roundUp(k, divisor == 1 ? 4 : 32, alignedK) || !roundUp(n, nr, alignedN) ||
        !product(alignedK / divisor, kernelSize, payload) ||
        payload > size_t(std::numeric_limits<int>::max()) - 8 || !product(alignedN, payload + 8, bytes)) {
        return 0;
    }
    // Validate the checked formula against the upstream size API before allocating.
    size_t apiBytes;
    if (modern()) {
        const kai_matmul_pack_rhs_uker_config config = {};
        const kai_matmul_pack_rhs_uker_rhs_packed_dim_args shape = {n, k};
        const auto stride = rhs.get_rhs_packed_stride(&config, &shape);
        apiBytes = rhs.get_rhs_packed_size(&config, &shape, &stride);
    } else if (indirect()) {
        apiBytes = kai_get_rhs_packed_size_rhs_imatmul_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme(n, kernelSize, k);
    } else {
        apiBytes = kai_get_rhs_packed_size_rhs_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme(n, k);
    }
    return apiBytes == bytes ? bytes : 0;
}

bool Kernels::lhsBytes(size_t rows, size_t k, size_t kernelSize, size_t& bytes) const {
    size_t alignedM, alignedK;
    return roundUp(rows, mr, alignedM) && roundUp(k, 4, alignedK) &&
           product(alignedK, kernelSize, alignedK) && product(alignedM, alignedK, bytes);
}

void Kernels::packRhs(size_t n, size_t k, size_t kernelSize, const void* weights,
                      const int32_t* bias, const float* scale, void* packed) const {
    // Legacy MNN bias already folds input/output zero points; all KAI corrections stay zero.
    const int32_t zero = 0;
    const float one = 1.0f;
    if (modern()) {
        const kai_matmul_pack_rhs_uker_config config = {};
        const kai_matmul_pack_rhs_uker_rhs_dim_args shape = {n, k};
        const kai_matmul_pack_rhs_uker_rhs_packed_dim_args packedShape = {n, k};
        kai_matmul_pack_rhs_uker_args args = {};
        args.shape = {n, k};
        args.operand.rhs.ptr = weights;
        args.operand.rhs.stride = rhs.get_rhs_stride(&config, &shape);
        args.operand.rhs_packed.ptr = packed;
        args.operand.rhs_packed.stride = rhs.get_rhs_packed_stride(&config, &packedShape);
        args.operand.bias_n.ptr = bias;
        args.operand.k_sum_scale_global.ptr = &zero;
        args.operand.scale_n.ptr = scale;
        args.operand.scale_global.ptr = &one;
        rhs.run(&config, &args);
    } else {
        const kai_rhs_pack_qsi8cx_params params = {zero, one};
        if (indirect()) {
            kai_run_rhs_imatmul_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme(
                n, kernelSize, k, n, weights, bias, scale, packed, &params);
        } else {
            kai_run_rhs_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme(
                1, n, k, nr, 4, 1, n, weights, bias, scale, packed, 0, &params);
        }
    }
}

void Kernels::packLhs(size_t rows, size_t k, size_t kernelSize, const int8_t* input,
                      const void* const* pointers, const void* padding, void* packed) const {
    if (indirect()) {
        kai_run_lhs_imatmul_pack_x8p2vlx4_x8p_sme(rows, kernelSize, k, pointers, 0, padding, packed);
    } else if (family == Family::W8) {
        const auto packer = kai_matmul_pack_lhs_mxk_x8p4vsx4_x8_sme();
        const kai_matmul_pack_lhs_uker_config config = {};
        const kai_matmul_pack_lhs_uker_lhs_dim_args shape = {rows, k};
        const kai_matmul_pack_lhs_uker_lhs_packed_dim_args packedShape = {rows, k};
        kai_matmul_pack_lhs_uker_args args = {};
        args.shape = {rows, k};
        args.operand.lhs.ptr = input;
        args.operand.lhs.stride = packer.get_lhs_stride(&config, &shape);
        args.operand.lhs_packed.ptr = packed;
        args.operand.lhs_packed.stride = packer.get_lhs_packed_stride(&config, &packedShape);
        packer.run(&config, &args);
    } else {
        // The v1.32 static QSU2 kernels use a signed LUT, unlike dynamic QSU2 sum-based kernels.
        kai_run_lhs_pack_x8p2vlx4_x8_sme(rows, k, mr, 4, 1, 0, input, k, packed);
    }
}

void Kernels::run(bool useDot, size_t rows, size_t n, size_t k, size_t kernelSize,
                  const void* lhs, const void* rhsPacked, void* output, int32_t min, int32_t max) const {
    if (modern()) {
        const auto& kernel = useDot ? dot : gemm;
        const kai_matmul_uker_config config = {};
        const kai_matmul_uker_lhs_dim_args lhsShape = {rows, k};
        const kai_matmul_uker_rhs_dim_args rhsShape = {n, k};
        const int32_t zero = 0;
        kai_matmul_uker_args args = {};
        args.flags = KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP;
        args.shape = {rows, n, k};
        args.operand.lhs.ptr = lhs;
        args.operand.lhs.stride = kernel.get_lhs_stride(&config, &lhsShape);
        args.operand.rhs.ptr = rhsPacked;
        args.operand.rhs.stride = kernel.get_rhs_stride(&config, &rhsShape);
        args.operand.dst.ptr = output;
        args.operand.dst.stride.m = n;
        args.operand.bias.scale_bias_global.ptr = &zero;
        args.activation.clamp.min_ptr = &min;
        args.activation.clamp.max_ptr = &max;
        kernel.run(&config, &args);
        return;
    }
    const kai_matmul_requantize32_params params = {min, max, 0};
    if (indirect()) {
        kai_run_imatmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa(
            rows, n, kernelSize, k, lhs, rhsPacked, output, n, &params);
    } else {
        const auto fn = useDot ? kai_run_matmul_clamp_qai8_qai8_qsi8cxp2vlx4sb_1x16vl_sme2_dot
                              : kai_run_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa;
        fn(rows, n, k, lhs, rhsPacked, output, n, 1, &params);
    }
}

} // namespace KleidiAIStaticInt8
} // namespace MNN
#endif