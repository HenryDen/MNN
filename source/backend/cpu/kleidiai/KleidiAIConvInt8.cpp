//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#ifdef MNN_KLEIDIAI_ENABLED
#include "KleidiAIConvInt8.hpp"
#include "core/Macro.h"
#include "core/BufferAllocator.hpp"

#include <arm_neon.h>
#include <math.h>
#include <string.h>
#include "backend/cpu/CPUBackend.hpp"
#include "backend/cpu/CPURuntime.hpp"
#include "core/Concurrency.h"
#include "core/TensorUtils.hpp"
#include "backend/cpu/CPUTensorConvert.hpp"
#include <MNN/AutoTime.hpp>

// KleidiAI micro-kernel headers (int4 / int8 dynamic-quant matmul + packing).
// The symmetric per-channel int4 path is served by the asymmetric qsi8d32/qai4c32
// kernels below. The asym packer stores signed int4 (v-8), so the dequant is
// w = scale*(v-8) + zero; symmetric weights are exactly this with per-channel zero = 0.
// so no dedicated qai8dxp/qsi4cxp ukernels are needed here.
#include "kai_common.h"
#include "kai_lhs_quant_pack_qsi8d32pscalef32_f16_neon.h"
#include "kai_lhs_quant_pack_qsi8d32pscalef32_f32_neon.h"
#include "kai_rhs_pack_nxk_qai4c32p_qau4c32s0s1_f32_f32_f32_neon.h"
#include "kai_rhs_pack_nxk_qai4c32ps1s0nrx4_qau4c32s0s1_f32_f32_f32_neon.h"
#include "kai_matmul_clamp_f16_qsi8d32p1x8_qai4c32p4x8_1x4_neon_dotprod.h"
#include "kai_matmul_clamp_f16_qsi8d32p4x8_qai4c32p4x8_8x4_neon_i8mm.h"
#include "kai_matmul_clamp_f32_qsi8d32p1x8_qai4c32p4x8_1x4_neon_dotprod.h"
#include "kai_matmul_clamp_f32_qsi8d32p4x8_qai4c32p4x8_8x4_neon_i8mm.h"
#include "kai_matmul_clamp_f32_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa.h"
#include "kai_matmul_clamp_f32_qsi8d32p1x4_qai4c32p4vlx4_1x4vl_sme2_dot.h"
#include "kai_matmul_clamp_f16_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa.h"
#include "kai_matmul_clamp_f16_qsi8d32p1x4_qai4c32p4vlx4_1x4vl_sme2_dot.h"
#include "kai_lhs_pack_f16pmrx4_f32_neon.h"
#include "kai_matmul_clamp_f32_f16p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa.h"

#define QUANT_INFO_BYTES 4
namespace MNN {

// ===================================================================
// Static classification / gating (moved out of the former KleidiAI class).

KleidiAIConvInt8::KernelType KleidiAIConvInt8::getKernelType(size_t bits, bool bAsymmetric, size_t blockSize, size_t bytes) {
    // Only 4-bit dynamic-quant weights are accelerated today. The variant is picked from
    // symmetry, quant granularity (per-channel when blockSize == 0, else per-block) and the
    // activation precision (f32 when bytes == 4, f16 when bytes == 2). Anything else falls back.
    if (bits != 4) {
        return KernelType::KERNEL_TYPE_ERROR;
    }
    const bool perChannel = (blockSize == 0);
    if (bAsymmetric) {
        if (bytes == 4) {
            return perChannel ? KernelType::QI4_ASYM_PERCHANNEL_F32 : KernelType::QI4_ASYM_PERBLOCK_F32;
        }
        if (bytes == 2) {
            return perChannel ? KernelType::QI4_ASYM_PERCHANNEL_F16 : KernelType::QI4_ASYM_PERBLOCK_F16;
        }
        return KernelType::KERNEL_TYPE_ERROR;
    }
    // Symmetric: only per-channel f32 has a ukernel.
    if (perChannel && bytes == 4) {
        return KernelType::QI4_SYM_PERCHANNEL_F32;
    }
    return KernelType::KERNEL_TYPE_ERROR;
}

// Whether the running CPU provides the ukernels required by this KernelType.
static bool kaiKernelSupport(KleidiAIConvInt8::KernelType type) {
    auto cpu = MNNGetCPUInfo();
    bool hasKernel = cpu->sme2 || (cpu->dot && cpu->i8mm);
    switch (type) {
        case KleidiAIConvInt8::KernelType::QI4_SYM_PERCHANNEL_F32:
        case KleidiAIConvInt8::KernelType::QI4_ASYM_PERCHANNEL_F32:
        case KleidiAIConvInt8::KernelType::QI4_ASYM_PERBLOCK_F32:
        case KleidiAIConvInt8::KernelType::QI4_ASYM_PERCHANNEL_F16:
        case KleidiAIConvInt8::KernelType::QI4_ASYM_PERBLOCK_F16:
            return hasKernel;
        default:
            return false;
    }
}

bool KleidiAIConvInt8::isSupported(KernelType type, const Convolution2DCommon* common) {
    if (type == KernelType::KERNEL_TYPE_ERROR) {
        return false;
    }
    if (common->group() != 1) {
        return false;
    }
    if (type == KernelType::QI4_ASYM_PERCHANNEL_F32 || type == KernelType::QI4_ASYM_PERCHANNEL_F16
        || type == KernelType::QI8_ASYM_PERCHANNEL || type == KernelType::QI4_SYM_PERCHANNEL_F32) {
        // Symmetric per-channel reuses the asymmetric qsi8d32/qai4c32 kernels, which require
        // the K dimension to be a multiple of 32.
        if (common->inputCount() % 32 != 0) {
            return false;
        }
    }
    if (common->kernelX() == 1 && common->kernelY() == 1
        && common->padX() == 0 && common->padY() == 0
        && common->strideX() == 1 && common->strideY() == 1
        && common->dilateX() == 1 && common->dilateY() == 1) {
        return kaiKernelSupport(type);
    }
    return false;
}

size_t KleidiAIConvInt8::getVecNumPerThread(size_t totalVec, size_t totalThread, size_t minStep) {
    return kai_roundup((totalVec + totalThread - 1) / totalThread, minStep);
}

// ===================================================================
// Per-instance kernel parameter resolution and ukernel dispatch.

// ===================================================================
// Uniform-signature adapters over the concrete KleidiAI micro-kernels.
// Each adapter matches one KleidiAIConvInt8::Ukernel slot; `bl` is ignored by the channel-quant
// (qsi4cx / qai8dx) kernels that do not take it. All are bound once in configKernel().
namespace {

// The rhs/lhs "size" and "offset" getters are pure forwarders that differ only by the concrete
// kai function and whether the trailing granularity arg is sr (channel-quant) or bl (block-quant).
// Generate them from a single pattern to avoid a wall of near-identical one-liners.
//   DEFINE_RHS_INFO      : rhs size/offset, shape (idx, k, nr, kr, <sr|bl>).
//   DEFINE_LHS_INFO_CHNL : lhs size/offset for channel-quant kernels that take no bl.
//   DEFINE_LHS_INFO_BLK  : lhs size/offset for block-quant kernels that take bl (3rd arg).
#define DEFINE_RHS_INFO(NAME, KAIFN, LAST) \
    size_t NAME(size_t idx, size_t k, size_t nr, size_t kr, size_t sr, size_t bl) { \
        (void)sr; (void)bl; \
        return KAIFN(idx, k, nr, kr, LAST); \
    }
#define DEFINE_LHS_INFO_CHNL(NAME, KAIFN) \
    size_t NAME(size_t idx, size_t k, size_t bl, size_t mr, size_t kr, size_t sr) { \
        (void)bl; \
        return KAIFN(idx, k, mr, kr, sr); \
    }
#define DEFINE_LHS_INFO_BLK(NAME, KAIFN) \
    size_t NAME(size_t idx, size_t k, size_t bl, size_t mr, size_t kr, size_t sr) { \
        return KAIFN(idx, k, bl, mr, kr, sr); \
    }

// ---- rhs packed size ----
DEFINE_RHS_INFO(rhsSizeAsymSme2, kai_get_rhs_packed_size_rhs_pack_nxk_qai4c32ps1s0nrx4_qau4c32s0s1_f32_f32_f32_neon, bl)
DEFINE_RHS_INFO(rhsSizeAsymNeon, kai_get_rhs_packed_size_rhs_pack_nxk_qai4c32p_qau4c32s0s1_f32_f32_f32_neon,      bl)

// ---- rhs packed offset ----
DEFINE_RHS_INFO(rhsOffAsymSme2,  kai_get_rhs_packed_offset_rhs_pack_nxk_qai4c32ps1s0nrx4_qau4c32s0s1_f32_f32_f32_neon, bl)
DEFINE_RHS_INFO(rhsOffAsymNeon,  kai_get_rhs_packed_offset_rhs_pack_nxk_qai4c32p_qau4c32s0s1_f32_f32_f32_neon,    bl)

// ---- rhs pack ----
void rhsPackAsymSme2(size_t numGroups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, size_t bl,
                     const void* rhs, const void* scale, const void* zeroPoint, const void* bias, void* rhsPacked) {
    struct kai_rhs_pack_nxk_qai4c32p_params params;
    params.lhs_zero_point = 1;
    params.rhs_zero_point = 8;
    kai_run_rhs_pack_nxk_qai4c32ps1s0nrx4_qau4c32s0s1_f32_f32_f32_neon(numGroups, n, k, nr, kr, sr, bl,
        (const uint8_t*)rhs, zeroPoint, bias, scale, rhsPacked, 0, &params);
}
void rhsPackAsymNeon(size_t numGroups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, size_t bl,
                     const void* rhs, const void* scale, const void* zeroPoint, const void* bias, void* rhsPacked) {
    struct kai_rhs_pack_nxk_qai4c32p_params params;
    params.lhs_zero_point = 1;
    params.rhs_zero_point = 8;
    kai_run_rhs_pack_nxk_qai4c32p_qau4c32s0s1_f32_f32_f32_neon(numGroups, n, k, nr, kr, sr, bl,
        (const uint8_t*)rhs, zeroPoint, bias, scale, rhsPacked, 0, &params);
}

// ---- lhs quanted packed size ----
DEFINE_LHS_INFO_BLK(lhsSizeAsymF32,  kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32pscalef32_f32_neon)
DEFINE_LHS_INFO_BLK(lhsSizeAsymF16,  kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32pscalef32_f16_neon)
size_t lhsSizeDirectF32(size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr) {
    if (m == 1) {
        return kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32pscalef32_f32_neon(m, k, bl, 1, kr, sr);
    }
    return kai_get_lhs_packed_size_lhs_pack_f16pmrx4_f32_neon(m, k, bl, mr, kr, sr);
}

// ---- lhs quanted packed offset ----
DEFINE_LHS_INFO_BLK(lhsOffAsymF32,   kai_get_lhs_packed_offset_lhs_quant_pack_qsi8d32pscalef32_f32_neon)
DEFINE_LHS_INFO_BLK(lhsOffAsymF16,   kai_get_lhs_packed_offset_lhs_quant_pack_qsi8d32pscalef32_f16_neon)
DEFINE_LHS_INFO_BLK(lhsOffDirectF32,  kai_get_lhs_packed_offset_lhs_pack_f16pmrx4_f32_neon)

// ---- lhs quant + pack ----
void lhsPackAsymF32(size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr, const void* lhs, void* out) {
    kai_run_lhs_quant_pack_qsi8d32pscalef32_f32_neon(m, k, bl, mr, kr, sr, 0, (const float*)lhs, k * sizeof(float), out);
}
void lhsPackAsymF16(size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr, const void* lhs, void* out) {
    kai_run_lhs_quant_pack_qsi8d32pscalef32_f16_neon(m, k, bl, mr, kr, sr, 0, (const __fp16*)lhs, k * sizeof(__fp16), out);
}
void lhsPackDirectF32(size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr, const void* lhs, void* out) {
    if (m == 1) {
        lhsPackAsymF32(m, k, bl, 1, kr, sr, lhs, out);
        return;
    }
    kai_run_lhs_pack_f16pmrx4_f32_neon(m, k, bl, mr, kr, sr, 0, lhs, k * sizeof(float), out);
}

// ---- matmul (GEMV when m == 1, GEMM otherwise) ----
void matmulAsymF32Sme2(size_t m, size_t n, size_t k, size_t bl, const void* lhs, const void* rhs, void* dst,
                       size_t sr, size_t sc, float mn, float mx) {
    if (m == 1) {
        kai_run_matmul_clamp_f32_qsi8d32p1x4_qai4c32p4vlx4_1x4vl_sme2_dot(m, n, k, bl, lhs, rhs, (float*)dst, sr, sc, mn, mx);
    } else {
        kai_run_matmul_clamp_f32_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa(m, n, k, bl, lhs, rhs, (float*)dst, sr, sc, mn, mx);
    }
}
void matmulAsymF32Neon(size_t m, size_t n, size_t k, size_t bl, const void* lhs, const void* rhs, void* dst,
                       size_t sr, size_t sc, float mn, float mx) {
    if (m == 1) {
        kai_run_matmul_clamp_f32_qsi8d32p1x8_qai4c32p4x8_1x4_neon_dotprod(m, n, k, bl, lhs, rhs, (float*)dst, sr, sc, mn, mx);
    } else {
        kai_run_matmul_clamp_f32_qsi8d32p4x8_qai4c32p4x8_8x4_neon_i8mm(m, n, k, bl, lhs, rhs, (float*)dst, sr, sc, mn, mx);
    }
}
void matmulAsymF16Sme2(size_t m, size_t n, size_t k, size_t bl, const void* lhs, const void* rhs, void* dst,
                       size_t sr, size_t sc, float mn, float mx) {
    if (m == 1) {
        kai_run_matmul_clamp_f16_qsi8d32p1x4_qai4c32p4vlx4_1x4vl_sme2_dot(m, n, k, bl, lhs, rhs, (float*)dst, sr, sc, mn, mx);
    } else {
        kai_run_matmul_clamp_f16_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa(m, n, k, bl, lhs, rhs, (float*)dst, sr, sc, mn, mx);
    }
}
void matmulAsymF16Neon(size_t m, size_t n, size_t k, size_t bl, const void* lhs, const void* rhs, void* dst,
                       size_t sr, size_t sc, float mn, float mx) {
    if (m == 1) {
        kai_run_matmul_clamp_f16_qsi8d32p1x8_qai4c32p4x8_1x4_neon_dotprod(m, n, k, bl, lhs, rhs, (float*)dst, sr, sc, mn, mx);
    } else {
        kai_run_matmul_clamp_f16_qsi8d32p4x8_qai4c32p4x8_8x4_neon_i8mm(m, n, k, bl, lhs, rhs, (float*)dst, sr, sc, mn, mx);
    }
}
void matmulDirectF32Sme2(size_t m, size_t n, size_t k, size_t bl, const void* lhs, const void* rhs, void* dst,
                         size_t sr, size_t sc, float mn, float mx) {
    if (m == 1) {
        kai_run_matmul_clamp_f32_qsi8d32p1x4_qai4c32p4vlx4_1x4vl_sme2_dot(
            m, n, k, bl, lhs, rhs, (float*)dst, sr, sc, mn, mx);
        return;
    }
    kai_run_matmul_clamp_f32_f16p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa(
        m, n, k, bl, lhs, rhs, (float*)dst, sr, sc, mn, mx);
}

#undef DEFINE_RHS_INFO
#undef DEFINE_LHS_INFO_CHNL
#undef DEFINE_LHS_INFO_BLK

} // namespace

// ===================================================================
// Per-instance kernel parameter resolution and ukernel dispatch.

void KleidiAIConvInt8::configKernel() {
    auto cpu = MNNGetCPUInfo();
    mSme2 = cpu->sme2;
    mDot  = cpu->dot;
    mI8mm = cpu->i8mm;
    mHybrid = false;
    mChnlQuant = (mKernelType == KernelType::QI4_SYM_PERCHANNEL_F32
                  || mKernelType == KernelType::QI4_ASYM_PERCHANNEL_F32
                  || mKernelType == KernelType::QI4_ASYM_PERCHANNEL_F16);

    // Slot fillers. Each binds one (KernelParam, Ukernel) pair to a concrete kernel family so that
    // both the primary (SME) and, when hybrid, the secondary (NEON) slot are configured identically.
    auto fillSmeF32 = [](KernelParam& p, Ukernel& u) {
        u.lhsPackedSize   = lhsSizeAsymF32;
        u.lhsPackedOffset = lhsOffAsymF32;
        u.runLhsQuantPack = lhsPackAsymF32;
        p.mKaiMstepGemv = 1;
        p.mKaiMstepGemm = kai_get_m_step_matmul_clamp_f32_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa();
        p.mKaiNStep     = kai_get_n_step_matmul_clamp_f32_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa();
        p.mKaiMrGemv    = 1;
        p.mKaiMrGemm    = kai_get_mr_matmul_clamp_f32_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa();
        p.mKaiNr        = kai_get_nr_matmul_clamp_f32_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa();
        p.mKaiKr        = kai_get_kr_matmul_clamp_f32_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa();
        p.mKaiSr        = kai_get_sr_matmul_clamp_f32_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa();
        u.rhsPackedSize   = rhsSizeAsymSme2;
        u.rhsPackedOffset = rhsOffAsymSme2;
        u.runRhsPack      = rhsPackAsymSme2;
        u.matmul          = matmulAsymF32Sme2;
    };
    auto fillNeonF32 = [](KernelParam& p, Ukernel& u) {
        u.lhsPackedSize   = lhsSizeAsymF32;
        u.lhsPackedOffset = lhsOffAsymF32;
        u.runLhsQuantPack = lhsPackAsymF32;
        p.mKaiMstepGemv = 1;
        p.mKaiMstepGemm = 8;
        p.mKaiNStep     = 4;
        p.mKaiMrGemv    = 1;
        p.mKaiMrGemm    = 4;
        p.mKaiNr        = 4;
        p.mKaiKr        = 16;
        p.mKaiSr        = 2;
        u.rhsPackedSize   = rhsSizeAsymNeon;
        u.rhsPackedOffset = rhsOffAsymNeon;
        u.runRhsPack      = rhsPackAsymNeon;
        u.matmul          = matmulAsymF32Neon;
    };
    auto fillSmeDirectF32 = [](KernelParam& p, Ukernel& u) {
        u.lhsPackedSize   = lhsSizeDirectF32;
        u.lhsPackedOffset = lhsOffDirectF32;
        u.runLhsQuantPack = lhsPackDirectF32;
        p.mKaiMstepGemv = 1;
        p.mKaiMstepGemm = kai_get_m_step_matmul_clamp_f32_f16p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa();
        p.mKaiNStep     = kai_get_n_step_matmul_clamp_f32_f16p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa();
        p.mKaiMrGemv    = 1;
        p.mKaiMrGemm    = kai_get_mr_matmul_clamp_f32_f16p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa();
        p.mKaiNr        = kai_get_nr_matmul_clamp_f32_f16p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa();
        p.mKaiKr        = kai_get_kr_matmul_clamp_f32_f16p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa();
        p.mKaiSr        = kai_get_sr_matmul_clamp_f32_f16p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa();
        u.rhsPackedSize   = rhsSizeAsymSme2;
        u.rhsPackedOffset = rhsOffAsymSme2;
        u.runRhsPack      = rhsPackAsymSme2;
        u.matmul          = matmulDirectF32Sme2;
    };
    auto fillSmeF16 = [](KernelParam& p, Ukernel& u) {
        u.lhsPackedSize   = lhsSizeAsymF16;
        u.lhsPackedOffset = lhsOffAsymF16;
        u.runLhsQuantPack = lhsPackAsymF16;
        p.mKaiMstepGemv = 1;
        p.mKaiMstepGemm = kai_get_m_step_matmul_clamp_f16_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa();
        p.mKaiNStep     = kai_get_n_step_matmul_clamp_f16_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa();
        p.mKaiMrGemv    = 1;
        p.mKaiMrGemm    = kai_get_mr_matmul_clamp_f16_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa();
        p.mKaiNr        = kai_get_nr_matmul_clamp_f16_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa();
        p.mKaiKr        = kai_get_kr_matmul_clamp_f16_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa();
        p.mKaiSr        = kai_get_sr_matmul_clamp_f16_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa();
        u.rhsPackedSize   = rhsSizeAsymSme2;
        u.rhsPackedOffset = rhsOffAsymSme2;
        u.runRhsPack      = rhsPackAsymSme2;
        u.matmul          = matmulAsymF16Sme2;
    };
    auto fillNeonF16 = [](KernelParam& p, Ukernel& u) {
        u.lhsPackedSize   = lhsSizeAsymF16;
        u.lhsPackedOffset = lhsOffAsymF16;
        u.runLhsQuantPack = lhsPackAsymF16;
        p.mKaiMstepGemv = 1;
        p.mKaiMstepGemm = 8;
        p.mKaiNStep     = 4;
        p.mKaiMrGemv    = 1;
        p.mKaiMrGemm    = 4;
        p.mKaiNr        = 4;
        p.mKaiKr        = 16;
        p.mKaiSr        = 2;
        u.rhsPackedSize   = rhsSizeAsymNeon;
        u.rhsPackedOffset = rhsOffAsymNeon;
        u.runRhsPack      = rhsPackAsymNeon;
        u.matmul          = matmulAsymF16Neon;
    };

    switch (mKernelType) {
        // Symmetric per-channel int4 is served by the asymmetric qsi8d32/qai4c32 kernels:
        // the asym packer stores signed int4 (v-8), so w = scale*(v-8) + zero; symmetric is
        // exactly this with per-channel zero = 0. The symmetric scale/zero are synthesized in
        // the constructor.
        case KernelType::QI4_SYM_PERCHANNEL_F32:
        case KernelType::QI4_ASYM_PERCHANNEL_F32:
            if (mSme2) {
                fillSmeF32(mParam, mUkernel);
                if (mDot && mI8mm) {
                    // Also configure the NEON slot so the two can run concurrently (SME + NEON).
                    fillNeonF32(mParamNeon, mUkernelNeon);
                    mHybrid = true;
                }
            } else if (mDot && mI8mm) {
                fillNeonF32(mParam, mUkernel);
            }
            break;
        case KernelType::QI4_ASYM_PERBLOCK_F32:
            if (mSme2) {
                // The direct kernel converts FP32 activations to packed FP16 and accumulates in
                // FP32 ZA, avoiding dynamic INT8 quantization and per-block requantization.
                fillSmeDirectF32(mParam, mUkernel);
            } else if (mDot && mI8mm) {
                fillNeonF32(mParam, mUkernel);
            }
            break;
        case KernelType::QI4_ASYM_PERCHANNEL_F16:
        case KernelType::QI4_ASYM_PERBLOCK_F16:
            if (mSme2) {
                fillSmeF16(mParam, mUkernel);
                if (mDot && mI8mm) {
                    fillNeonF16(mParamNeon, mUkernelNeon);
                    mHybrid = true;
                }
            } else if (mDot && mI8mm) {
                fillNeonF16(mParam, mUkernel);
            }
            break;
        default:
            break;
    }
}

size_t KleidiAIConvInt8::getRhsPackedSize(const Ukernel& u, const KernelParam& p, size_t n, size_t k, size_t bl) const {
    return u.rhsPackedSize(n, k, getNr(p), getKr(p), getSr(p), mChnlQuant ? k : bl);
}

size_t KleidiAIConvInt8::getRhsPackedOffset(const Ukernel& u, const KernelParam& p, size_t nIdx, size_t k, size_t bl) const {
    if (nIdx == 0) {
        return 0;
    }
    return u.rhsPackedOffset(nIdx, k, getNr(p), getKr(p), getSr(p), mChnlQuant ? k : bl);
}

void KleidiAIConvInt8::runRhsPack(const Ukernel& u, const KernelParam& p, size_t numGroups, size_t n, size_t k, size_t bl,
                                  const void* rhs, const void* scale, const void* zeroPoint, const void* bias,
                                  void* rhsPacked) const {
    u.runRhsPack(numGroups, n, k, getNr(p), getKr(p), getSr(p), mChnlQuant ? k : bl,
                 rhs, scale, zeroPoint, bias, rhsPacked);
}

size_t KleidiAIConvInt8::getLhsQuantedPackedSize(const Ukernel& u, const KernelParam& p, size_t m, size_t k, size_t bl) const {
    return u.lhsPackedSize(m, k, mChnlQuant ? k : bl, getMr(p, m), getKr(p), getSr(p));
}

size_t KleidiAIConvInt8::getLhsQuantedPackedOffset(const Ukernel& u, const KernelParam& p, size_t m, size_t mIdx, size_t k, size_t bl) const {
    if (mIdx == 0) {
        return 0;
    }
    return u.lhsPackedOffset(mIdx, k, mChnlQuant ? k : bl, getMr(p, m), getKr(p), getSr(p));
}

void KleidiAIConvInt8::runLhsQuantPack(const Ukernel& u, const KernelParam& p, size_t m, size_t k, size_t bl, size_t mr,
                                       const void* lhs, void* lhsQuantedPacked) const {
    u.runLhsQuantPack(m, k, mChnlQuant ? k : bl, mr, getKr(p), getSr(p), lhs, lhsQuantedPacked);
}

void KleidiAIConvInt8::runMatmul(const Ukernel& u, const KernelParam& p, size_t m, size_t n, size_t k, size_t bl,
                                 const void* lhsPacked, const void* rhsPacked, void* dst,
                                 size_t dstStrideRow, size_t dstStrideCol,
                                 const float scalarMax, const float scalarMin) const {
    (void)p;
    u.matmul(m, n, k, mChnlQuant ? k : bl, lhsPacked, rhsPacked, dst,
             dstStrideRow, dstStrideCol, scalarMin, scalarMax);
}

KleidiAIConvInt8::KleidiAIConvInt8(Backend* backend, const Op* op, std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon, bool isDynamicQuant,
    KernelType kernelType, int32_t blockNum)
    : CPUConvolution(op->main_as_Convolution2D()->common(), backend), mKernelType(kernelType), mBlockNum(blockNum) {
    // Resolve CPU features and kernel packing parameters for this KernelType.
    configKernel();

    // convolution info
    auto convOp = op->main_as_Convolution2D();
    int oc = convOp->common()->outputCount();
    int ic = convOp->common()->inputCount();

    // backend info
    auto core = static_cast<CPUBackend*>(backend)->functions();
    int pack = core->pack;

    // compute info
    int ocUp4 = ROUND_UP(oc, pack);
    int scaleSize = ocUp4 * mBlockNum;

    // kleidia info
    bool bFP16 = core->bytes == 2 ? true : false;
    bool bAsym = quanCommon->asymmetric;
    size_t blkSize = mBlockNum == 1 ? 0 : ic / mBlockNum;

    AutoStorage<int8_t> reorderedQuantInfo;
    reorderedQuantInfo.reset(2 * scaleSize * QUANT_INFO_BYTES + oc * QUANT_INFO_BYTES);
    if (reorderedQuantInfo.get() == nullptr) {
        MNN_ERROR("Memory not enough\n");
        return;
    }

    // Prepare bias (needed by every path) and, for the symmetric path, scale/zero.
    // The asymmetric path fills scale/zero below in the ukernel-specific linear layout,
    // so we intentionally skip them here to avoid computing them twice with different layouts.
    {
        int outputCount = convOp->common()->outputCount();
        auto quanInfoPtr = quanCommon->alpha.get();
        auto scalePtr = reinterpret_cast<float*>(reorderedQuantInfo.get());
        auto zeroPtr = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(scalePtr) + scaleSize * QUANT_INFO_BYTES);
        auto biasPtr = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(zeroPtr) + scaleSize * QUANT_INFO_BYTES);
        if (!quanCommon->asymmetric) {
            // Symmetric weights routed through the asymmetric ukernel: the packer stores signed
            // int4 (v-8), so w = scale*(v-8) + zero. Symmetric is exactly scale*(v-8), i.e. zero = 0.
            for (int i = 0; i < blockNum; ++i) {
                auto dstScale = scalePtr + i * ocUp4;
                auto dstZero  = zeroPtr + i * ocUp4;
                for (int j = 0; j < outputCount; ++j) {
                    int scaleIndex = j * blockNum + i;
                    dstScale[j] = quanInfoPtr[scaleIndex];
                    dstZero[j] = 0.f;
                }
            }
        }
        ::memcpy(biasPtr, convOp->bias()->data(), oc * QUANT_INFO_BYTES);
    }

    int n = oc;
    int k = ic;
    int packedWeightSize = getRhsPackedSize(n, k, blkSize);

    //Alloc packed weight tensor.
    mWeightInt8.reset(Tensor::createDevice<uint8_t>({packedWeightSize}));
    bool success = backend->onAcquireBuffer(mWeightInt8.get(), Backend::STATIC);

    if (!success) {
        MNN_ERROR("Out of static memory!\n");
        return;
    }

    size_t paraNum = scaleSize;
    float *scalePtr = reinterpret_cast<float*>(reorderedQuantInfo.get());
    float *zeroPtr = reinterpret_cast<float*>(reorderedQuantInfo.get()) + paraNum;
    float *biasPtr = reinterpret_cast<float*>(reorderedQuantInfo.get()) + 2 * paraNum;
    //Reload some parameters to fit ukernels' layout.
    auto quanInfoPtr = quanCommon->alpha.get();
    auto alphaSize = quanCommon->alpha.size();
    if(bAsym) {
        for(int i = 0; i < paraNum; i++) {
            if(i*2 >= alphaSize){
                zeroPtr[i] = 0;
                scalePtr[i] = 0;
            }
            else{
                zeroPtr[i] = quanInfoPtr[i * 2];
                scalePtr[i] = quanInfoPtr[i * 2 + 1];
            }
        }
    } else {
        if(blkSize != 0) {
            memcpy(scalePtr, (uint8_t*)quanInfoPtr, paraNum * sizeof(float));
        }
    }

    //Run rhs pack.
    auto weightPackedData = mWeightInt8->host<uint8_t>();
    runRhsPack(1, n, k, blkSize,
               (uint8_t*)quanCommon->weight.get(),
               (const void*)scalePtr, (const void*)zeroPtr, (const void*)biasPtr,
               weightPackedData);

    if (mHybrid) {
        // Pack a second copy of the weights in the NEON slot layout so the NEON kernels can run
        // concurrently with the SME kernel on the remaining threads. Same scale/zero/bias, but a
        // different packed layout, hence a separate static buffer (~2x weight memory).
        int packedWeightSizeNeon = getRhsPackedSize(mUkernelNeon, mParamNeon, n, k, blkSize);
        mWeightInt8Neon.reset(Tensor::createDevice<uint8_t>({packedWeightSizeNeon}));
        bool successNeon = backend->onAcquireBuffer(mWeightInt8Neon.get(), Backend::STATIC);
        if (!successNeon) {
            MNN_ERROR("Out of static memory!\n");
            return;
        }
        runRhsPack(mUkernelNeon, mParamNeon, 1, n, k, blkSize,
                   (uint8_t*)quanCommon->weight.get(),
                   (const void*)scalePtr, (const void*)zeroPtr, (const void*)biasPtr,
                   mWeightInt8Neon->host<uint8_t>());
    }
    return;
}


KleidiAIConvInt8::KleidiAIConvInt8(Backend* backend, const Op* op, const KleidiAIConvInt8& exe)
    : CPUConvolution(op->main_as_Convolution2D()->common(), backend),
    mWeightInt8(exe.mWeightInt8), mTempIm2ColBuffer(exe.mTempIm2ColBuffer),
    mWeightInt8Neon(exe.mWeightInt8Neon),
    mKernelType(exe.mKernelType), mBlockNum(exe.mBlockNum) {
    configKernel();
}

KleidiAIConvInt8::~KleidiAIConvInt8() {
    // Do nothing
}

bool KleidiAIConvInt8::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (nullptr == dst) {
        return true;
    }
    auto exe = new KleidiAIConvInt8(bn, op, *this);
    if (!exe->valid()) {
        return false;
    }
    *dst = exe;
    return true;
}

// need
ErrorCode KleidiAIConvInt8::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    // Initialize.
    auto input  = inputs[0];
    auto output = outputs[0];
    auto core =static_cast<CPUBackend*>(backend())->functions();
    auto b = backend();

    const size_t m = inputs[0]->batch() * inputs[0]->width() * inputs[0]->height(); //lhs vector number.
    const size_t n = outputs[0]->channel(); //rhs vector number.
    const size_t k = inputs[0]->channel(); //vector size.
    const size_t blkSize = mBlockNum == 1 ? 0 : k / mBlockNum;

    auto inputOriginFmt = TensorUtils::getDescribe(inputs[0])->dimensionFormat;
    auto outputOriginFmt = TensorUtils::getDescribe(outputs[0])->dimensionFormat;
    halide_type_t dataType = core->bytes == 2 ? halide_type_of<int16_t>() : halide_type_of<float>();

    if(inputOriginFmt != MNN_DATA_FORMAT_NHWC){
        mInputConvertBuffer.reset(Tensor::createDevice(std::vector<int>{input->batch(), input->height(), input->width(), input->channel()}, dataType, Tensor::DimensionType::TENSORFLOW));
        mValid = b->onAcquireBuffer(mInputConvertBuffer.get(), Backend::DYNAMIC);
        if (!mValid) {
            MNN_ERROR("Out of dynamic memory!\n");
            return OUT_OF_MEMORY;
        }
    }
    if (outputOriginFmt != MNN_DATA_FORMAT_NHWC){
        mOutputConvertBuffer.reset(Tensor::createDevice(std::vector<int>{output->batch(), output->height(), output->width(), output->channel()}, dataType, Tensor::DimensionType::TENSORFLOW));
        mValid = b->onAcquireBuffer(mOutputConvertBuffer.get(), Backend::DYNAMIC);
        if (!mValid) {
            MNN_ERROR("Out of dynamic memory!\n");
            return OUT_OF_MEMORY;
        }
    }

    int packedSize = getLhsQuantedPackedSize(m, k, blkSize);
    int elementSize = core->bytes;

    //Split mTempIm2ColBuffer as two parts for linear/tile transfer:
    //Part0: Lhs_packed.
    //Part1: Lhs/Dst before transfer.
    mTempIm2ColBuffer.reset(Tensor::createDevice<int8_t>({packedSize}));
    bool success = backend()->onAcquireBuffer(mTempIm2ColBuffer.get(), Backend::DYNAMIC);
    if (!success) {
        MNN_ERROR("Out of dynamic memory!\n");
        return OUT_OF_MEMORY;
    }

    if (mHybrid) {
        // The NEON slot packs lhs with a different mr, so it needs its own packed buffer.
        int packedSizeNeon = getLhsQuantedPackedSize(mUkernelNeon, mParamNeon, m, k, blkSize);
        mTempIm2ColBufferNeon.reset(Tensor::createDevice<int8_t>({packedSizeNeon}));
        bool successNeon = backend()->onAcquireBuffer(mTempIm2ColBufferNeon.get(), Backend::DYNAMIC);
        if (!successNeon) {
            MNN_ERROR("Out of dynamic memory!\n");
            return OUT_OF_MEMORY;
        }
    }

    backend()->onReleaseBuffer(mTempIm2ColBuffer.get(), Backend::DYNAMIC);
    if (mHybrid) {
        backend()->onReleaseBuffer(mTempIm2ColBufferNeon.get(), Backend::DYNAMIC);
    }

    if(inputOriginFmt != MNN_DATA_FORMAT_NHWC){
        b->onReleaseBuffer(mInputConvertBuffer.get(), Backend::DYNAMIC);
    }
    if (outputOriginFmt != MNN_DATA_FORMAT_NHWC){
        b->onReleaseBuffer(mOutputConvertBuffer.get(), Backend::DYNAMIC);
    }
    return NO_ERROR;
}

// ---------------------------------------------------------------------------
// Matmul latency cost model (calibrated on Apple M4, single core, fp16 lhs).
//
// Analytic estimate of the *matmul-only* latency in microseconds for one engine
// slot, derived from the instruction counts of the SME2 MOPA int4 kernel
// (kai_matmul_clamp_f16_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa) and fit on
// a single-thread stride-16 MNK sweep. Only used to balance the hybrid SME/NEON
// column split, so only the SME-vs-NEON *ratio* matters; the absolute numbers
// are M4-specific.
//
//   The MOPA kernel nests: N loop (step NR=64) x M loop (step MR=16) x K loop.
//   Per (M,N) tile it issues K `smopa` and one dequant/store epilogue. Hence the
//   dominant instruction counts are, with mTile=ceil(M/16), nPanel=ceil(N/64):
//       T     = mTile * nPanel                (number of MOPA tiles)
//       smopa = T * Kpad                       (Kpad = roundup(K,32))
//   giving the per-channel model
//       t = t0 + a*T + b*T*Kpad
//   plus a K-independent narrow-store penalty for a single non-full N panel
//   (nPanel==1, N<64), which is measurably slower per row:
//       t += d * mTile * (NR - N)/NR
//   This fits the sweep to ~2% MAPE (N>=64: ~1.3%). Per-block adds one extra
//   dequant/rescale contribution per K block, proportional to T*(K/bl). M==1
//   uses a separate GEMV dot-kernel model with its own per-block term. F16/F32
//   and SME/NEON use separate coefficient sets because they bind different KAI
//   micro-kernels (MOPA/DOT vs i8mm/dotprod).
// ---------------------------------------------------------------------------
static bool kaiKernelIsF16(KleidiAIConvInt8::KernelType type) {
    return type == KleidiAIConvInt8::KernelType::QI4_ASYM_PERCHANNEL_F16
        || type == KleidiAIConvInt8::KernelType::QI4_ASYM_PERBLOCK_F16;
}

static double kaiEstimateSmeUs(KleidiAIConvInt8::KernelType type, size_t m, size_t nCols, size_t k, size_t blkSize) {
    if (m == 0 || nCols == 0) {
        return 0.0;
    }
    bool perBlock = (blkSize != 0);
    const int MR = 16, NR = 64;
    int nPanel = (int)((nCols + NR - 1) / NR);
    bool isF16 = kaiKernelIsF16(type);
    if (m == 1) {                                       // GEMV fast path (dot kernel)
        if (perBlock) {
            if (!isF16) {
                return 0.149284 + (double)nPanel * (0.00504713 + 3.0225e-4 * (double)k
                                  + 0.00578995 * (double)k / (double)blkSize);
            }
            return 0.083925 + (double)nPanel * (0.00565402 + 2.7488e-4 * (double)k
                              + 0.00773111 * (double)k / (double)blkSize);
        }
        if (!isF16) {
            return (double)nPanel * (0.047204 + 3.31259e-4 * (double)k);
        }
        return nPanel * (0.0732 + 2.539e-4 * (double)k);
    }
    int mTile = (int)((m + MR - 1) / MR);
    double T = (double)mTile * (double)nPanel;
    double kpad = (double)(((k + 31) / 32) * 32);
    double singleNarrow = 0.0;
    if (nPanel == 1 && nCols < (size_t)NR) {
        singleNarrow = (double)mTile * (double)(NR - (int)nCols) / (double)NR;
    }
    if (!isF16) {
        if (!perBlock) {
            return 0.272156 + 0.258691 * T + 5.210733e-04 * T * kpad + 0.558753 * singleNarrow;
        }
        return 0.397107 + 0.0778586 * T + 4.391517e-04 * T * kpad
               + 0.186019 * T * (double)k / (double)blkSize + 0.59724 * singleNarrow;
    }
    if (!perBlock) {                                    // per-channel (blkSize == 0)
        // Instruction-count model of the MOPA kernel (see comment above).
        double t = 0.3797 + 0.29089 * T + 4.836073e-04 * T * kpad;
        t += 0.43818 * singleNarrow;                    // single narrow panel penalty
        return t;
    }
    // per-block (blkSize > 0): same MOPA count plus one dequant/rescale term per K block.
    return 0.4685 + 0.09406 * T + 3.682232e-04 * T * kpad
           + 0.19966 * T * (double)k / (double)blkSize;
}

static double kaiEstimateNeonUs(KleidiAIConvInt8::KernelType type, size_t m, size_t nCols, size_t k, size_t blkSize) {
    if (m == 0 || nCols == 0) {
        return 0.0;
    }
    bool perBlock = (blkSize != 0);
    bool isF16 = kaiKernelIsF16(type);
    const int MR = 8, NR = 4;
    int nPanel = (int)((nCols + NR - 1) / NR);
    if (m == 1) {                                       // GEMV dotprod kernel
        if (perBlock) {
            if (isF16) {
                return 0.0300534 + (double)nPanel * (3.04287e-4 + 4.384333e-5 * (double)k
                                  - 1.019584e-4 * (double)k / (double)blkSize);
            }
            return 0.0289183 + (double)nPanel * (3.53137e-4 + 4.364133e-5 * (double)k
                              - 7.037887e-5 * (double)k / (double)blkSize);
        }
        if (isF16) {
            return 0.0369377 + (double)nPanel * (-0.0029569 + 6.484345e-5 * (double)k);
        }
        return 0.0381022 + (double)nPanel * (-0.00302672 + 6.469778e-5 * (double)k);
    }
    int mTile = (int)((m + MR - 1) / MR);
    double T = (double)mTile * (double)nPanel;
    double mac = (double)m * (double)nCols * (double)k;
    if (perBlock) {                                     // GEMM i8mm kernel, block-quant rhs
        if (isF16) {
            return -0.00574413 + 0.00306348 * T + 4.992718e-6 * mac
                   + 7.96544e-4 * T * (double)k / (double)blkSize;
        }
        return -0.00475343 + 0.00235611 * T + 4.996617e-6 * mac
               + 9.35294e-4 * T * (double)k / (double)blkSize;
    }
    if (isF16) {                                        // GEMM i8mm kernel, channel-quant rhs
        return 0.0173908 + 0.00132875 * T + 5.538539e-6 * mac;
    }
    return 0.022823 + 0.000806611 * T + 5.522018e-6 * mac;
}

ErrorCode KleidiAIConvInt8::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    const auto input = inputs[0];
    auto output      = outputs[0];
    auto core = static_cast<CPUBackend*>(backend())->functions();

    // Initialize for convert
    auto inputDes = TensorUtils::getDescribe(inputs[0]);
    auto outputDes = TensorUtils::getDescribe(outputs[0]);
    auto b = backend();
    halide_type_t dataType = core->bytes == 2 ? halide_type_of<int16_t>() : halide_type_of<float>();

    const size_t m = input->batch() * input->width() * input->height(); //lhs vector number.
    const size_t n = output->channel(); //rhs vector number.
    const size_t k = input->channel(); //vector size.
    const size_t blkSize = mBlockNum == 1 ? 0 : k / mBlockNum;

    size_t elementSize = core->bytes;

    auto lhs = input->host<uint8_t>();
    int threadNum = static_cast<CPUBackend*>(backend())->threadNumber();

    if(inputDes->dimensionFormat != MNN_DATA_FORMAT_NHWC) {
        // Convert input to NHWC format.
        MNN_CONCURRENCY_BEGIN(tId, threadNum) {
            CPUTensorConverter::convert(input, mInputConvertBuffer.get(), core, tId, threadNum);
        };
        MNN_CONCURRENCY_END();
        lhs = mInputConvertBuffer->host<uint8_t>();
    }

    // Dynamic-quant + pack lhs into `out` using the given kernel slot. Splits the M dimension over
    // the thread pool (single call for the GEMV m == 1 case).
    auto packLhs = [&](const Ukernel& u, const KernelParam& p, int8_t* out) {
        if (m == 1) {
            runLhsQuantPack(u, p, 1, k, blkSize, getMr(p, m), lhs, out);
            return;
        }
        size_t mr = getMr(p, m);
        int vecPer = getVecNumPerThread(m, threadNum, mr);
        int need = m % vecPer == 0 ? m / vecPer : (m / vecPer + 1);
        size_t srcStride = (size_t)vecPer * k * elementSize;
        MNN_CONCURRENCY_BEGIN(tId, need) {
            int t = (int)tId;
            auto threadSrc = lhs + (size_t)t * srcStride;
            auto threadDst = out + getLhsQuantedPackedOffset(u, p, m, (size_t)t * vecPer, k, blkSize);
            int vecNum = (t == need - 1) ? (m - vecPer * t) : vecPer; //Last threadN may less than vecPer.
            runLhsQuantPack(u, p, vecNum, k, blkSize, mr, threadSrc, threadDst);
        }
        MNN_CONCURRENCY_END();
    };

    //Run matmul.
    auto dst = output->host<uint8_t>();
    if(outputDes->dimensionFormat != MNN_DATA_FORMAT_NHWC) {
        //store matmul result to convert buffer.
        dst = mOutputConvertBuffer->host<uint8_t>();
    }
    auto postPtr = getPostParameters();

    // Decide whether to split the N dimension between the SME slot (thread 0) and the NEON slot
    // (threads 1..threadNum-1) so both run concurrently. Only possible when a NEON slot was packed
    // (mHybrid) and more than one thread is available.
    bool doHybrid = mHybrid && threadNum > 1;
    size_t nSme = 0, nNeon = 0;
    int neonThreads = 0;
    if (doHybrid) {
        neonThreads = threadNum - 1;
        // Balance the N-split with the calibrated two-regime cost model: the SME slot runs
        // columns [0, nSme) on one thread concurrently with the NEON slot running the remaining
        // columns spread over neonThreads. Pick the SME column count (aligned to a whole number of
        // SME N-steps) that minimises the concurrent finish time max(t_sme, t_neon_per_thread).
        size_t nStepSme = getNStep(mParam);
        size_t bestNSme = nStepSme;
        double bestFinish = 1e300;
        for (size_t cand = nStepSme; cand < (size_t)n; cand += nStepSme) {
            size_t nn = (size_t)n - cand;
            int perT = (int)((nn + (size_t)neonThreads - 1) / (size_t)neonThreads);
            double tSme  = kaiEstimateSmeUs(mKernelType, m, cand, k, blkSize);
            double tNeon = kaiEstimateNeonUs(mKernelType, m, (size_t)perT, k, blkSize);
            double finish = tSme > tNeon ? tSme : tNeon;
            if (finish < bestFinish) {
                bestFinish = finish;
                bestNSme = cand;
            }
        }
        nSme = bestNSme;
        if (nSme > n) {
            nSme = n;
        }
        nNeon = n - nSme;
        if (nSme == 0 || nNeon == 0) {
            doHybrid = false;
        }
    }

    if (!doHybrid) {
        // Single-slot path: SME-only on one thread (SME prefers a single thread for better
        // performance/power ratio) or NEON-only spread across all threads.
        // TEMP bench hooks: MNN_KAI_MODE=neon forces the NEON slot (for isolated per-engine latency
        // on a single thread); MNN_KAI_PROF=1 prints matmul-only latency as CSV for dataset collection.
        const Ukernel* su = &mUkernel;
        const KernelParam* sp = &mParam;
        int8_t* sLhs = mTempIm2ColBuffer->host<int8_t>();
        uint8_t* sRhs = mWeightInt8->host<uint8_t>();
        const char* engTag = "SME";
        static const char* kaiMode = getenv("MNN_KAI_MODE");
        if (kaiMode != nullptr && kaiMode[0] == 'n' && mHybrid) {
            su = &mUkernelNeon; sp = &mParamNeon;
            sLhs = mTempIm2ColBufferNeon->host<int8_t>();
            sRhs = mWeightInt8Neon->host<uint8_t>();
            engTag = "NEON";
        }
        packLhs(*su, *sp, sLhs);
        auto lhsPacked = sLhs;
        auto rhsPacked = sRhs;
        int matThreadNum = (bSupportSme2() && su == &mUkernel) ? 1 : threadNum;
        int vecPerThread = getVecNumPerThread(n, matThreadNum, getNStep(*sp));
        int threadNeed = n % vecPerThread == 0 ? n / vecPerThread : (n / vecPerThread + 1);
        static const char* kaiProf = getenv("MNN_KAI_PROF");
        static const char* kaiRepEnv = getenv("MNN_KAI_REP");
        int kaiRep = (kaiProf != nullptr && kaiRepEnv != nullptr) ? atoi(kaiRepEnv) : 1;
        if (kaiRep < 1) { kaiRep = 1; }
        MNN::Timer _pt;
        for (int rep = 0; rep < kaiRep; ++rep) {
        MNN_CONCURRENCY_BEGIN(tId, threadNeed) {
            int t = (int)tId;
            auto threadRhsPacked = rhsPacked + getRhsPackedOffset(*su, *sp, t * vecPerThread, k, blkSize);
            auto threadDst = dst + getDstOffset(0, t * vecPerThread, n, elementSize);
            int vecNum = (t == threadNeed - 1) ? (n - vecPerThread * t) : vecPerThread; //Last threadN may less than vecPerThread.
            runMatmul(*su, *sp, m, vecNum, k, blkSize, lhsPacked, threadRhsPacked, threadDst, n * elementSize, elementSize, postPtr[3], postPtr[2]);
        }
        MNN_CONCURRENCY_END();
        }
        if (kaiProf != nullptr) {
            MNN_PRINT("KAISWEEP,%s,%d,%d,%d,%d,%.5f\n", engTag, (int)m, (int)n, (int)k, (int)blkSize,
                      (double)_pt.durationInUs() / 1000.0 / kaiRep);
        }
    } else {
        // Hybrid path: pack lhs once per slot (different mr => different packed layout), then run the
        // SME kernel on thread 0 over columns [0, nSme) concurrently with NEON kernels on the
        // remaining threads over columns [nSme, n).
        auto lhsPackedSme  = mTempIm2ColBuffer->host<int8_t>();
        auto lhsPackedNeon = mTempIm2ColBufferNeon->host<int8_t>();
        packLhs(mUkernel, mParam, lhsPackedSme);
        packLhs(mUkernelNeon, mParamNeon, lhsPackedNeon);
        auto rhsPackedSme  = mWeightInt8->host<uint8_t>();
        auto rhsPackedNeon = mWeightInt8Neon->host<uint8_t>();
        size_t nStepNeon = getNStep(mParamNeon);
        int vecPerNeon = getVecNumPerThread(nNeon, neonThreads, nStepNeon);
        // TEMP bench hook: MNN_KAI_PROF=1 times the concurrent hybrid matmul only (packLhs excluded,
        // done once above), MNN_KAI_REP=N repeats for a stable median. Prints the chosen N-split.
        static const char* kaiProfH = getenv("MNN_KAI_PROF");
        static const char* kaiRepEnvH = getenv("MNN_KAI_REP");
        int kaiRepH = (kaiProfH != nullptr && kaiRepEnvH != nullptr) ? atoi(kaiRepEnvH) : 1;
        if (kaiRepH < 1) { kaiRepH = 1; }
        MNN::Timer _pth;
        for (int rep = 0; rep < kaiRepH; ++rep) {
        MNN_CONCURRENCY_BEGIN(tId, threadNum) {
            int t = (int)tId;
            if (t == 0) {
                // SME slot: columns [0, nSme).
                runMatmul(mUkernel, mParam, m, nSme, k, blkSize, lhsPackedSme, rhsPackedSme,
                          dst, n * elementSize, elementSize, postPtr[3], postPtr[2]);
            } else {
                // NEON slot: columns [nSme, n) split among neonThreads.
                int neonId = t - 1;
                int localStart = neonId * vecPerNeon;
                if (localStart < (int)nNeon) {
                    int vecNum = (localStart + vecPerNeon > (int)nNeon) ? ((int)nNeon - localStart) : vecPerNeon;
                    size_t globalStart = nSme + (size_t)localStart;
                    auto threadRhsPacked = rhsPackedNeon + getRhsPackedOffset(mUkernelNeon, mParamNeon, globalStart, k, blkSize);
                    auto threadDst = dst + getDstOffset(0, globalStart, n, elementSize);
                    runMatmul(mUkernelNeon, mParamNeon, m, vecNum, k, blkSize, lhsPackedNeon, threadRhsPacked,
                              threadDst, n * elementSize, elementSize, postPtr[3], postPtr[2]);
                }
            }
        }
        MNN_CONCURRENCY_END();
        }
        if (kaiProfH != nullptr) {
            MNN_PRINT("KAIHYB,%d,%d,%d,%d,%d,%d,%.5f\n", (int)m, (int)n, (int)k, (int)blkSize,
                      (int)nSme, (int)nNeon, (double)_pth.durationInUs() / 1000.0 / kaiRepH);
        }
    }

    if(outputDes->dimensionFormat != MNN_DATA_FORMAT_NHWC) {
        // Convert output from NHWC format to original format.
        MNN_CONCURRENCY_BEGIN(tId, threadNum) {
            CPUTensorConverter::convert(mOutputConvertBuffer.get(), output, core, tId, threadNum);
        };
        MNN_CONCURRENCY_END();
    }

    return NO_ERROR;
}

} // namespace MNN
#endif //MNN_KLEIDIAI_ENABLED
