# ------------------------------------------------------------------------------
# Function: download_kleidiai_and_collect_sources
#
# Description:
#   Downloads the KleidiAI source code and collects relevant source files.
#   If the download fails, the function will not terminate the configuration
#   process but will return empty lists for the source files.
#
# Exposed Variables (via PARENT_SCOPE):
#   - MNN_SOURCES_KLEIDIAI : List of general KleidiAI source files.
#   - KLEIDIAI_FILES_SME2  : List of KleidiAI source files specific to SME2 architecture.
#
# Usage:
#   include(KleidiAI.cmake)
#   download_kleidiai_and_collect_sources()
#   Use MNN_SOURCES_KLEIDIAI and KLEIDIAI_FILES_SME2 in subsequent build steps.
# ------------------------------------------------------------------------------

# Capture the MNN cmake/ directory at include time. Inside a CMake function,
# CMAKE_CURRENT_LIST_DIR resolves to the *caller's* directory, so we must
# snapshot it here (file scope) to always get the MNN repo root regardless of
# whether MNN is the top-level project or an embedded dependency.
set(_MNN_KLEIDIAI_CMAKE_LIST_DIR "${CMAKE_CURRENT_LIST_DIR}")

function (download_kleidiai_and_collect_sources)
    set(MNN_SOURCES_KLEIDIAI "" PARENT_SCOPE)
    set(KLEIDIAI_FILES_SME2 "" PARENT_SCOPE)

    # Disable the KleidiAI tests
    set(KLEIDIAI_BUILD_TESTS OFF)

    set(KLEIDIAI_COMMIT_SHA "1.32.0")
    set(KLEIDIAI_DOWNLOAD_URL "https://github.com/ARM-software/kleidiai/releases/download/v${KLEIDIAI_COMMIT_SHA}/kleidiai-v${KLEIDIAI_COMMIT_SHA}-src.tar.gz")
    set(KLEIDIAI_ARCHIVE_SHA256 "a48d373e7c5d721a0aecbf5ddaf0bdf245bc8064e6b84086f13db0e102b553a6")

    set(_kleidiai_src_dir "")
    if(DEFINED KLEIDIAI_SRC_DIR AND EXISTS "${KLEIDIAI_SRC_DIR}")
        set(_kleidiai_src_dir "${KLEIDIAI_SRC_DIR}")
    else()
        # Anchor downloads to MNN's own binary dir. CMAKE_BINARY_DIR points at the
        # top-level project's build tree, which is wrong (and pollutes the parent)
        # when MNN is consumed as a subproject. CMAKE_CURRENT_BINARY_DIR follows the
        # include() chain and always resolves to MNN's build directory.
        set(_deps_dir "${CMAKE_CURRENT_BINARY_DIR}/_deps")
        file(MAKE_DIRECTORY "${_deps_dir}")

        set(_tar_path "${_deps_dir}/kleidiai-v${KLEIDIAI_COMMIT_SHA}-src.tar.gz")

        file(
            DOWNLOAD "${KLEIDIAI_DOWNLOAD_URL}" "${_tar_path}"
            STATUS _dl_status
            SHOW_PROGRESS
            TIMEOUT 180)
        list(GET _dl_status 0 _dl_code)
        if(NOT _dl_code EQUAL 0)
            message(WARNING "KleidiAI download failed: ${_dl_status}. Building without KleidiAI. "
                            "If you have the source locally, set KLEIDIAI_SRC_DIR to skip downloading.")
            return()
        endif()

        file(SHA256 "${_tar_path}" _actual_sha256)
        if(NOT _actual_sha256 STREQUAL "${KLEIDIAI_ARCHIVE_SHA256}")
            message(WARNING "KleidiAI archive hash mismatch. Building without KleidiAI.")
            return()
        endif()

        execute_process(
            COMMAND ${CMAKE_COMMAND} -E tar xzf "${_tar_path}"
            WORKING_DIRECTORY "${_deps_dir}"
            RESULT_VARIABLE _untar_code)
        if(NOT _untar_code EQUAL 0)
            message(WARNING "KleidiAI extract failed (code=${_untar_code}). Building without KleidiAI.")
            return()
        endif()

        set(_kleidiai_src_dir "${_deps_dir}/kleidiai-v${KLEIDIAI_COMMIT_SHA}")
        if(NOT EXISTS "${_kleidiai_src_dir}/kai")
            message(WARNING "KleidiAI source tree not found at expected location. Building without KleidiAI.")
            return()
        endif()

        set(KLEIDIAI_SRC_DIR
            "${_kleidiai_src_dir}"
            CACHE PATH "Path to KleidiAI source (downloaded or provided)" FORCE)
    endif()

    message(STATUS "KleidiAI source: ${_kleidiai_src_dir} (default release v${KLEIDIAI_COMMIT_SHA})")

    set(MNN_KLEIDIAI_DIR "${_MNN_KLEIDIAI_CMAKE_LIST_DIR}/../source/backend/cpu/kleidiai")
    if(NOT EXISTS "${MNN_KLEIDIAI_DIR}/KleidiAIConvInt8.cpp")
        # These sources live inside the MNN repo itself (committed alongside this
        # file), so a correctly-resolved path always finds them. Reaching here means
        # the MNN checkout is incomplete/corrupted, not an external dependency issue
        # -- fail loudly instead of silently disabling a feature the user enabled.
        message(FATAL_ERROR "KleidiAI MNN source files not found at ${MNN_KLEIDIAI_DIR}. "
                            "Your MNN checkout looks incomplete; update it to a version that "
                            "includes source/backend/cpu/kleidiai, or pass -DMNN_KLEIDIAI=OFF.")
    endif()
    set(MNN_KLEIDIAI_INTEGRATION_SOURCES
        ${MNN_KLEIDIAI_DIR}/KleidiAICommon.cpp
        ${MNN_KLEIDIAI_DIR}/KleidiAIConvolution.cpp
        ${MNN_KLEIDIAI_DIR}/KleidiAIConvolutionDepthwise.cpp
        ${MNN_KLEIDIAI_DIR}/KleidiAIConvInt8.cpp
        ${MNN_KLEIDIAI_DIR}/KleidiAIDenseConvolution.cpp)
    list(APPEND MNN_SOURCES_KLEIDIAI ${MNN_KLEIDIAI_INTEGRATION_SOURCES})

    # All nine static signed-INT8 kernels are one optional extension. Source availability
    # and MNN_SME2 are compile gates; CoreFunctions::supportSME2 is the runtime ISA gate.
    set(_static_int8_kernels
        matmul/pack/kai_lhs_pack_x8p2vlx4_x8_sme
        matmul/pack/kai_lhs_imatmul_pack_x8p2vlx4_x8p_sme
        matmul/pack/kai_matmul_pack_lhs_mxk_x8p4vsx4_x8_sme
        matmul/pack/kai_rhs_pack_nxk_qsi4cxp8vsx4sf32bi32_qsi4cx_f32_i32_sme
        matmul/pack/kai_rhs_pack_nxk_qsi4cxp8vsx4sf32bi32_qsx4cx_f32_i32_sme
        matmul/pack/kai_matmul_pack_rhs_nxk_qsu2cxp16vsx4sf32bi32_qsu2cx_f32_i32_sme
        matmul/pack/kai_matmul_pack_rhs_kxn_qsi8cxp4vsx4bi32sf32_qsi8_i32_f32_sme
        matmul/pack/kai_rhs_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme
        matmul/pack/kai_rhs_imatmul_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme
        matmul/matmul_clamp_qai8_qai8_qsi4cxp/kai_matmul_clamp_qai8_qai8_qsi4cxp8vsx4sf32bi32_1x64vs_sme2_dot
        matmul/matmul_clamp_qai8_qai8p_qsi4cxp/kai_matmul_clamp_qai8_qai8p8vsx4_qsi4cxp8vsx4sf32bi32_8vsx8vs_sme2_mopa
        matmul/matmul_clamp_qai8_qai8_qsu2cxp/kai_matmul_clamp_qai8_qai8_qsu2cxp16vsx4sf32bi32_1x64vs_sme2_dot
        matmul/matmul_clamp_qai8_qai8p_qsu2cxp/kai_matmul_clamp_qai8_qai8p8vsx4_qsu2cxp16vsx4sf32bi32_8vsx16vs_sme2_mopa
        matmul/matmul_clamp_qai8_qai8_qsi8cxp/kai_matmul_clamp_qai8_qai8_qsi8cxp4vsx4bi32sf32_1x32vs_sme2_dot
        matmul/matmul_clamp_qai8_qai8p_qsi8cxp/kai_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4bi32sf32_8vsx8vs_sme2_mopa
        matmul/matmul_clamp_qai8_qai8_qsi8cxp/kai_matmul_clamp_qai8_qai8_qsi8cxp2vlx4sb_1x16vl_sme2_dot
        matmul/matmul_clamp_qai8_qai8p_qsi8cxp/kai_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa
        matmul/imatmul_clamp_qai8_qai8p_qsi8cxp/kai_imatmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa)
    # The new W8 C wrappers call six MOPA tiles (including edges).
    # Several packers also call ASM whose basename differs from their C wrapper.
    set(_static_int8_asm
        matmul/pack/kai_lhs_pack_x8p2vlx4_x8_sme_asm
        matmul/pack/kai_lhs_imatmul_pack_x8p2vlx4_x8p_sme_asm
        matmul/pack/kai_matmul_pack_rows_x8p4vsx4_x8_sme_asm
        matmul/pack/kai_matmul_pack_rhs_kxn_qsi8cxp4vsx4bi32sf32_qsi8_i32_f32_sme_asm
        matmul/pack/kai_rhs_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme_asm
        matmul/pack/kai_rhs_imatmul_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme_asm
        matmul/matmul_clamp_qai8_qai8_qsi4cxp/kai_matmul_clamp_qai8_qai8_qsi4cxp8vsx4sf32bi32_1x64vs_sme2_dot_asm
        matmul/matmul_clamp_qai8_qai8p_qsi4cxp/kai_matmul_clamp_qai8_qai8p8vsx4_qsi4cxp8vsx4sf32bi32_8vsx8vs_sme2_mopa_asm
        matmul/matmul_clamp_qai8_qai8_qsu2cxp/kai_matmul_clamp_qai8_qai8_qsu2cxp16vsx4sf32bi32_1x64vs_sme2_dot_asm
        matmul/matmul_clamp_qai8_qai8p_qsu2cxp/kai_matmul_clamp_qai8_qai8p8vsx4_qsu2cxp16vsx4sf32bi32_8vsx16vs_sme2_mopa_asm
        matmul/matmul_clamp_qai8_qai8_qsi8cxp/kai_matmul_clamp_qai8_qai8_qsi8cxp4vsx4bi32sf32_1x32vs_sme2_dot_asm
        matmul/matmul_clamp_qai8_qai8p_qsi8cxp/kai_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4bi32sf32_4vsx4vs_sme2_mopa_asm
        matmul/matmul_clamp_qai8_qai8p_qsi8cxp/kai_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4bi32sf32_4vsx8vs_sme2_mopa_asm
        matmul/matmul_clamp_qai8_qai8p_qsi8cxp/kai_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4bi32sf32_4vsx16vs_sme2_mopa_asm
        matmul/matmul_clamp_qai8_qai8p_qsi8cxp/kai_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4bi32sf32_8vsx4vs_sme2_mopa_asm
        matmul/matmul_clamp_qai8_qai8p_qsi8cxp/kai_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4bi32sf32_8vsx8vs_sme2_mopa_asm
        matmul/matmul_clamp_qai8_qai8p_qsi8cxp/kai_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4bi32sf32_16vsx4vs_sme2_mopa_asm
        matmul/matmul_clamp_qai8_qai8_qsi8cxp/kai_matmul_clamp_qai8_qai8_qsi8cxp2vlx4sb_1x16vl_sme2_dot_asm
        matmul/matmul_clamp_qai8_qai8p_qsi8cxp/kai_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa_asm
        matmul/imatmul_clamp_qai8_qai8p_qsi8cxp/kai_imatmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa_asm)
    set(_have_static_int8 TRUE)
    foreach(_kernel ${_static_int8_kernels})
        if(NOT EXISTS "${KLEIDIAI_SRC_DIR}/kai/ukernels/${_kernel}.c")
            set(_have_static_int8 FALSE)
        endif()
    endforeach()
    foreach(_kernel ${_static_int8_asm})
        if(NOT EXISTS "${KLEIDIAI_SRC_DIR}/kai/ukernels/${_kernel}.S")
            set(_have_static_int8 FALSE)
        endif()
    endforeach()
    set(_static_int8_headers
        kai/kai_common.h
        kai/kai_common_sme_asm.S
        kai/ukernels/matmul/kai_matmul.h
        kai/ukernels/matmul/kai_matmul_types.h
        kai/ukernels/matmul/kai_matmul_pack_lhs.h
        kai/ukernels/matmul/kai_matmul_pack_lhs_types.h
        kai/ukernels/matmul/kai_matmul_pack_rhs.h
        kai/ukernels/matmul/kai_matmul_pack_rhs_types.h)
    foreach(_kernel ${_static_int8_kernels})
        # New API wrappers use the shared headers; legacy wrappers use matching headers.
        if(_kernel MATCHES "2vlx4")
            list(APPEND _static_int8_headers "kai/ukernels/${_kernel}.h")
        endif()
    endforeach()
    foreach(_header ${_static_int8_headers})
        if(NOT EXISTS "${KLEIDIAI_SRC_DIR}/${_header}")
            set(_have_static_int8 FALSE)
        endif()
    endforeach()
    if(NOT MNN_SME2)
        set(_have_static_int8 FALSE)
    endif()
    if(_have_static_int8)
        add_definitions(-DMNN_KLEIDIAI_STATIC_INT8=1)
        set(_static_int8_adapters
            ${MNN_KLEIDIAI_DIR}/KleidiAIStaticConvInt8.cpp
            ${MNN_KLEIDIAI_DIR}/KleidiAIStaticInt8Kernels.cpp)
        list(APPEND MNN_KLEIDIAI_INTEGRATION_SOURCES ${_static_int8_adapters})
        list(APPEND MNN_SOURCES_KLEIDIAI ${_static_int8_adapters})
        foreach(_kernel ${_static_int8_kernels})
            list(APPEND KLEIDIAI_FILES_SME2 "${KLEIDIAI_SRC_DIR}/kai/ukernels/${_kernel}.c")
        endforeach()
        foreach(_kernel ${_static_int8_asm})
            list(APPEND KLEIDIAI_FILES_SME2 "${KLEIDIAI_SRC_DIR}/kai/ukernels/${_kernel}.S")
        endforeach()
    elseif(MNN_SME2)
        message(STATUS "KleidiAI static INT8 disabled: source override lacks the complete v1.32 extension")
    endif()

    include_directories(${MNN_KLEIDIAI_DIR})

    include_directories(
        ${KLEIDIAI_SRC_DIR}/
        ${KLEIDIAI_SRC_DIR}/kai/
        ${KLEIDIAI_SRC_DIR}/kai/ukernels/
        ${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/
        ${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/
        ${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/matmul_clamp_f16_qsi8d32p_qai4c32p/
        ${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/matmul_clamp_f32_f16p_qai4c32p/
        ${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qai4c32p/
        ${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/pack/
        ${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/matmul_clamp_f32_f32p_f32p/
        ${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/
        ${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/matmul_clamp_f16_f16p_f16p/
        ${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/matmul_clamp_f16_f16_f16p/
        ${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/imatmul_clamp_f32_f32p_f32p/
        ${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/imatmul_clamp_f16_f16p_f16p/
        ${KLEIDIAI_SRC_DIR}/kai/ukernels/dwconv/pack/
        ${KLEIDIAI_SRC_DIR}/kai/ukernels/dwconv/dwconv_f32_f32_f32p/)

    # Compile only kernels used by MNN. New upstream kernels may require newer
    # compiler ISA support even when runtime dispatch would never select them.
    set(_neon_kernels
        matmul/pack/kai_lhs_quant_pack_qsi8d32pscalef32_f16_neon
        matmul/pack/kai_lhs_quant_pack_qsi8d32pscalef32_f32_neon
        matmul/pack/kai_rhs_pack_nxk_qai4c32p_qau4c32s0s1_f32_f32_f32_neon
        matmul/matmul_clamp_f16_qsi8d32p_qai4c32p/kai_matmul_clamp_f16_qsi8d32p1x8_qai4c32p4x8_1x4_neon_dotprod
        matmul/matmul_clamp_f16_qsi8d32p_qai4c32p/kai_matmul_clamp_f16_qsi8d32p4x8_qai4c32p4x8_8x4_neon_i8mm
        matmul/matmul_clamp_f32_qsi8d32p_qai4c32p/kai_matmul_clamp_f32_qsi8d32p1x8_qai4c32p4x8_1x4_neon_dotprod
        matmul/matmul_clamp_f32_qsi8d32p_qai4c32p/kai_matmul_clamp_f32_qsi8d32p4x8_qai4c32p4x8_8x4_neon_i8mm)
    set(_sme_kernels
        matmul/pack/kai_rhs_pack_nxk_qai4c32ps1s0nrx4_qau4c32s0s1_f32_f32_f32_neon
        matmul/pack/kai_lhs_pack_f32p2vlx1_f32_sme
        matmul/pack/kai_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme
        matmul/pack/kai_lhs_pack_x16p2vlx2_x16_sme
        matmul/pack/kai_rhs_pack_nxk_x16p2vlx2b_x16_x16_sme
        matmul/pack/kai_lhs_imatmul_pack_x32p2vlx1_x32p_sme
        matmul/pack/kai_rhs_imatmul_pack_kxn_x32p2vlx1b_x32_x32_sme
        matmul/pack/kai_lhs_imatmul_pack_x16p2vlx2_x16p_sme
        matmul/pack/kai_rhs_imatmul_pack_kxn_x16p2vlx2b_x16_x16_sme
        matmul/matmul_clamp_f32_f32p_f32p/kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa
        matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla
        matmul/matmul_clamp_f16_f16p_f16p/kai_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa
        matmul/matmul_clamp_f16_f16_f16p/kai_matmul_clamp_f16_f16_f16p2vlx2b_1x16vl_sme2_dot
        matmul/imatmul_clamp_f32_f32p_f32p/kai_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme2_mopa
        matmul/imatmul_clamp_f16_f16p_f16p/kai_imatmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa
        matmul/matmul_clamp_f16_qsi8d32p_qai4c32p/kai_matmul_clamp_f16_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa
        matmul/matmul_clamp_f16_qsi8d32p_qai4c32p/kai_matmul_clamp_f16_qsi8d32p1x4_qai4c32p4vlx4_1x4vl_sme2_dot
        matmul/matmul_clamp_f32_qsi8d32p_qai4c32p/kai_matmul_clamp_f32_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa
        matmul/matmul_clamp_f32_qsi8d32p_qai4c32p/kai_matmul_clamp_f32_qsi8d32p1x4_qai4c32p4vlx4_1x4vl_sme2_dot
        dwconv/pack/kai_rhs_dwconv_pack_x32p1vlx1b_x32_x32_sme
        dwconv/dwconv_f32_f32_f32p/kai_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla)
    foreach(_isa NEON SME2)
        if(_isa STREQUAL "NEON")
            set(_kernels ${_neon_kernels})
        else()
            set(_kernels ${_sme_kernels})
        endif()
        foreach(_kernel ${_kernels})
            set(_base "${KLEIDIAI_SRC_DIR}/kai/ukernels/${_kernel}")
            if(NOT EXISTS "${_base}.c")
                message(FATAL_ERROR "Required KleidiAI kernel missing: ${_base}.c")
            endif()
            list(APPEND KLEIDIAI_FILES_${_isa} "${_base}.c")
            if(EXISTS "${_base}_asm.S")
                list(APPEND KLEIDIAI_FILES_${_isa} "${_base}_asm.S")
            endif()
        endforeach()
    endforeach()
    list(APPEND MNN_SOURCES_KLEIDIAI ${KLEIDIAI_FILES_NEON})
    list(APPEND KLEIDIAI_FILES_SME2 "${KLEIDIAI_SRC_DIR}/kai/kai_common_sme_asm.S")

    # This legacy direct FP32 -> packed FP16 int4 API is not in stock releases.
    # Enable it only for a supplied source tree containing the complete
    # extension; stock releases retain the supported dynamic-quant SME path.
    set(_direct_pack "${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/pack/kai_lhs_pack_f16pmrx4_f32_neon")
    set(_direct_matmul "${KLEIDIAI_SRC_DIR}/kai/ukernels/matmul/matmul_clamp_f32_f16p_qai4c32p/kai_matmul_clamp_f32_f16p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa")
    if(EXISTS "${_direct_pack}.h" AND EXISTS "${_direct_pack}.c"
       AND EXISTS "${_direct_matmul}.h" AND EXISTS "${_direct_matmul}.c"
       AND EXISTS "${_direct_matmul}_asm.S")
        list(APPEND MNN_SOURCES_KLEIDIAI "${_direct_pack}.c")
        list(APPEND KLEIDIAI_FILES_SME2 "${_direct_matmul}.c" "${_direct_matmul}_asm.S")
        set_property(SOURCE "${MNN_KLEIDIAI_DIR}/KleidiAIConvInt8.cpp" APPEND PROPERTY
            COMPILE_DEFINITIONS MNN_KLEIDIAI_F16_PACKED_INT4=1)
    endif()

    set_source_files_properties(
        ${MNN_KLEIDIAI_INTEGRATION_SOURCES}
        PROPERTIES COMPILE_OPTIONS
            "-fno-tree-vectorize;-march=armv8.2-a+fp16")
    set_source_files_properties(
        ${KLEIDIAI_FILES_NEON}
        PROPERTIES COMPILE_OPTIONS
            "-fno-tree-vectorize;-march=armv8.2-a+i8mm+dotprod+fp16")
    set_source_files_properties(
        ${KLEIDIAI_FILES_SME2}
        PROPERTIES COMPILE_OPTIONS
                   "-fno-tree-vectorize;-march=armv8.2-a+sve+sve2")

    if(CMAKE_C_COMPILER_ID STREQUAL "AppleClang" AND CMAKE_C_COMPILER_VERSION VERSION_LESS 17)
        # AppleClang 16 crashes selecting SLP-generated SVE unpack operations in the new W8 wrappers.
        # The arithmetic microkernels are handwritten ASM and are unaffected by this restriction.
        set_property(SOURCE ${KLEIDIAI_FILES_SME2} APPEND PROPERTY COMPILE_OPTIONS "-fno-slp-vectorize")
    endif()

    set(MNN_SOURCES_KLEIDIAI "${MNN_SOURCES_KLEIDIAI}" PARENT_SCOPE)
    set(KLEIDIAI_FILES_SME2 "${KLEIDIAI_FILES_SME2}" PARENT_SCOPE)

    # Define macro to indicate KleidiAI is enabled (only on aarch64)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)")
        add_definitions(-DMNN_KLEIDIAI_ENABLED=1)
    endif()
endfunction()
