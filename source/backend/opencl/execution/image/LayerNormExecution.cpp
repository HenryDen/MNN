//
//  LayerNormExecution.cpp
//  MNN
//
//  Created by MNN on 2023/07/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opencl/execution/image/LayerNormExecution.hpp"
#include "core/TensorUtils.hpp"

namespace MNN {
namespace OpenCL {

LayerNormExecution::LayerNormExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend)
    : CommonExecution(backend, op) {
    mUnits.resize(1);
    auto &unit = mUnits[0];
    mOpenCLBackend = static_cast<OpenCLBackend *>(backend);
    auto runtime   = mOpenCLBackend->getOpenCLRuntime();
    mResource.reset(new LayernormResource);
    const auto* layer_norm_param = op->main_as_LayerNorm();
    if (nullptr != layer_norm_param->axis()) {
        mResource->axis_size = layer_norm_param->axis()->size();
    }
    mResource->epsilon_ = layer_norm_param->epsilon();
    mResource->group_ = layer_norm_param->group();
    mResource->RMSNorm = layer_norm_param->useRMSNorm();
    auto bufferUnitSize = mOpenCLBackend->getPrecision() != BackendConfig::Precision_High ? sizeof(half_float::half) : sizeof(float);
    unit.kernel = runtime->buildKernel("layernorm", "layernorm_w", {"-DLOCAL_SIZE=512"}, mOpenCLBackend->getPrecision());
    mResource->mMaxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(unit.kernel));

    mResource->has_gamma_beta_ = (layer_norm_param->gamma() && layer_norm_param->beta());
    int gammasize = 0;
    if (mResource->has_gamma_beta_) {
        MNN_ASSERT(layer_norm_param->gamma()->size() == layer_norm_param->beta()->size());
        gammasize = layer_norm_param->gamma()->size();
    }
    mResource->has_gamma_beta_ = mResource->has_gamma_beta_ || (layer_norm_param->external() && layer_norm_param->external()->size() > 1 && layer_norm_param->external()->data()[1] > 0);
    if (mResource->has_gamma_beta_ && gammasize == 0) {
        gammasize = layer_norm_param->external()->data()[1] / sizeof(float);
    }
        
    if(mResource->has_gamma_beta_){
        {
            auto error = CL_SUCCESS;
            int size = gammasize;
            mResource->mGammaBuffer.reset(new cl::Buffer(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, ALIGN_UP4(size) * bufferUnitSize));
            auto GammaPtrCL = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(*(mResource->mGammaBuffer.get()), true, CL_MAP_WRITE, 0, ALIGN_UP4(size) * bufferUnitSize, nullptr, nullptr, &error);
            const float* gamma_data = layer_norm_param->gamma()->data();
            if(GammaPtrCL != nullptr && error == CL_SUCCESS){
                if(mOpenCLBackend->getPrecision() != BackendConfig::Precision_High){
                    for (int i = 0; i < size; i++)
                    {
                        ((half_float::half*)GammaPtrCL)[i] = (half_float::half)(gamma_data[i]);
                    }
                    for(int i=size; i<ALIGN_UP4(size); i++) {
                        ((half_float::half*)GammaPtrCL)[i] = (half_float::half)(0.0f);
                    }
                }else{
                    ::memset(GammaPtrCL, 0, ALIGN_UP4(size) * sizeof(float));
                    ::memcpy(GammaPtrCL, gamma_data, size * sizeof(float));
                }
            }else{
                MNN_ERROR("Map error GammaPtrCL == nullptr \n");
            }
            mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(*mResource->mGammaBuffer.get(), GammaPtrCL);
        }
        {
            auto error = CL_SUCCESS;
            int size = gammasize;
            mResource->mBetaBuffer.reset(new cl::Buffer(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, ALIGN_UP4(size) * bufferUnitSize));
            auto BetaPtrCL = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(*(mResource->mBetaBuffer.get()), true, CL_MAP_WRITE, 0, ALIGN_UP4(size) * bufferUnitSize, nullptr, nullptr, &error);
            const float* beta_data = layer_norm_param->beta()->data();
            if(BetaPtrCL != nullptr && error == CL_SUCCESS){
                if(mOpenCLBackend->getPrecision() != BackendConfig::Precision_High){
                    for (int i = 0; i < size; i++)
                    {
                        ((half_float::half*)BetaPtrCL)[i] = (half_float::half)(beta_data[i]);
                    }
                    for(int i=size; i<ALIGN_UP4(size); i++) {
                        ((half_float::half*)BetaPtrCL)[i] = (half_float::half)(0.0f);
                    }
                }else{
                    ::memset(BetaPtrCL, 0, ALIGN_UP4(size) * sizeof(float));
                    ::memcpy(BetaPtrCL, beta_data, size * sizeof(float));
                }
            }else{
                MNN_ERROR("Map error BetaPtrCL == nullptr \n");
            }
            mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(*mResource->mBetaBuffer.get(), BetaPtrCL);
        }
    }
}

LayerNormExecution::LayerNormExecution(std::shared_ptr<LayernormResource> resource, const Op* op, Backend* backend) : CommonExecution(backend, op) {
    mResource = resource;
    mOpenCLBackend = (OpenCLBackend *)backend;
}

bool LayerNormExecution::onClone(Backend *bn, const Op *op, Execution **dst) {
    if (!mValid) {
        return false;
    }
    if (nullptr == dst) {
        return true;
    }
    *dst = new LayerNormExecution(mResource, op, bn);
    return true;
}

int LayerNormExecution::getLocalSize(int size, int maxGroupSize){
    int local_size = 1;
    while(local_size * 2 <= maxGroupSize && local_size * 2 <= size){
        local_size *= 2;
    }
    return local_size;
}

ErrorCode LayerNormExecution::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mUnits.resize(1);
    auto &unit = mUnits[0];
    Tensor *input  = inputs[0];
    Tensor *output = outputs[0];
    auto runtime = ((OpenCLBackend *)backend())->getOpenCLRuntime();
    auto MaxLocalSize = std::min(runtime->getMaxWorkItemSizes()[0], mResource->mMaxWorkGroupSize);

    std::vector<int> inputShape  = tensorShapeFormat(input);
    std::vector<int> outputShape = tensorShapeFormat(output);

    const int inputBatch    = inputShape[0];
    const int inputHeight   = inputShape[1];
    const int inputWidth    = inputShape[2];
    const int inputChannels = inputShape[3];
    int local_size;
    int rank = inputs.at(0)->dimensions();
    int outter_size = 1;
    int inner_size = 1;
    for (int i = 0; i < rank - mResource->axis_size; ++i) {
        outter_size *= inputs.at(0)->length(i);
    }
    for (int i = rank - mResource->axis_size; i < rank; ++i) {
        inner_size *= inputs.at(0)->length(i);
    }
    
    std::vector<uint32_t> mLWS{0, 0, 0, 0};
    std::vector<uint32_t> mGWS{0, 0, 0, 0};
    std::set<std::string> buildOptions;
    if(mResource->RMSNorm){
        buildOptions.emplace("-DRMSNORM");
    }
    if(mResource->has_gamma_beta_){
        buildOptions.emplace("-DGAMMA_BETA");
    }
    std::string kernelName;
    if (inner_size == inputWidth && outter_size == inputBatch * inputHeight * inputChannels) {
        kernelName = "layernorm_w";
        local_size = getLocalSize(inputWidth, MaxLocalSize);
        buildOptions.emplace("-DLOCAL_SIZE=" + std::to_string(local_size));
        unit.kernel = runtime->buildKernel("layernorm", kernelName, buildOptions, mOpenCLBackend->getPrecision());
        
        mGWS = {static_cast<uint32_t>(local_size),
                static_cast<uint32_t>(inputHeight * UP_DIV(inputChannels, 4)),
                static_cast<uint32_t>(inputBatch)};
    }else if(inner_size == inputWidth * inputHeight && outter_size == inputBatch * inputChannels){
        kernelName = "layernorm_hw";
        local_size = getLocalSize(inputWidth * inputHeight, MaxLocalSize);
        buildOptions.emplace("-DLOCAL_SIZE=" + std::to_string(local_size));
        unit.kernel = runtime->buildKernel("layernorm", kernelName, buildOptions, mOpenCLBackend->getPrecision());
        
        mGWS = {static_cast<uint32_t>(local_size),
                static_cast<uint32_t>(UP_DIV(inputChannels, 4)),
                static_cast<uint32_t>(inputBatch)};
    }else if(inner_size == inputWidth * inputHeight * inputChannels && outter_size == inputBatch){
        kernelName = "layernorm_chw";
        local_size = getLocalSize(inputWidth * inputHeight, MaxLocalSize);
        buildOptions.emplace("-DLOCAL_SIZE=" + std::to_string(local_size));
        unit.kernel = runtime->buildKernel("layernorm", kernelName, buildOptions, mOpenCLBackend->getPrecision());
        
        mGWS = {static_cast<uint32_t>(local_size),
                static_cast<uint32_t>(1),
                static_cast<uint32_t>(inputBatch)};
    }
    mLWS = {static_cast<uint32_t>(local_size), 1, 1};

    uint32_t idx = 0;
    cl_int ret = CL_SUCCESS;
    ret |= unit.kernel->get().setArg(idx++, mGWS[0]);
    ret |= unit.kernel->get().setArg(idx++, mGWS[1]);
    ret |= unit.kernel->get().setArg(idx++, mGWS[2]);
    ret |= unit.kernel->get().setArg(idx++, openCLImage(input));
    ret |= unit.kernel->get().setArg(idx++, openCLImage(output));
    ret |= unit.kernel->get().setArg(idx++, static_cast<int32_t>(inputWidth));
    ret |= unit.kernel->get().setArg(idx++, static_cast<int32_t>(inputHeight));
    ret |= unit.kernel->get().setArg(idx++, static_cast<int32_t>(inputChannels));
    if(mResource->has_gamma_beta_){
        ret |= unit.kernel->get().setArg(idx++, *mResource->mGammaBuffer.get());
        ret |= unit.kernel->get().setArg(idx++, *mResource->mBetaBuffer.get());
    }
    ret |= unit.kernel->get().setArg(idx++, mResource->epsilon_);
    MNN_CHECK_CL_SUCCESS(ret, "setArg LayerNormExecution");

    mOpenCLBackend->recordKernel3d(unit.kernel, mGWS, mLWS);
    unit.globalWorkSize = {mGWS[0], mGWS[1], mGWS[2]};
    unit.localWorkSize = {mLWS[0], mLWS[1], mLWS[2]};
    return NO_ERROR;

}

class LayerNormCreator : public OpenCLBackend::Creator {
public:
    virtual ~LayerNormCreator() = default;
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        const auto* layer_norm_param = op->main_as_LayerNorm();
        int group = layer_norm_param->group();
        if(group > 1){
			return nullptr;
        }
        return new LayerNormExecution(inputs, op, backend);
    }
};

REGISTER_OPENCL_OP_CREATOR(LayerNormCreator, OpType_LayerNorm, IMAGE);

} // namespace OpenCL
} // namespace MNN
