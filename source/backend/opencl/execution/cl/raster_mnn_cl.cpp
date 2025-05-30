#include "opencl_source_map.hpp" 
namespace MNN { 
const char* raster = 
"#ifdef MNN_SUPPORT_FP16\n"
"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
"#endif\n"
"#define GLOBAL_SIZE_2_DIMS __private const int global_size_dim0,__private const int global_size_dim1,\n"
"#define DEAL_NON_UNIFORM_DIM2(input1, input2) "" if (input1 >= global_size_dim0 || input2 >= global_size_dim1) { "" return; "" }\n"
"__constant sampler_t SAMPLER=CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n"
"#define GLOBAL_SIZE_3_DIMS "" __private const int global_size_dim0,__private const int global_size_dim1,__private const int global_size_dim2,\n"
"#define DEAL_NON_UNIFORM_DIM3(input1, input2, input3) "" if (input1 >= global_size_dim0 || input2 >= global_size_dim1 || input3 >= global_size_dim2) { "" return; "" }\n"
"__kernel void buffer_set_zero(\n"
" GLOBAL_SIZE_2_DIMS\n"
" __global OUTPUT_TYPE *output\n"
" ) {\n"
" const int x=get_global_id(0);\n"
" const int y=get_global_id(1);\n"
" \n"
" DEAL_NON_UNIFORM_DIM2(x,y);\n"
" \n"
" output[y*global_size_dim0+x]=(OUTPUT_TYPE)(0);\n"
"}\n"
"__kernel void image_set_zero(\n"
" GLOBAL_SIZE_2_DIMS\n"
" __write_only image2d_t output\n"
" ) {\n"
" const int x=get_global_id(0);\n"
" const int y=get_global_id(1);\n"
" \n"
" DEAL_NON_UNIFORM_DIM2(x,y);\n"
" WI_DATA(output,(int2)(x,y),(OUTPUT_TYPE_I4)(0));\n"
"}\n"
"__kernel void raster_buffer(\n"
" GLOBAL_SIZE_3_DIMS\n"
" __global INPUT_TYPE *input,\n"
" __private const int inputOffset,\n"
" __private const int inputStride0,\n"
" __private const int inputStride1,\n"
" __private const int inputStride2,\n"
" __global OUTPUT_TYPE *output,\n"
" __private const int outputOffset,\n"
" __private const int outputStride0,\n"
" __private const int outputStride1,\n"
" __private const int outputStride2\n"
" ) {\n"
" const int x=get_global_id(0);\n"
" const int y=get_global_id(1);\n"
" const int z=get_global_id(2);\n"
" \n"
" DEAL_NON_UNIFORM_DIM3(x,y,z);\n"
" \n"
" int inputIndex=inputOffset+z*inputStride0+y*inputStride1+x*inputStride2;\n"
" int outputIndex=outputOffset+z*outputStride0+y*outputStride1+x*outputStride2;\n"
" output[outputIndex]=(OUTPUT_TYPE)input[inputIndex];\n"
"}\n"
"__kernel void raster_buffer_combine(\n"
" GLOBAL_SIZE_3_DIMS\n"
" __global INPUT_TYPE *input,\n"
" __private const int inputOffset,\n"
" __private const int combineSrcOffset,\n"
" __private const int inputStride0,\n"
" __private const int inputStride1,\n"
" __private const int inputStride2,\n"
" __global OUTPUT_TYPE *output,\n"
" __private const int outputOffset,\n"
" __private const int combineDstOffset,\n"
" __private const int outputStride0,\n"
" __private const int outputStride1,\n"
" __private const int outputStride2,\n"
" __private const int global_size0\n"
" ) {\n"
" const int idx=get_global_id(0);\n"
" const int y=get_global_id(1);\n"
" const int z=get_global_id(2);\n"
" \n"
" DEAL_NON_UNIFORM_DIM3(idx,y,z);\n"
" const int x=idx % global_size0;\n"
" const int id=idx/global_size0;\n"
" \n"
" int inputIndex=inputOffset+id*combineSrcOffset+z*inputStride0+y*inputStride1+x*inputStride2;\n"
" int outputIndex=outputOffset+id*combineDstOffset+z*outputStride0+y*outputStride1+x*outputStride2;\n"
" output[outputIndex]=(OUTPUT_TYPE)input[inputIndex];\n"
"}\n"
"__kernel void raster_image(\n"
" GLOBAL_SIZE_3_DIMS\n"
" __read_only image2d_t input,\n"
" __private const int inputOffset,\n"
" __private const int inputStride0,\n"
" __private const int inputStride1,\n"
" __private const int inputStride2,\n"
" __private const int inputHeight,\n"
" __private const int inputWidth,\n"
" __private const int inputChannel,\n"
" __write_only image2d_t output,\n"
" __private const int outputOffset,\n"
" __private const int outputStride0,\n"
" __private const int outputStride1,\n"
" __private const int outputStride2,\n"
" __private const int outputHeight,\n"
" __private const int outputWidth,\n"
" __private const int outputChannel\n"
" ) {\n"
" const int x=get_global_id(0);\n"
" const int y=get_global_id(1);\n"
" const int z=get_global_id(2);\n"
" \n"
" DEAL_NON_UNIFORM_DIM3(x,y,z);\n"
" \n"
" int inputIndex=inputOffset+(z*inputStride0+y*inputStride1+x*inputStride2)*4;\n"
" int outputIndex=outputOffset+(z*outputStride0+y*outputStride1+x*outputStride2)*4;\n"
" int inp_idx_n=inputIndex/((inputChannel+3)/4*inputHeight*inputWidth*4);\n"
" int inputIndex_left=inputIndex % ((inputChannel+3)/4*inputHeight*inputWidth*4);\n"
" int inp_idx_c4=inputIndex_left/(inputHeight*inputWidth*4);\n"
" inputIndex_left=inputIndex_left % (inputHeight*inputWidth*4);\n"
" int inp_idx_h=inputIndex_left/(inputWidth*4);\n"
" inputIndex_left=inputIndex_left % (inputWidth*4);\n"
" int inp_idx_w=inputIndex_left/4;\n"
" \n"
" int out_idx_n=outputIndex/((outputChannel+3)/4*outputHeight*outputWidth*4);\n"
" int outputIndex_left=outputIndex % ((outputChannel+3)/4*outputHeight*outputWidth*4);\n"
" int out_idx_c4=outputIndex_left/(outputHeight*outputWidth*4);\n"
" outputIndex_left=outputIndex_left % (outputHeight*outputWidth*4);\n"
" int out_idx_h=outputIndex_left/(outputWidth*4);\n"
" outputIndex_left=outputIndex_left % (outputWidth*4);\n"
" int out_idx_w=outputIndex_left/4;\n"
" \n"
" int inp_idx0=inp_idx_c4*inputWidth+inp_idx_w;\n"
" int inp_idx1=inp_idx_n*inputHeight+inp_idx_h;\n"
" int out_idx0=out_idx_c4*outputWidth+out_idx_w;\n"
" int out_idx1=out_idx_n*outputHeight+out_idx_h;\n"
" INPUT_TYPE_I4 out=RI_DATA(input,SAMPLER,(int2)(inp_idx0,inp_idx1));\n"
" WI_DATA(output,(int2)(out_idx0,out_idx1),CONVERT_OUTPUT_I4(out));\n"
"}\n"
;
}
