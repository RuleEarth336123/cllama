#include "emb_kernel.cuh"
namespace kernel{

    __global__ void emb_kernel_cu_fp32(int32_t vocab_size, int32_t token_num, int32_t weight_dim,\
        const int32_t* input_ptr, const float* weight_ptr,float* output_ptr){

            int32_t token_idx = blockIdx.x;
            if(token_idx >= token_num){
                return;
            }

            int32_t token = input_ptr[token_idx];
            if(token >= vocal_size){
                return;
            }

            float* output_ptr_start = output_ptr + token_idx * weight_dim;
            const float* weight_ptr_start = weight_ptr + token * weight_dim;

            for(int32_t i = threadIdx.x; i < weight_dim; i+= blockDim.x){
                output_ptr_start[i] = weight_ptr_start[i];
            }

    }

}