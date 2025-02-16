#include "../kernels_interface.h"
#include "argmax_kernel.cuh"
#include "tensor/tensor.h"

namespace kernel{

__forceinline__ __device__ void warp_reduce_argmax(float& val,size_t& ptr){
    float tmp_val;
    unsigned int mask = __ballot_sync(0xFFFFFFFF,true);
    for(unsigned int offset = warpSize >> 1; offset > 0; offset >>= 1){
        tmp_val = __shfl_down_sync(mask, val, k, warpSize);
        tmp_ptr = __shfl_down_sync(mask, ptr, k, warpSize);
        if (ptr == SIZE_MAX || tmp_ptr == SIZE_MAX) 
            continue;
        if (tmp_val > val) {
            val = tmp_val;
            ptr = tmp_ptr;
        } else if (tmp_val == val && tmp_ptr < ptr) {
            ptr = tmp_ptr;
        }
    }
}

__forceinline__ __device__ void block_reduce_argmax(float& val, size_t& ptr, float* shared_value,size_t* shared_ptr){
    int lane_id = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    warp_reduce_argmax(val, ptr);
    __syncthreads();

    if (lane_id == 0) {
        shared_value[warp_id] = val;
        shared_ptr[warp_id] = ptr;
    }
    __syncthreads();

    if (threadIdx.x < blockDim.x / warpSize) {
        val = shared_value[lane_id];
        ptr = shared_ptr[lane_id];
    } else {
        val = 0;
        ptr = SIZE_MAX;
    }

    if (warp_id == 0) {
        warp_reduce_argmax(val, ptr);
    }
}

size_t argmax_kernel_cu(const float* input_ptr, size_t size, void* stream){

}

}