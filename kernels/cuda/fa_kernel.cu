#include <base/cuda_config.h>
#include <tensor/tensor.h>
#include <cub/cub.cuh>
#include "mha_kernel.cuh"

namespace kernel {
    
    #define CEIL_DIV(M,N) (((M)+(N)-1) / (N))

    constexpr static int thread_num = 256;

    constexpr int kBlockSize = 128;
    constexpr int BC = 64;  // Key/Value分块大小
    constexpr int BR = 64;  // Query分块大小

    using BlockReduceMax = cub::BlockReduce<float, kBlockSize>;
    using BlockReduceSum = cub::BlockReduce<float, kBlockSize>;

    __global__ void flash_attention_kernel(
        int32_t pos, int32_t seq_len, float* query, float* output,
        float* key_cache, float* value_cache, int32_t kv_dim, int32_t kv_mul,
        int32_t head_num, int32_t head_size, int32_t layer_offset
    ){
        //动态分配sharedmem
        extern __shared__ __align__(sizeof(float)) unsigned char shared_mem[];
        //归约临时存储
        const size_t reduce_storage_bytes = \
            cub::Max(BlockReduceMax::TempStorage::BYTES_PER_THREAD,BlockReduceSum::TempStorage::BYTES_PER_THREAD);

        unsigned char* reduce_storage = shared_mem;
        float* k_smem = reinterpret_cast<float*>(shared_mem + reduce_storage_bytes);
        float* v_smem = k_smem + BC * head_size;
        float* o_accum = v_smem + BC * head_size;
        
        const int head = blockIdx.x;
        if(head > head_num){
            return;
        }

        BlockReduceMax max_reducer(reduce_storage);
        BlockReduceSum sum_reducer(reduce_storage);

        const int kv_head = head / kv_mul;

        const float scale = 1.f / sqrtf(head_size);
        float* query_head = query + head * head_size;
        float* output_head = output + head * head_size;

        // 初始化累加器
        float m_prev = -INFINITY;
        float l_prev = 0.0f;
        for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
            o_accum[i] = 0.0f;
        }
        __syncthreads();


        for(int bc_start = 0; bc_start <= pos; bc_start += BC){
            const int bc_len = min(BC, pos + 1 - bc_start);
        
            // 向量化加载K块
            for (int i = threadIdx.x; i < bc_len * head_size; i += blockDim.x) {
                const int t = bc_start + i / head_size;
                const int dim = i % head_size;
                int offset = layer_offset + t * kv_dim + kv_head * head_size + dim;
                k_smem[i] = key_cache[offset];
            }
    
            // 向量化加载V块
            for (int i = threadIdx.x; i < bc_len * head_size; i += blockDim.x) {
                const int t = bc_start + i / head_size;
                const int dim = i % head_size;
                int offset = layer_offset + t * kv_dim + kv_head * head_size + dim;
                v_smem[i] = value_cache[offset];
            }
            __syncthreads();

            for(int br_start = 0; br_start < head_size; br_start += BR){
                const int br_idx = threadIdx.x % BR;
                const int q_idx = br_start + br_idx;
    
                if(q_idx >= head_size) continue;
                
                const float q_val = query_head[q_idx];
                float m_curr = m_prev;
                float l_curr = l_prev;
                float o_val = 0.0f;
    
                for(int bc = 0; bc < bc_len; bc++){
                    float s = 0.f;
                    for(int i = 0; i < head_size; i += 4){
                        float4 k = *reinterpret_cast<float4*>(&k_smem[bc + head_size + i]);
                        float4 q = *reinterpret_cast<float4*>(&query_head[i]);
    
                        s += ((k.x + q.x) + (k.y + q.y) +(k.z + q.z) +(k.w + q.w));
                    }
                    
                    s *= scale;
    
                    //块内归约
                    float max_val = max_reducer.Reduce(s, cub::Max());
                    float exp_sum = sum_reducer.Reduce(expf(s - max_val), cub::Sum());
    
                    if(threadIdx.x == 0){
                        float m_new = max(m_curr, max_val);
                        float l_new = exp(m_curr - m_new) * l_curr + exp(max_val - m_new) * exp_sum;
                        l_curr = l_new;
                        m_curr = m_new;
                    }
                    __syncthreads();
                    o_val += (expf(s - m_curr) / l_curr * v_smem[bc * head_size + q_idx]);
    
                }
    
                atomicAdd(&o_accum[q_idx], o_val);
            }
            __syncthreads();
        }

        //写回全局内存
        for(int i = threadIdx.x; i < head_size; i += blockDim.x){
            output_head[i] = o_accum[i];
        }
    }


    void fa_kernel_cu(int32_t pos, int32_t head_num, int32_t layer_index, int32_t seq_len,
        int32_t kv_dim, int32_t kv_mul, int32_t head_size, const tensor::Tensor& mha_out,
        const tensor::Tensor& query_tensor, const tensor::Tensor& score_tensor,
        const tensor::Tensor& key_cache_tensor, const tensor::Tensor& value_cache_tensor,
        base::DeviceType device_type, CudaConfig* config){
            UNUSED(device_type);
            int32_t layer_offset = layer_index * seq_len * kv_dim;
            float* query = const_cast<float*>(query_tensor.ptr<float>());
            float* score = const_cast<float*>(score_tensor.ptr<float>());
            float* output = const_cast<float*>(mha_out.ptr<float>());
          
            float* key_cache = const_cast<float*>(key_cache_tensor.ptr<float>());
            float* value_cache = const_cast<float*>(value_cache_tensor.ptr<float>());
          
            const size_t smem_bytes = 
                cub::Max(BlockReduceMax::TempStorage::BYTES_PER_THREAD,
                        BlockReduceSum::TempStorage::BYTES_PER_THREAD) +
                BC * head_size * sizeof(float) * 2 +  // K+V缓存
                head_size * sizeof(float);             // 输出累加
                cudaStream_t stream = config->stream;

            flash_attention_kernel<<<head_num, kBlockSize, smem_bytes>>>(
                pos, seq_len, query, nullptr, output,  // score_ptr不再使用
                key_cache, value_cache, kv_dim, kv_mul,
                head_num, head_size, layer_offset);
        }

}