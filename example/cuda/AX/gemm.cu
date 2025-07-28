#include "gemm.cuh"
#include <cstdio>
#include <cuda_runtime.h>



namespace cuda {
    template<uint COL_PER_BLOCK, uint STRIDE, bool RELU = false>
    __global__ void gemm_4_AX(const CSRGraph_t d_csrA, // v_num * v_num 
                              const f32* X,            // v_num * dim
                              f32* Y,                  // v_num * dim
                              const uint v_num,
                              const uint dim)          // dim must be a multiple of 4 !!
    {
        static_assert(COL_PER_BLOCK % 4 == 0, "COL_PER_BLOCK must be a multiple of 4");
        
        const uint Y_row_index = blockIdx.x;
        const uint Y_col_index = blockIdx.y * COL_PER_BLOCK + threadIdx.x * 4;
        if (Y_col_index >= dim) 
            return;

        const uint start_index = d_csrA.index_pointers[Y_row_index];
        const uint end_index = d_csrA.index_pointers[Y_row_index + 1];

        float4 local_sum = {0.0f, 0.0f, 0.0f, 0.0f};
        __shared__ float4 buffer[STRIDE][COL_PER_BLOCK / 4 + 1];

        for (uint i = start_index; 
             i < end_index; 
             i += STRIDE) {
            
            const uint local_index = i + threadIdx.y;
            if (local_index >= end_index) 
                continue;
            
            const uint A_col_index = d_csrA.col_indices[local_index];
            const uint X_row_index = A_col_index;
            const uint X_col_index = Y_col_index;

            const f32 A_ele = d_csrA.data[local_index];
            if (X_col_index + 3 < dim) {
                float4 X_ele
                     = (float4&) X[X_row_index * dim + X_col_index];
                local_sum.x += A_ele * X_ele.x;
                local_sum.y += A_ele * X_ele.y;
                local_sum.z += A_ele * X_ele.z;
                local_sum.w += A_ele * X_ele.w;
            }

            else {
                if (X_col_index < dim) 
                    local_sum.x += A_ele * X[X_row_index * dim + X_col_index];
                if (X_col_index + 1 < dim) 
                    local_sum.y += A_ele * X[X_row_index * dim + X_col_index + 1];
                if (X_col_index + 2 < dim) 
                    local_sum.z += A_ele * X[X_row_index * dim + X_col_index + 2];
            }
        }
        
        __syncthreads();
        (float4& )buffer[threadIdx.y][threadIdx.x] = local_sum;
        __syncthreads();

        if (threadIdx.y == 0) {
            float4 sum = {0.0f, 0.0f, 0.0f, 0.0f};
            #pragma unroll
            for (uint j = 0; j < STRIDE; ++j) {
                float4 buffer_val = (float4&) buffer[j][threadIdx.x];
                sum.x += buffer_val.x;
                sum.y += buffer_val.y;
                sum.z += buffer_val.z;
                sum.w += buffer_val.w;
            }

            if constexpr (RELU) {
                sum.x = sum.x > 0 ? sum.x : 0;
                sum.y = sum.y > 0 ? sum.y : 0;
                sum.z = sum.z > 0 ? sum.z : 0;
                sum.w = sum.w > 0 ? sum.w : 0;
            }

            if (Y_col_index + 3 < dim) {
                (float4&) Y[Y_row_index * dim + Y_col_index] = (float4&) sum;
            } else {
                f32 tmp[4] = {sum.x, sum.y, sum.z, sum.w};
                for (uint i = 0; Y_col_index + i < dim && i < 4; ++i) {
                    Y[Y_row_index * dim + Y_col_index + i] = tmp[i];
                }
            }
        }
    }
}


namespace cuda 
{

void launch_kernel_AX( CSRGraph_t d_csrA,
                       const f32* d_X,
                       f32* d_Y,
                       const uint v_num,
                       const uint dim)
{
    constexpr uint COL_PER_BLOCK = 16 * 4;
    constexpr uint STRIDE = 8;
    const dim3 BlockSize = dim3{COL_PER_BLOCK / 4, STRIDE, 1};
    const dim3 gridSize = dim3{v_num,
                               (dim + COL_PER_BLOCK - 1) / COL_PER_BLOCK,
                               1};

    cudaMemset(d_Y, 0, v_num * dim * sizeof(float)); 
    cuda::gemm_4_AX<COL_PER_BLOCK, STRIDE> <<<gridSize, BlockSize>>>(d_csrA, 
                                                   d_X,
                                                   d_Y,
                                                   v_num,
                                                   dim);
    cudaDeviceSynchronize();
}


void launch_kernel_AX_Relu( CSRGraph_t d_csrA,
                            const f32* d_X,
                            f32* d_Y,
                            const uint v_num,
                            const uint dim)
{
    constexpr bool Relu = true;
    constexpr uint COL_PER_BLOCK = 16 * 4;
    constexpr uint STRIDE = 8;
    const dim3 BlockSize = dim3{COL_PER_BLOCK / 4, STRIDE, 1};
    const dim3 gridSize = dim3{v_num,
                               (dim + COL_PER_BLOCK - 1) / COL_PER_BLOCK,
                               1};

    cudaMemset(d_Y, 0, v_num * dim * sizeof(float)); 
    cuda::gemm_4_AX<COL_PER_BLOCK, STRIDE, Relu> <<<gridSize, BlockSize>>>(d_csrA, 
                                                        d_X,
                                                        d_Y,
                                                        v_num,
                                                        dim);
    cudaDeviceSynchronize();
}



}

