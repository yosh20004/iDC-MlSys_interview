#include "kernel.cuh"
#include "prepare.h"
#include <cstdio>


namespace cuda {
    __global__ void gemm_4_AX_v1(const CSRGraph_t d_csrA, // v_num * v_num 
                                 const f32* X,            // v_num * dim
                                 f32* Y,                  // v_num * dim
                                 const uint v_num,
                                 const uint dim) 
    {   
        const uint global_id = blockIdx.x;
        const uint A_col_index = d_csrA.col_indices[global_id];
        const uint A_row_index = d_csrA.row_indices[global_id];
        const uint X_row_index = A_col_index;
        const f32 A_ele = d_csrA.data[global_id];

        const uint tid = threadIdx.x + blockDim.x * blockIdx.z;
        const uint X_col_index = tid;
        const uint Y_row_index = A_row_index;
        const uint Y_col_index = X_col_index;

        if (tid < dim) {
            const f32 X_ele = X[X_row_index * dim + X_col_index]; 
            const f32 tmp = A_ele * X_ele;
            atomicAdd(&Y[Y_row_index * dim + Y_col_index], tmp);
        } 
    }
}


namespace cuda {
    // this kernel requires BlockSize == 512
    __global__ void gemm_4_AX_v2(const CSRGraph_t d_csrA, // v_num * v_num 
                                 const f32* X,            // v_num * dim
                                 f32* Y,                  // v_num * dim
                                 const uint v_num,
                                 const uint dim)    
    {   
        const uint global_id = blockIdx.x;
        const uint A_col_index = d_csrA.col_indices[global_id];
        const uint A_row_index = d_csrA.row_indices[global_id];
        const uint X_row_index = A_col_index;
        const f32 A_ele = d_csrA.data[global_id];

        const uint tid = threadIdx.x + blockDim.x * blockIdx.z;
        const uint X_col_index = tid;
        const uint Y_row_index = A_row_index;

        __shared__ f32 Y_local[512];
        if (tid >= dim) 
            return;

        const f32 X_ele = X[X_row_index * dim + X_col_index]; 
        const f32 tmp = A_ele * X_ele;
        Y_local[threadIdx.x] = tmp;

        // 只有一个线程负责访存，速度太慢了
        if (threadIdx.x == 0) {
            for (uint i = 0; i < blockDim.x; ++i) {
                atomicAdd(&Y[Y_row_index * dim + i + blockDim.x * blockIdx.z], Y_local[i]);
            }
        }
    }
}


namespace cuda {
    template<typename T, typename Op>
    __device__ T block_reduce(T* smem, Op op)
    {
        int tid = threadIdx.x;
        T val   = smem[tid];
        
        for (int s = blockDim.x / 2; s > 32; s >>= 1) {
            if (tid < s) smem[tid] = op(smem[tid], smem[tid + s]);
            __syncthreads();
        }

        if (tid < 32) val = (blockDim.x > 32) ? smem[tid] : val;
        for (int offset = 16; offset > 0; offset >>= 1) {
            T other = __shfl_down_sync(0xffffffff, val, offset);
            val = op(val, other);
        }

        return val;
    }


    // this kernel requires BlockSize == 16
    __global__ void gemm_4_AX_v3(const CSRGraph_t d_csrA, // v_num * v_num 
                                 const f32* X,            // v_num * dim
                                 f32* Y,                  // v_num * dim
                                 const uint v_num,
                                 const uint dim)
    {
        const uint Y_row_index = blockIdx.x;
        const uint Y_col_index = blockIdx.y;
        const uint stride = blockDim.x;

        const uint start_index = d_csrA.index_pointers[Y_row_index];
        const uint end_index = d_csrA.index_pointers[Y_row_index + 1];

        __shared__ f32 buffer[16];
        buffer[threadIdx.x] = 0.0f;

        for (uint i = start_index; 
             i < end_index; 
             i += stride) {
            
            const uint local_index = i + threadIdx.x;
            if (local_index >= end_index) 
                continue;
            
            const uint A_col_index = d_csrA.col_indices[local_index];
            const uint X_row_index = A_col_index;
            const uint X_col_index = Y_col_index;

            const f32 A_ele = d_csrA.data[local_index];
            const f32 X_ele = X[X_row_index * dim + X_col_index];

            buffer[threadIdx.x] += A_ele * X_ele;
        }

        float sum = block_reduce(buffer, 
                      [] __device__ (f32 a, f32 b) { return a + b; });

        if (threadIdx.x == 0)
            Y[Y_row_index * dim + Y_col_index] += sum;
    }        
}



namespace cuda {
    template<uint COL_PER_BLOCK, uint STRIDE>
    __global__ void gemm_4_AX_v4(const CSRGraph_t d_csrA, // v_num * v_num 
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


namespace cuda {
    template<uint COL_PER_BLOCK, uint STRIDE>
    __global__ void gemm_4_AX_v5(const CSRGraph_t d_csrA, // v_num * v_num 
                                 const f32* X,            // v_num * dim
                                 f32* Y,                  // v_num * dim
                                 const uint v_num,
                                 const uint dim,          // dim must be a multiple of 4 !!
                                 uint* global_A_row_index)          
    {
        static_assert(COL_PER_BLOCK % 4 == 0, "COL_PER_BLOCK must be a multiple of 4");
        
        __shared__ uint Y_row_index;
        __shared__ float4 buffer[STRIDE][COL_PER_BLOCK / 4 + 1];

        while (true) {
            if (threadIdx.x == 0 && threadIdx.y == 0) {
                int tmp = atomicAdd(&global_A_row_index[blockIdx.y], 1);
                Y_row_index = tmp;      
            }
            __syncthreads();
            if (Y_row_index >= v_num) 
                return;
    
            float4 local_sum = {0.0f, 0.0f, 0.0f, 0.0f};
            const uint Y_col_index = blockIdx.y * COL_PER_BLOCK + threadIdx.x * 4;
            
            if (Y_col_index < dim) { 
                const uint start_index = d_csrA.index_pointers[Y_row_index];
                const uint end_index = d_csrA.index_pointers[Y_row_index + 1];

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
            }
                
            __syncthreads();
            (float4& )buffer[threadIdx.y][threadIdx.x] = local_sum;
            __syncthreads();

            if (Y_col_index < dim) { 
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
    }
}






namespace cuda {
    template<>
    void launch_kernel<version::v1>(CSRGraph_t d_csrA,
                                   const uint nnz, 
                                   const f32* d_X,
                                   f32* d_Y,
                                   const uint v_num,
                                   const uint dim, 
                                   RunMode mode)
    {
        if (mode == RunMode::Bench)
            printf("BENCHMARKING Version : v1\n");
        const dim3 BlockSize = 512;
        const dim3 gridSize = dim3{ nnz,
                                   1,
                                   (dim + BlockSize.x - 1) / BlockSize.x};
        cudaMemset(d_Y, 0, v_num * dim * sizeof(float)); 
        std::printf("nnz = %d\n", nnz);
        for (int i = 0; i < GET_RUN_TIMES(mode); ++i) {
            cudaMemset(d_Y, 0, v_num * dim * sizeof(float)); 
            cuda::gemm_4_AX_v1<<<gridSize, BlockSize>>>(d_csrA, 
                                                       d_X,
                                                       d_Y,
                                                       v_num,
                                                       dim);
            cudaDeviceSynchronize();
        }
    }
}


namespace cuda {
    template<>
    void launch_kernel<version::v2>(CSRGraph_t d_csrA,
                                   const uint nnz, 
                                   const f32* d_X,
                                   f32* d_Y,
                                   const uint v_num,
                                   const uint dim,
                                   RunMode mode)
    {
        if (mode == RunMode::Bench)
            printf("BENCHMARKING Version : v2\n");
        const dim3 BlockSize = 512;
        const dim3 gridSize = dim3{ nnz,
                                   1,
                                   (dim + BlockSize.x - 1) / BlockSize.x};
        cudaMemset(d_Y, 0, v_num * dim * sizeof(float)); 
        for (int i = 0; i < GET_RUN_TIMES(mode); ++i) {
            cudaMemset(d_Y, 0, v_num * dim * sizeof(float)); 
            cuda::gemm_4_AX_v2<<<gridSize, BlockSize>>>(d_csrA, 
                                                       d_X,
                                                       d_Y,
                                                       v_num,
                                                       dim);
            cudaDeviceSynchronize();
        }
    }
}



namespace cuda {
    template<>
    void launch_kernel<version::v3>(CSRGraph_t d_csrA,
                                   const uint nnz, 
                                   const f32* d_X,
                                   f32* d_Y,
                                   const uint v_num,
                                   const uint dim,
                                   RunMode mode)
    {
        if (mode == RunMode::Bench)
            printf("BENCHMARKING Version : v3\n");
        const dim3 BlockSize = 16;
        const dim3 gridSize = dim3{v_num,
                                   dim, 
                                   1};
        cudaMemset(d_Y, 0, v_num * dim * sizeof(float)); 
        for (int i = 0; i < GET_RUN_TIMES(mode); ++i) {
            cudaMemset(d_Y, 0, v_num * dim * sizeof(float)); 
            cuda::gemm_4_AX_v3<<<gridSize, BlockSize>>>(d_csrA, 
                                                       d_X,
                                                       d_Y,
                                                       v_num,
                                                       dim);
            cudaDeviceSynchronize();
        }
    }
}



namespace cuda {
    template<>
    void launch_kernel<version::v4>(CSRGraph_t d_csrA,
                                   const uint nnz, 
                                   const f32* d_X,
                                   f32* d_Y,
                                   const uint v_num,
                                   const uint dim,
                                   RunMode mode)
    {
        if (mode == RunMode::Bench)
            printf("BENCHMARKING Version : v4\n");
        constexpr uint COL_PER_BLOCK = 16 * 4;
        constexpr uint STRIDE = 8;
        const dim3 BlockSize = dim3{COL_PER_BLOCK / 4, STRIDE, 1};
        const dim3 gridSize = dim3{v_num,
                                   (dim + COL_PER_BLOCK - 1) / COL_PER_BLOCK,
                                   1};

        cudaMemset(d_Y, 0, v_num * dim * sizeof(float)); 
        for (int i = 0; i < GET_RUN_TIMES(mode); ++i) {
            cudaMemset(d_Y, 0, v_num * dim * sizeof(float)); 
            cuda::gemm_4_AX_v4<COL_PER_BLOCK, STRIDE> <<<gridSize, BlockSize>>>(d_csrA, 
                                                       d_X,
                                                       d_Y,
                                                       v_num,
                                                       dim);
            cudaDeviceSynchronize();
        }
    }
}



namespace cuda {
    template<>
    void launch_kernel<version::v5>(CSRGraph_t d_csrA,
                                   const uint nnz, 
                                   const f32* d_X,
                                   f32* d_Y,
                                   const uint v_num,
                                   const uint dim,
                                   RunMode mode)
    {
        if (mode == RunMode::Bench)
            printf("BENCHMARKING Version : v5\n");
        constexpr uint COL_PER_BLOCK = 16 * 4;
        constexpr uint STRIDE = 8;

        uint* global_A_row_index = nullptr;
        cudaMalloc(&global_A_row_index, ((dim + COL_PER_BLOCK - 1) / COL_PER_BLOCK) * sizeof(uint));

        const dim3 BlockSize = dim3{COL_PER_BLOCK / 4, STRIDE, 1};
        const dim3 gridSize = dim3{v_num / 8,
                                   (dim + COL_PER_BLOCK - 1) / COL_PER_BLOCK,
                                   1};

        cudaMemset(d_Y, 0, v_num * dim * sizeof(float)); 
        for (int i = 0; i < GET_RUN_TIMES(mode); ++i) {
            cudaMemset(d_Y, 0, v_num * dim * sizeof(float)); 
            cudaMemset(global_A_row_index, 0, gridSize.y * sizeof(uint));
            cuda::gemm_4_AX_v5<COL_PER_BLOCK, STRIDE> <<<gridSize, BlockSize>>>(d_csrA, 
                                                       d_X,
                                                       d_Y,
                                                       v_num,
                                                       dim,
                                                       global_A_row_index);
            cudaDeviceSynchronize();
        }
        cudaFree(global_A_row_index);
    }
}



namespace cuda {
    template<>
    void launch_kernel<version::cuSPARSE>(CSRGraph_t d_csrA,
                                   const uint nnz, 
                                   const f32* d_X,
                                   f32* d_Y,
                                   const uint v_num,
                                   const uint dim,
                                   RunMode mode)
    {
        cusparseHandle_t handle;
        cusparseSpMatDescr_t matA;
        cusparseDnMatDescr_t matX, matY;
        void* dBuffer = nullptr;
        
        auto init = [&]() {
            cusparseCreate(&handle);
            cusparseCreateCsr(&matA, v_num, v_num, nnz,
                             d_csrA.index_pointers,
                             d_csrA.col_indices,
                             d_csrA.data,
                             CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                             CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
            cusparseCreateDnMat(&matX, v_num, dim, dim, (void*)d_X,
                               CUDA_R_32F, CUSPARSE_ORDER_ROW);
            cusparseCreateDnMat(&matY, v_num, dim, dim, (void*)d_Y,
                               CUDA_R_32F, CUSPARSE_ORDER_ROW);
            
            size_t bufferSize = 0;
            const float alpha = 1.0f;
            const float beta = 0.0f;
            cusparseSpMM_bufferSize(handle, 
                                   CUSPARSE_OPERATION_NON_TRANSPOSE,
                                   CUSPARSE_OPERATION_NON_TRANSPOSE,
                                   &alpha, matA, matX, &beta, matY,
                                   CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT,
                                   &bufferSize);
            cudaMalloc(&dBuffer, bufferSize);
        };

        auto cleanup = [&]() {
            cudaFree(dBuffer);
            cusparseDestroySpMat(matA);
            cusparseDestroyDnMat(matX);
            cusparseDestroyDnMat(matY);
            cusparseDestroy(handle);
        };
        
        init();
        cudaMemset(d_Y, 0, v_num * dim * sizeof(float));

        const float alpha = 1.0f;
        const float beta = 0.0f;
        
        for (int i = 0; i < GET_RUN_TIMES(mode); ++i) {
            cusparseSpMM(handle,
                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &alpha, matA, matX, &beta, matY,
                        CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT,
                        dBuffer);
        }
        cudaDeviceSynchronize();
        
        cleanup();
    }
}