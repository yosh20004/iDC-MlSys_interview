#include "kernel.cuh"
#include "prepare.h"


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
    __global__ void gemm_4_AX_v4(const CSRGraph_t d_csrA, // v_num * v_num 
                                 const f32* X,            // v_num * dim
                                 f32* Y,                  // v_num * dim
                                 const uint v_num,
                                 const uint dim)
    {
        const uint Y_row_index = blockIdx.x;
        const uint Y_col_index = blockIdx.y * 16 + threadIdx.y;
        const uint stride = blockDim.x;
        const uint Y_line_size = blockDim.y;
        if (Y_col_index >= dim) 
            return;

        const uint start_index = d_csrA.index_pointers[Y_row_index];
        const uint end_index = d_csrA.index_pointers[Y_row_index + 1];
        __shared__ f32 buffer[16][16];
        buffer[threadIdx.y][threadIdx.x] = 0.0f;
        __syncthreads();

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

            buffer[threadIdx.y][threadIdx.x] += A_ele * X_ele;
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            f32 sum = buffer[threadIdx.y][0] + 
                      buffer[threadIdx.y][1] + 
                      buffer[threadIdx.y][2] + 
                      buffer[threadIdx.y][3] +
                      buffer[threadIdx.y][4] +
                      buffer[threadIdx.y][5] +
                      buffer[threadIdx.y][6] +
                      buffer[threadIdx.y][7] +
                      buffer[threadIdx.y][8] +
                      buffer[threadIdx.y][9] +
                      buffer[threadIdx.y][10] +
                      buffer[threadIdx.y][11] +
                      buffer[threadIdx.y][12] +
                      buffer[threadIdx.y][13] +
                      buffer[threadIdx.y][14] +
                      buffer[threadIdx.y][15];

            if (Y_col_index < dim) {
                Y[Y_row_index * dim + Y_col_index] = sum;
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
                                   const uint dim)
    {
        const dim3 BlockSize = 512;
        const dim3 gridSize = dim3{ nnz,
                                   1,
                                   (dim + BlockSize.x - 1) / BlockSize.x};
        cudaMemset(d_Y, 0, v_num * dim * sizeof(float)); 
        for (int i = 0; i < TIMES; ++i) {
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
                                   const uint dim)
    {
        const dim3 BlockSize = 512;
        const dim3 gridSize = dim3{ nnz,
                                   1,
                                   (dim + BlockSize.x - 1) / BlockSize.x};
        cudaMemset(d_Y, 0, v_num * dim * sizeof(float)); 
        for (int i = 0; i < TIMES; ++i) {
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
                                   const uint dim)
    {
        const dim3 BlockSize = 16;
        const dim3 gridSize = dim3{v_num,
                                   dim, 
                                   1};
        cudaMemset(d_Y, 0, v_num * dim * sizeof(float)); 
        for (int i = 0; i < 10; ++i) {
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
                                   const uint dim)
    {
        const dim3 BlockSize = dim3{16, 16, 1};
        const dim3 gridSize = dim3{v_num,
                                   (dim + 16 - 1) / 16,
                                   1};
        cudaMemset(d_Y, 0, v_num * dim * sizeof(float)); 
        for (int i = 0; i < TIMES; ++i) {
            cudaMemset(d_Y, 0, v_num * dim * sizeof(float)); 
            cuda::gemm_4_AX_v4<<<gridSize, BlockSize>>>(d_csrA, 
                                                       d_X,
                                                       d_Y,
                                                       v_num,
                                                       dim);
            cudaDeviceSynchronize();
        }
    }
}