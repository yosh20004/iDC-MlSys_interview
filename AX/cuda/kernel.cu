#include "kernel.cuh"


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