#include "prepare.h"
#include <cuda_runtime.h>


namespace cuda {
    __global__ void gemm_4_AX(const CSRGraph_t d_csrA, // v_num * v_num 
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


int main() {
    const auto raw_graph = make_raw_graph(v_num); // A:(v_num, v_num)
    const cpu::CSRGraph_t csrA = RawGraph2CSR(raw_graph, v_num);

    const auto X = alloc<f32, true>(v_num * dim); // X:(v_num, dim)
    const uint nnz = csrA.data.size();

    f32* d_X = nullptr;
    f32* d_Y = nullptr;
    int* d_col_indices = nullptr;
    int* d_row_indices = nullptr;
    f32* d_data = nullptr;

    cudaMalloc((void**)&d_X, v_num * dim * sizeof(f32));
    cudaMalloc((void**)&d_col_indices, csrA.col_indices.size() * sizeof(int));
    cudaMalloc((void**)&d_row_indices, csrA.row_indices.size() * sizeof(int));
    cudaMalloc((void**)&d_data, csrA.data.size() * sizeof(f32));
    cudaMalloc((void**)&d_Y, v_num * dim * sizeof(f32));

    cudaMemcpy(d_X, X, v_num * dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_indices, csrA.col_indices.data(), nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_indices, csrA.row_indices.data(), nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_data, csrA.data.data(), nnz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_Y, 0, v_num * dim * sizeof(float)); 

    cuda::CSRGraph_t d_csrA = {d_col_indices, 
                               d_row_indices,
                               d_data};

    const dim3 BlockSize = 512;
    const dim3 gridSize = dim3{ nnz,
                                1,
                                (dim + BlockSize.x - 1) / BlockSize.x};
    
    cuda::gemm_4_AX<<<gridSize, BlockSize>>>(d_csrA, 
                                             d_X,
                                             d_Y,
                                             v_num,
                                             dim           );

    cudaDeviceSynchronize();

    cudaError_t err_msg = cudaGetLastError();
    if (err_msg != cudaSuccess) {
        printf("err : %s", cudaGetErrorString(err_msg));
    }

    f32* h_Y;
    f32* correct_Y;
    
    h_Y = alloc<f32>(v_num * dim);
    correct_Y = alloc<f32>(v_num * dim);
    cpu::gemm_4_AX(csrA, X, correct_Y, dim, v_num);
    
    double err = 0;
    cudaMemcpy(h_Y, d_Y, v_num * dim * sizeof(f32), cudaMemcpyDeviceToHost);
    for (uint i = 0; i < v_num * dim; ++i) {
        err = err > std::abs(h_Y[i] - correct_Y[i]) ? 
              err : std::abs(h_Y[i] - correct_Y[i]);
    }
    std::printf("max diff = %.3e\n", err);
}