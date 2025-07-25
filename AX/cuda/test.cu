#include "prepare.h"
#include "kernel.cuh"
#include <cuda_runtime.h>
#include <chrono>

typedef std::chrono::time_point<std::chrono::steady_clock> TimePoint;

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
    
    cudaDeviceSynchronize();
    TimePoint t1 = std::chrono::steady_clock::now();
    for (int i = 0; i < TIMES; ++i) {
        cudaMemset(d_Y, 0, v_num * dim * sizeof(float)); 
        cuda::gemm_4_AX_v2<<<gridSize, BlockSize>>>(d_csrA, 
                                                    d_X,
                                                    d_Y,
                                                    v_num,
                                                    dim           );

        cudaDeviceSynchronize();
    }
    TimePoint t2 = std::chrono::steady_clock::now();

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
    std::printf("avg_time : %.3f ms\n",
                std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() / (double)TIMES);
}