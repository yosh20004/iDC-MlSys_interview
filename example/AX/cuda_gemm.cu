#include "cuda_gemm.cuh"
#include <cusparse.h>

namespace cuda {
    void gemm_4_AX(const CSRGraph_t &A_csr, // raw_graph : (v_num * v_num)
                const f32*      X,       // X : (v_num * dim)
                f32*            Y,       // Y : (v_num * dim)
                const uint      dim,
                const uint      v_num) 
    {
        // 1. 获取矩阵维度信息
        const int m = v_num;
        const int n = dim;
        const int k = v_num;
        const int nnz = A_csr.data.size();

        // 2. 在设备(GPU)上分配内存
        int* d_A_row_pointers;
        int* d_A_indices;
        f32* d_A_data;
        f32* d_X;
        f32* d_Y;

        cudaMalloc(&d_A_row_pointers, A_csr.index_pointers.size() * sizeof(int));
        cudaMalloc(&d_A_indices, nnz * sizeof(int));
        cudaMalloc(&d_A_data, nnz * sizeof(f32));
        cudaMalloc(&d_X, (size_t)k * n * sizeof(f32));
        cudaMalloc(&d_Y, (size_t)m * n * sizeof(f32));

        // 3. 将数据从主机(CPU)拷贝到设备(GPU)
        cudaMemcpy(d_A_row_pointers, A_csr.index_pointers.data(), A_csr.index_pointers.size() * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_A_indices, A_csr.indices.data(), nnz * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_A_data, A_csr.data.data(), nnz * sizeof(f32), cudaMemcpyHostToDevice);
        cudaMemcpy(d_X, X, (size_t)k * n * sizeof(f32), cudaMemcpyHostToDevice);
        
        // 4. cuSPARSE 计算
        cusparseHandle_t handle;
        cusparseCreate(&handle);

        // 创建矩阵描述符
        cusparseSpMatDescr_t matA;
        cusparseDnMatDescr_t matB, matC;

        // 描述稀疏矩阵 A (m x k)
        cusparseCreateCsr(&matA, m, k, nnz,
                        d_A_row_pointers, d_A_indices, d_A_data,
                        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
        
        // 描述稠密矩阵 B (X) (k x n)，行主序
        cusparseCreateDnMat(&matB, k, n, n, d_X, CUDA_R_32F, CUSPARSE_ORDER_ROW);
        
        // 描述稠密矩阵 C (Y) (m x n)，行主序
        cusparseCreateDnMat(&matC, m, n, n, d_Y, CUDA_R_32F, CUSPARSE_ORDER_ROW);

        // 执行 SpMM (Y = 1.0 * A * X + 0.0 * Y)
        const f32 alpha = 1.0f;
        const f32 beta  = 0.0f;
        void* d_buffer = nullptr;
        size_t buffer_size = 0;

        cusparseSpMM_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                CUSPARSE_SPMM_ALG_DEFAULT, &buffer_size);
        cudaMalloc(&d_buffer, buffer_size);

        cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                    CUSPARSE_SPMM_ALG_DEFAULT, d_buffer);

        // 5. 将结果从设备拷贝回主机
        cudaMemcpy(Y, d_Y, (size_t)m * n * sizeof(f32), cudaMemcpyDeviceToHost);
        
        // 6. 释放所有资源
        cudaFree(d_buffer);
        cudaFree(d_A_row_pointers);
        cudaFree(d_A_indices);
        cudaFree(d_A_data);
        cudaFree(d_X);
        cudaFree(d_Y);
        
        cusparseDestroySpMat(matA);
        cusparseDestroyDnMat(matB);
        cusparseDestroyDnMat(matC);
        cusparseDestroy(handle);
    }
} // namespace cuda