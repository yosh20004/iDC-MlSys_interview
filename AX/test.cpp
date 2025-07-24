#include <cassert>
#include <cmath>
#include "prepare.h"
#include <chrono>
#include <cstring>
#include <omp.h>
constexpr int TIMES = 1000;

void AX_test(const CSRGraph &A_csr, // raw_graph : (v_num * v_num)
             const f32*      X,     // X : (v_num * dim)
             f32*            Y,     // Y : (v_num * dim)
             const uint      dim,
             const uint      v_num) 
{   
    // A行主序遍历
    #pragma omp parallel for
    for (uint i = 0; i < v_num; ++i) {
        const uint start = A_csr.index_pointers[i];
        const uint end = A_csr.index_pointers[i + 1];
        f32* const Y_row = Y + i * dim;
    
        #pragma omp simd
        for (uint j = 0; j < dim; ++j) {
            Y_row[j] = 0.0f;
        }    

        for (uint k = start; k < end; ++k) {
            const uint A_col_index = A_csr.indices[k];
            const f32 A_val = A_csr.data[k];
            const f32* X_row = &X[A_col_index * dim];
            
            #pragma omp simd
            for (uint j = 0; j < dim; ++j) {
                Y_row[j] += A_val * X_row[j];
            }
        }
    }
}


int main() {
    auto raw_graph = make_raw_graph(v_num);
    const auto X = alloc<f32, true>(v_num * v_num);
    auto A_csr = RawGraph2CSR(raw_graph, v_num);
    auto Y_mkl = alloc<f32>(v_num * v_num);
    auto Y_self = alloc<f32>(v_num * v_num);

    auto warmup = [&A_csr, &X, &Y_mkl, &Y_self]() -> void
    {
        // 预热
        for (int i=0; i < 100; ++i) {
            gemm_IntelMKL(A_csr, X, Y_mkl, v_num);
            AX_test(A_csr, X, Y_self, v_num);
        }
    };

    warmup();
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i=0; i < TIMES; ++i) {
        gemm_IntelMKL(A_csr, X, Y_mkl, v_num);
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    warmup();
    auto t3 = std::chrono::high_resolution_clock::now();
    for (int i=0; i < TIMES; ++i) {
        AX_test(A_csr, X, Y_self, v_num);
    }
    auto t4 = std::chrono::high_resolution_clock::now();

    double err = 0;
    for (uint i = 0; i < v_num * v_num; ++i)
        err = err > std::abs(Y_mkl[i] - Y_self[i]) ? 
              err : std::abs(Y_mkl[i] - Y_self[i]);

    std::printf("max diff = %.3e\n", err);
    std::printf("Intel MKL = %.3f ms\n",
                std::chrono::duration<double, std::milli>(t2 - t1).count());
    std::printf("My impl2  = %.3f ms\n",
                std::chrono::duration<double, std::milli>(t4 - t3).count());

    free(X);
    free(Y_mkl);
    free(Y_self);
}