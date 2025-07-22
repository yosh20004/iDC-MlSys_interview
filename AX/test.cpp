#include <cassert>
#include <cmath>
#include "prepare.h"
#include <chrono>
#include <cstring>
constexpr int TIMES = 2;

void AX_test(const CSRGraph &A_csr, 
             const f32*      X,
             f32*            Y,
             const uint      v_num) 
{
    const auto& indices = A_csr.indices;
    const auto& data = A_csr.data;
    const auto& index_pointers = A_csr.index_pointers;

    std::memset(Y, 0, v_num * v_num * sizeof(f32));

    for (uint i = 0; i < v_num; ++i) {
        // 第i行的A非零元素列索引范围为[start, end)
        const uint start = index_pointers[i];
        const uint end = index_pointers[i + 1];

        for (uint j = 0; j < v_num; ++j) {
            for (uint k = start; k < end; ++k) {
                const uint A_col_index = indices[k];
                const f32 A_val = data[k];
                const f32 X_val = X[A_col_index * v_num + j];
                Y[i * v_num + j] += A_val * X_val;
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