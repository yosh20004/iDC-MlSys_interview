#include <cassert>
#include <cmath>
#include "prepare.h"
#include <chrono>
constexpr int TIMES = 100;

void AX_test(const CSRGraph &A_csr, 
             const f32*      X,
             f32*            Y,
             const uint      v_num) 
{

            
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
        err = err > std::abs(Y_mkl - Y_self) ? err : std::abs(Y_mkl - Y_self);

    std::printf("max diff = %.3e\n", err);
    std::printf("Intel MKL = %.3f ms\n",
                std::chrono::duration<double, std::milli>(t2 - t1).count());
    std::printf("My impl2  = %.3f ms\n",
                std::chrono::duration<double, std::milli>(t4 - t3).count());

    free(X);
    free(Y_mkl);
    free(Y_self);
}