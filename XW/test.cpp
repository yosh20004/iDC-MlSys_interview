// test_xw_minimal.cpp
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cstring>

#ifdef USE_BLAS
#include <cblas.h>
#endif

constexpr int M = 1024;   // 行
constexpr int K = 64;     // 内
constexpr int N = 16;     // 列
constexpr int TIMES = 1000;

using f32 = float;

template<typename T>
T* alloc(int n) {
    T *p = (float *)aligned_alloc(64, n * sizeof(T));
    for (int i = 0; i < n; ++i) p[i] = (T)(rand()) / RAND_MAX;
    return p;
}


template<bool Transpose = false>
void gemm_kernel(const float *A_pack,
                 const float *B_pack,
                 float *C_block, 
                 int ldc,       
                 int mc_real, 
                 int nc_real, 
                 int kc_real) {

    // 朴素子块乘法：A(mc,kc) * B(kc,nc) → C(mc,nc)
    if constexpr (!Transpose) {
        for (int i = 0; i < mc_real; ++i)
            for (int j = 0; j < nc_real; ++j) {
                float sum = 0.f;
                for (int k = 0; k < kc_real; ++k) 
                    sum += A_pack[i * kc_real + k] * 
                           B_pack[k * nc_real + j];
                C_block[i * ldc + j] += sum;
            }
        return;
    }



}

// A(M,K) * B(K,N) → C(M,N)
void gemm_test(const float *A, const float *B, float *C) {
    constexpr uint cacheSize = 16;
    f32* A_cache = alloc<f32>(cacheSize);
    f32* B_cache = alloc<f32>(cacheSize);

    int mc = 1;
    int kc = cacheSize / mc;
    int nc = 1;  // mc = nc

    for (int i = 0; i < M; i += mc) {
        for (int j = 0; j < N; j += nc) {
            for (int k = 0; k < K; k += kc) {

                // load A_cache
                for (int ii = 0; ii < mc; ++ii) {
                    for (int kk = 0; kk < kc; ++kk) {
                        A_cache[ii * kc + kk] = A[(i + ii) * K + (k + kk)];
                    }
                }

                // load B_cache
                for (int ii = 0; ii < kc; ++ii) {
                    for (int jj = 0; jj < nc; ++jj) {
                        B_cache[ii * nc + jj] = B[(k + ii) * N + (j + jj)];
                    }
                }

                // call gemm_kernel
                gemm_kernel(A_cache, B_cache, 
                           C + i * N + j, N, mc, nc, kc);
            }
            
        }
    }
}

// native
void gemm_native(const float *A, const float *B, float *C) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] += sum;
        }
    }
}

// OpenBLAS
void gemm_blas(const float *A, const float *B, float *C) {
#ifdef USE_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
#else
    gemm_native(A, B, C);
#endif
}

int main() {
    srand(1234);                // 固定随机种子
    float *A = alloc<f32>(M * K);
    float *B = alloc<f32>(K * N);
    float *C1 = alloc<f32>(M * N);
    float *C2 = alloc<f32>(M * N);
    float *C3 = alloc<f32>(M * N);

    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i=0; i < TIMES; ++i) {
        gemm_blas(A, B, C1);       // OpenBLAS
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    for (int i=0; i < TIMES; ++i) {
        gemm_test(A, B, C2);       // test
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    for (int i=0; i < TIMES; ++i) {
        gemm_native(A, B, C3);     // native
    }

    {
        std::memset(C2, 0, M * N * sizeof(float));
        gemm_test(A, B, C2); 
    }

    double err = 0;
    for (int i = 0; i < M * N; ++i)
        err = err > std::abs(C1[i] - C2[i]) ? err : std::abs(C1[i] - C2[i]);
    std::printf("max diff = %.3e\n", err);
    std::printf("OpenBLAS = %.3f ms\n",
                std::chrono::duration<double, std::milli>(t2 - t1).count());
    std::printf("My impl  = %.3f ms\n",
                std::chrono::duration<double, std::milli>(t3 - t2).count());
    std::printf("Native   = %.3f ms\n",
                std::chrono::duration<double, std::milli>(t3 - t1).count());

    free(A); free(B); free(C1); free(C2); free(C3);
    return 0;
}