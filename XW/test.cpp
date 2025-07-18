// test_xw_minimal.cpp
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cstring>
#include <utility>
#include <memory>

#ifdef USE_BLAS
#include <cblas.h>
#endif

constexpr int M = 1024;   // 行
constexpr int K = 64;     // 内
constexpr int N = 1024;     // 列
constexpr int TIMES = 300;

using f32 = float;

template<typename T>
T* alloc(int n) {
    T *p = (float *)aligned_alloc(64, n * sizeof(T));
    for (int i = 0; i < n; ++i) p[i] = (T)(rand()) / RAND_MAX;
    return p;
}
struct AlignedDeleter {
    void operator()(void* p) const {
        free(p);  // aligned_alloc 使用 free 释放
    }
};


[[gnu::always_inline]] void gemm_kernel_4x4(const float* __restrict__ A_L1_local,
                     const float* __restrict__ B_L1_local,
                     float* __restrict__ C_block, 
                     int ldc)
{
    // 4x4 子块乘法：A(4,4) * B(4,4) → C(4,4)
    constexpr int stride = 4;
    for (int i = 0; i < stride; ++i)
        for (int j = 0; j < stride; ++j) {
            float sum = 0.f;
            for (int k = 0; k < stride; ++k) 
                sum += A_L1_local[i * stride + k] * 
                       B_L1_local[k * stride + j];
            C_block[i * ldc + j] += sum;
        }
}


void gemm_kernel(const float* __restrict__ A_pack,
                 const float* __restrict__ B_pack,
                 float* __restrict__ C_block, 
                 int ldc,       
                 int mc_real, 
                 int nc_real, 
                 int kc_real) {
                    
    constexpr int stride = 4; // 步长
    auto A_L1_local = std::unique_ptr<f32[], AlignedDeleter>(alloc<f32>(stride * stride));
    auto B_L1_local = std::unique_ptr<f32[], AlignedDeleter>(alloc<f32>(stride * stride));

    // 朴素子块乘法：A(mc,kc) * B(kc,nc) → C(mc,nc)
    for (int i = 0; i < mc_real; i += stride)
        for (int j = 0; j < nc_real; j += stride)
            for (int k = 0; k < kc_real; k += stride) {
                // load A_L1_local
                for (int ii = 0; ii < stride; ++ii) {
                    for (int jj = 0; jj < stride; ++jj) {
                        A_L1_local[ii * stride + jj] = 
                            A_pack[(i + ii) * kc_real + k + jj];
                    }
                }

                // load B_L1_local
                for (int ii = 0; ii < stride; ++ii) {
                    for (int jj = 0; jj < stride; ++jj) {
                        B_L1_local[ii * stride + jj] = 
                            B_pack[(k + ii) * nc_real + j + jj];
                    }
                }

                // 4*4 子块乘法
                gemm_kernel_4x4(
                                A_L1_local.get(), 
                                B_L1_local.get(), 
                                C_block + i * N + j, ldc);
            }
}

// A(M,K) * B(K,N) → C(M,N)
void gemm_test(const float *A, const float *B, float *C) {
    constexpr int mc = 16;
    constexpr int kc = 16;
    constexpr int nc = 128;  
    constexpr int A_cache_size = mc * kc;
    constexpr int B_cache_size = kc * nc;

    static f32* A_cache = alloc<f32>(A_cache_size);
    static f32* B_cache = alloc<f32>(B_cache_size);

    for (int j = 0; j < N; j += nc) {
        for (int k = 0; k < K; k += kc) {
            // load B_cache
            for (int ii = 0; ii < kc; ++ii) {
                for (int jj = 0; jj < nc; ++jj) {
                    B_cache[ii * nc + jj] = B[(k + ii) * N + (j + jj)];
                }
            }

            for (int i = 0; i < M; i += mc) {
                // load A_cache
                for (int ii = 0; ii < mc; ++ii) {
                    for (int kk = 0; kk < kc; ++kk) {
                        A_cache[ii * kc + kk] = A[(i + ii) * K + (k + kk)];
                    }
                }

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

    {
        // 预热
        for (int i=0; i < 50; ++i) {
            gemm_blas(A, B, C1); 
            std::memset(C2, 0, M * N * sizeof(float));
            gemm_test(A, B, C2); 
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i=0; i < TIMES; ++i) {
        gemm_blas(A, B, C1);       // OpenBLAS
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    for (int i=0; i < TIMES; ++i) {
        std::memset(C2, 0, M * N * sizeof(float));
        gemm_test(A, B, C2);     // test
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    // for (int i=0; i < TIMES; ++i) {
    //     std::memset(C3, 0, M * N * sizeof(float));
    //     //gemm_native(A, B, C3);     // native
    // }
    // auto t4 = std::chrono::high_resolution_clock::now();

    double err = 0;
    for (int i = 0; i < M * N; ++i)
        err = err > std::abs(C1[i] - C2[i]) ? err : std::abs(C1[i] - C2[i]);
    std::printf("max diff = %.3e\n", err);
    std::printf("OpenBLAS = %.3f ms\n",
                std::chrono::duration<double, std::milli>(t2 - t1).count());
    std::printf("My impl1  = %.3f ms\n",
                std::chrono::duration<double, std::milli>(t3 - t2).count());
    // std::printf("Native   = %.3f ms\n",
    //             std::chrono::duration<double, std::milli>(t4 - t3).count());

    free(A); free(B); free(C1); free(C2); free(C3);
    return 0;
}