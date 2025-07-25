#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cstring>
#include <immintrin.h>
#include <stdexcept>
#include <omp.h>


#ifdef USE_BLAS
#include <cblas.h>
#endif

constexpr int M = 1024;   // 行
constexpr int K = 64;     // 内
constexpr int N = 1024;     // 列
constexpr int TIMES = 100;

using f32 = float;

template<typename T, bool random = false>
T* alloc(int n) {
    T *p = (float *)aligned_alloc(64, n * sizeof(T));
    if constexpr (random)
        for (int i = 0; i < n; ++i) p[i] = (T)(rand()) / RAND_MAX;
    return p;
}


template<uint strideX, uint strideY, uint strideZ>
inline void gemm_micro_kernel(
                     const float* __restrict__ A_micro,
                     const float* __restrict__ B_micro,
                     float*       __restrict__ C_micro)
{
    // 子块乘法：A(X,Y) * B(Y,Z) → C(X,Z)
    for (uint i = 0; i < strideX; ++i) {
        for (uint j = 0; j < strideZ; ++j) {
            for (uint k = 0; k < strideY; ++k) {
                C_micro[i * strideZ + j] += 
                    A_micro[i * strideY + k] * 
                    B_micro[k * strideZ + j];
            }
        }
    }
}

void gemm_L1_kernel(const float* __restrict__ A_L1_pack,
                    const float* __restrict__ B_L1_pack,
                    float*       __restrict__ C_L1_pack, 
                    const                uint mc_real, 
                    const                uint nc_real, 
                    const                uint kc_real) {
                    
    constexpr int strideX = 8;
    constexpr int strideY = 8; 
    constexpr int strideZ = 8; 
    alignas(64) f32 A_L1_local[strideX * strideY] = {}; // ASize = X * Y
    alignas(64) f32 B_L1_local[strideZ * strideY] = {}; // BSize = Z * Y
    alignas(64) f32 C_L1_local[strideX * strideZ] = {}; // CSize = X * Z
    static_assert(strideX % 4 == 0 && strideY % 8 == 0 && strideZ % 4 == 0);

    // 子块乘法：A(mc,kc) * B(kc,nc) → C(mc,nc)
    for (uint i = 0; i < mc_real; i += strideX)
        for (uint j = 0; j < nc_real; j += strideZ) {
            std::memset(C_L1_local, 0, sizeof(C_L1_local));

            for (uint k = 0; k < kc_real; k += strideY) {
                // A_L1_pack[i + ii][k + jj] → Alocal[ii][jj]
                for (uint ii = 0; ii < strideX; ++ii) {
                    for (uint jj = 0; jj < strideY; jj += 8) {
                        __m256 A_L1_local_line = _mm256_loadu_ps(A_L1_pack + (i + ii) * kc_real + k + jj);
                        _mm256_storeu_ps(A_L1_local + ii * strideY + jj, A_L1_local_line);
                    }
                }

                // B_L1_pack[k + ii][j + jj] → Blocal[ii][jj]
                for (uint ii = 0; ii < strideY; ++ii) {
                    for (uint jj = 0; jj < strideZ; jj += 8) {
                        __m256 B_L1_local_line = _mm256_loadu_ps(B_L1_pack + (k + ii) * nc_real + j + jj);
                        _mm256_storeu_ps(B_L1_local + ii * strideZ + jj, B_L1_local_line);
                    }
                }

                // micro 子块乘法
                gemm_micro_kernel<strideX, strideY, strideZ>(
                                A_L1_local, 
                                B_L1_local, 
                                C_L1_local);
            }

            // 写回 C_L1_pack
            # pragma omp simd
            for (uint ii = 0; ii < strideX; ++ii) {
                for (uint jj = 0; jj < strideZ; jj += 1) {
                    C_L1_pack[(i + ii) * nc_real + j + jj] += 
                        C_L1_local[ii * strideZ + jj];

                }
            }
        }
}

// A(M,K) * B(K,N) → C(M,N)
void gemm_test(const float *A, 
               const float *B, 
               float       *C,
               int          M,
               int          K,
               int          N) 
{
    assert(M % 8 == 0 && K % 8 == 0 && N % 8 == 0);

    constexpr int mc = 64;
    constexpr int kc = 64;
    constexpr int nc = 64;  
    const int A_L1_pack_size = mc * kc;
    const int B_L1_pack_size = kc * nc;
    const int C_L1_pack_size = mc * nc;

    #pragma omp parallel
    { 
        f32 *A_L1_pack = alloc<f32>(A_L1_pack_size);
        f32 *B_L1_pack = alloc<f32>(B_L1_pack_size);
        f32 *C_L1_pack = alloc<f32>(C_L1_pack_size);

        #pragma omp for schedule(static)
        for (int i = 0; i < M; i += mc)
        {
            for (int j = 0; j < N; j += nc)
            {
                std::memset(C_L1_pack, 0, C_L1_pack_size * sizeof(f32));
                for (int k = 0; k < K; k += kc)
                {
                    // load A_L1_pack (左上角坐标为(i, k))
                    for (int ii = 0; ii < mc; ++ii)
                    {
                        for (int jj = 0; jj < kc; jj += 1) {
                            A_L1_pack[ii * kc + jj] = A[(i + ii) * K + (k + jj)];
                        }
                    }

                    // load B_L1_pack (左上角坐标为(k, j))
                    for (int ii = 0; ii < kc; ++ii)
                    {
                        for (int jj = 0; jj < nc; jj += 1) {
                            B_L1_pack[ii * nc + jj] = B[(k + ii) * N + (j + jj)];
                        }
                    }

                    gemm_L1_kernel(A_L1_pack, B_L1_pack,
                                C_L1_pack, mc, nc, kc);
                }

                for (int ii = 0; ii < mc; ++ii)
                {
                    for (int jj = 0; jj < nc; ++jj)
                    {
                        C[(i + ii) * N + j + jj] = C_L1_pack[ii * nc + jj];
                    }
                }
            }
        }

        free(A_L1_pack);
        free(B_L1_pack);
        free(C_L1_pack);
    }
}


// OpenBLAS
void gemm_blas(const float *A, const float *B, float *C) {
#ifdef USE_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
#else
    throw std::runtime_error("OpenBLAS is not enabled.");
#endif
}

int main() {
    srand(1234);                // 固定随机种子
    float *A = alloc<f32, true>(M * K);
    float *B = alloc<f32, true>(K * N);
    float *C1 = alloc<f32, true>(M * N);
    float *C2 = alloc<f32, true>(M * N);
    float *C3 = alloc<f32, true>(M * N);

    auto warmup = [&A, &B, &C1, &C2]() -> void
    {
        // 预热
        for (int i=0; i < 100; ++i) {
            gemm_test(A, B, C2, M, K, N); // test
        }
    };

    warmup();  
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i=0; i < TIMES; ++i) {
        gemm_blas(A, B, C1);       // OpenBLAS
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    warmup();
    auto t3 = std::chrono::high_resolution_clock::now();
    for (int i=0; i < TIMES; ++i) {
        gemm_test(A, B, C2, M, K, N);     // test
    }
    auto t4 = std::chrono::high_resolution_clock::now();

    double err = 0;
    for (int i = 0; i < M * N; ++i)
        err = err > std::abs(C1[i] - C2[i]) ? err : std::abs(C1[i] - C2[i]);
    std::printf("max diff = %.3e\n", err);
    std::printf("OpenBLAS = %.3f ms\n",
                std::chrono::duration<double, std::milli>(t2 - t1).count());
    std::printf("My impl1  = %.3f ms\n",
                std::chrono::duration<double, std::milli>(t4 - t3).count());
    // std::printf("Native   = %.3f ms\n",
    //             std::chrono::duration<double, std::milli>(t4 - t3).count());

    free(A); free(B); free(C1); free(C2); free(C3);
    return 0;
}
