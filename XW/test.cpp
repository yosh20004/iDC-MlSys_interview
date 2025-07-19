#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cstring>
#include <immintrin.h>
#include <stdexcept>

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
inline void gemm_kernel_Btransposed(
                     const float* __restrict__ A_L1_local,
                     const float* __restrict__ B_L1_local, //B需要被转置
                     float* __restrict__ C_L1_local, 
                     int ldc)
{
    // 子块乘法：A(X,Y) * B(Z,Y) → C(X,Z)
    for (uint i = 0; i < strideX; ++i) {
        for (uint j = 0; j < strideZ; ++j) {
            float sum = 0.f;
            for (uint k = 0; k < strideY; ++k) 
                sum += A_L1_local[i * strideY + k] * 
                       B_L1_local[j * strideY + k];
            C_L1_local[i * strideZ + j] += sum;
        }
    }
}

void gemm_kernel(const float* __restrict__ A_pack,
                 const float* __restrict__ B_pack,
                 float* __restrict__ C_block, 
                 int ldc,       
                 uint mc_real, 
                 uint nc_real, 
                 uint kc_real) {
                    
    constexpr int strideX = 8;
    constexpr int strideY = 8; 
    constexpr int strideZ = 4; 
    f32 A_L1_local[strideX * strideY] = {}; // ASize = X * Y
    f32 B_L1_local[strideZ * strideY] = {}; // BSize = Z * Y
    f32 C_L1_local[strideX * strideZ] = {}; // CSize = X * Z
    static_assert(strideX % 4 == 0 && strideY % 8 == 0 && strideZ % 4 == 0);

    // 子块乘法：A(mc,kc) * B(kc,nc) → C(mc,nc)
    for (uint i = 0; i < mc_real; i += strideX)
        for (uint j = 0; j < nc_real; j += strideZ) {

            std::memset(C_L1_local, 0, sizeof(C_L1_local));
            for (uint k = 0; k < kc_real; k += strideY) {

                // load A_L1_local (Alocal左上角的坐标为(i, k))
                // A_pack(X, Y) → Alocal(X, Y)
                // A_pack[i + ii][k + jj] → Alocal[ii][jj]
                for (uint ii = 0; ii < strideX; ++ii) {
                    for (uint jj = 0; jj < strideY; jj += 8) {
                        __m256 A_L1_local_line = _mm256_loadu_ps(A_pack + (i + ii) * kc_real + k + jj);
                        _mm256_storeu_ps(A_L1_local + ii * strideY + jj, A_L1_local_line);
                    }
                }

                // load B_L1_local (Blocal左上角的坐标为(k, j))
                // B_pack(Y, Z) → Blocal(Z, Y)
                // B_pack[k + ii][j + jj] → Blocal[jj][ii]
                for (uint ii = 0; ii < strideY; ++ii) {
                    for (uint jj = 0; jj < strideZ; ++jj) {
                        B_L1_local[jj * strideY + ii] = 
                            B_pack[(k + ii) * nc_real + j + jj];
                    }
                }

                // 4*4 子块乘法
                gemm_kernel_Btransposed<strideX, strideY, strideZ>(
                                A_L1_local, 
                                B_L1_local, 
                                C_L1_local, ldc);
            }

            // 将 C_L1_local 写回 C_block
            for (uint ii = 0; ii < strideX; ++ii) {
                __m128 C_L1_local_line = _mm_loadu_ps(C_L1_local + ii * strideZ);
                __m128 C_block_line = _mm_loadu_ps(C_block + (i + ii) * ldc + j);
                C_block_line = _mm_add_ps(C_block_line, C_L1_local_line);
                _mm_storeu_ps(C_block + (i + ii) * ldc + j, C_block_line);
            }
        }
}

// A(M,K) * B(K,N) → C(M,N)
void gemm_test(const float *A, const float *B, float *C) {
    constexpr int mc = 64;
    constexpr int kc = 64;
    constexpr int nc = 64;  
    constexpr int A_cache_size = mc * kc;
    constexpr int B_cache_size = kc * nc;

    static f32* A_cache = alloc<f32>(A_cache_size);
    static f32* B_cache = alloc<f32>(B_cache_size);

    for (int i = 0; i < M; i += mc) {
        for (int j = 0; j < K; j += kc) {
            // load A_cache
            for (int ii = 0; ii < mc; ++ii) {
                for (int jj = 0; jj < kc; ++jj) {
                    A_cache[ii * kc + jj] = A[(i + ii) * K + (j + jj)];
                }
            }

            for (int k = 0; k < N; k += nc) {
                // load B_cache
                for (int ii = 0; ii < kc; ++ii) {
                    for (int jj = 0; jj < nc; ++jj) {
                        B_cache[ii * nc + jj] = B[(j + ii) * N + (k + jj)];
                    }
                }

                gemm_kernel(A_cache, B_cache, 
                        C + i * N + k, N, mc, nc, kc);
            }
        }
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
        gemm_test(A, B, C2);     // test
    }
    auto t3 = std::chrono::high_resolution_clock::now();

    {
        // 比较计算结果
        std::memset(C2, 0, M * N * sizeof(float));
        gemm_test(A, B, C2);     // test
    }

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
