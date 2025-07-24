#pragma once

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cstring>
#include <immintrin.h>
#include <stdexcept>
#include <omp.h>

const int NUM_THREADS = 1;

using f32 = float;

template<typename T, bool random = false>
T* alloc(int n) {
    T *p = (float *)aligned_alloc(64, n * sizeof(T));
    if constexpr (random)
        for (int i = 0; i < n; ++i) p[i] = (T)(rand()) / RAND_MAX;
    return p;
}

template<uint strideX, uint strideY, uint strideZ>
inline void gemm_micro_kernel_Btransposed(
                     const float* __restrict__ A_micro,
                     const float* __restrict__ B_micro, //B需要被转置
                     float*       __restrict__ C_micro)
{
    // 子块乘法：A(X,Y) * B(Z,Y) → C(X,Z)
    for (uint i = 0; i < strideX; ++i) {
        for (uint j = 0; j < strideZ; ++j) {
            float sum = 0.f;
            for (uint k = 0; k < strideY; ++k) 
                sum += A_micro[i * strideY + k] * 
                       B_micro[j * strideY + k];
            C_micro[i * strideZ + j] += sum;
        }
    }
}

inline void gemm_L1_kernel(const float* __restrict__ A_L1_pack,
                    const float* __restrict__ B_L1_pack,
                    float*       __restrict__ C_L1_pack, 
                    const                uint mc_real, 
                    const                uint nc_real, 
                    const                uint kc_real) {
                    
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
                // A_L1_pack[i + ii][k + jj] → Alocal[ii][jj]
                for (uint ii = 0; ii < strideX; ++ii) {
                    for (uint jj = 0; jj < strideY; jj += 8) {
                        __m256 A_L1_local_line = _mm256_loadu_ps(A_L1_pack + (i + ii) * kc_real + k + jj);
                        _mm256_storeu_ps(A_L1_local + ii * strideY + jj, A_L1_local_line);
                    }
                }

                // B_L1_pack[k + ii][j + jj] → Blocal[jj][ii]
                for (uint ii = 0; ii < strideY; ++ii) {
                    for (uint jj = 0; jj < strideZ; ++jj) {
                        B_L1_local[jj * strideY + ii] = 
                            B_L1_pack[(k + ii) * nc_real + j + jj];
                    }
                }

                // 4*4 子块乘法
                gemm_micro_kernel_Btransposed<strideX, strideY, strideZ>(
                                A_L1_local, 
                                B_L1_local, 
                                C_L1_local);
            }

            // 写回 C_L1_pack
            for (uint ii = 0; ii < strideX; ++ii) {
                __m128 C_new = _mm_loadu_ps(C_L1_local + ii * strideZ);
                __m128 C_origin = _mm_loadu_ps(C_L1_pack + (i + ii) * nc_real + j);
                C_new = _mm_add_ps(C_origin, C_new);
                _mm_storeu_ps(C_L1_pack + (i + ii) * nc_real + j, C_new);
            }
        }
}

// A(M,K) * B(K,N) → C(M,N)
inline void gemm_4_XW(const float *A, 
               const float *B, 
               float       *C,
               int          M,
               int          K,
               int          N) 
{
    assert(M % 8 == 0 && K % 8 == 0 && N % 8 == 0);

    const int mc = M >= 64? 64 : M;
    const int kc = K >= 64? 64 : K;
    const int nc = N >= 16? 16 : N;  
    const int A_L1_pack_size = mc * kc;
    const int B_L1_pack_size = kc * nc;
    const int C_L1_pack_size = mc * nc;

    omp_set_num_threads(NUM_THREADS);

    #pragma omp parallel
    {
        f32* A_L1_pack = alloc<f32>(A_L1_pack_size);
        f32* B_L1_pack = alloc<f32>(B_L1_pack_size);
        f32* C_L1_pack = alloc<f32>(C_L1_pack_size);

        #pragma omp for schedule(static)
        for (int i = 0; i < M; i += mc) {
            for (int j = 0; j < N; j += nc) {
                std::memset(C_L1_pack, 0, C_L1_pack_size * sizeof(f32));

                for (int k = 0; k < K; k += kc) {
                    // load A_L1_pack (左上角坐标为(i, k))
                    for (int ii = 0; ii < mc; ++ii) {
                        for (int jj = 0; jj < kc; jj += 8) {
                            __m256 A_L1_pack_line = _mm256_loadu_ps(A + (i + ii) * K + k + jj);
                            _mm256_storeu_ps(A_L1_pack + ii * kc + jj, A_L1_pack_line);
                        }
                    }

                    // load B_L1_pack (左上角坐标为(k, j))
                    for (int ii = 0; ii < kc; ++ii) {
                        for (int jj = 0; jj < nc; jj += 8) {
                            __m256 B_L1_pack_line = _mm256_loadu_ps(B + (k + ii) * N + j + jj);
                            _mm256_storeu_ps(B_L1_pack + ii * nc + jj, B_L1_pack_line);
                        }
                    }

                    gemm_L1_kernel(A_L1_pack, B_L1_pack, 
                                   C_L1_pack, mc, nc, kc);
                }

                for (int ii = 0; ii < mc; ++ii) {
                    for (int jj = 0; jj < nc; ++jj) {
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

