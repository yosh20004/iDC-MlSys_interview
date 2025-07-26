#pragma once
#include "prepare.h"

namespace cuda {
    __global__
    void gemm_4_AX_v1(const CSRGraph_t d_csrA, // v_num * v_num 
                      const f32* X,            // v_num * dim
                      f32* Y,                  // v_num * dim
                      const uint v_num,
                      const uint dim);
    
    __global__
    void gemm_4_AX_v2(const CSRGraph_t d_csrA, // v_num * v_num 
                      const f32* X,            // v_num * dim
                      f32* Y,                  // v_num * dim
                      const uint v_num,
                      const uint dim);

    __global__
    void gemm_4_AX_v3(const CSRGraph_t d_csrA, // v_num * v_num 
                      const f32* X,            // v_num * dim
                      f32* Y,                  // v_num * dim
                      const uint v_num,
                      const uint dim);

    __global__
    void gemm_4_AX_v4(const CSRGraph_t d_csrA, 
                      const f32* X,            
                      f32* Y,                  
                      const uint v_num,
                      const uint dim);

    enum class version { v1, v2, v3, v4 };

    template<version v>
    void launch_kernel(CSRGraph_t d_csrA,
                      const uint nnz, 
                      const f32* d_X,
                      f32* d_Y,
                      const uint v_num,
                      const uint dim);

    template<>
    void launch_kernel<version::v1>(CSRGraph_t d_csrA,
                                   const uint nnz, 
                                   const f32* d_X,
                                   f32* d_Y,
                                   const uint v_num,
                                   const uint dim);
    template<>
    void launch_kernel<version::v2>(CSRGraph_t d_csrA,
                                   const uint nnz, 
                                   const f32* d_X,
                                   f32* d_Y,
                                   const uint v_num,
                                   const uint dim);
    template<>
    void launch_kernel<version::v3>(CSRGraph_t d_csrA,
                                   const uint nnz, 
                                   const f32* d_X,
                                   f32* d_Y,
                                   const uint v_num,
                                   const uint dim);

    template<>
    void launch_kernel<version::v4>(CSRGraph_t d_csrA,
                                   const uint nnz, 
                                   const f32* d_X,
                                   f32* d_Y,
                                   const uint v_num,
                                   const uint dim);
}