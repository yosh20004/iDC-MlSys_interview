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
}