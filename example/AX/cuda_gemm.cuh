#pragma once

#include <iostream>
#include "common.h"

namespace cuda {
    void gemm_4_AX(const CSRGraph_t &A_csr, // raw_graph : (v_num * v_num)
                const f32*      X,       // X : (v_num * dim)
                f32*            Y,       // Y : (v_num * dim)
                const uint      dim,
                const uint      v_num);
}