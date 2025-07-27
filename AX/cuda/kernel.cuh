#pragma once
#include "prepare.h"
#include <cusparse.h>

#define GET_RUN_TIMES(mode) ((mode) == RunMode::Bench ? TIMES : 1)

namespace cuda {
    enum class version { v1, v2, v3, v4, v5, cuSPARSE };
    enum class RunMode { Bench, Production };

    template<version v>
    void launch_kernel(CSRGraph_t d_csrA,
                      const uint nnz, 
                      const f32* d_X,
                      f32* d_Y,
                      const uint v_num,
                      const uint dim,
                      RunMode mode = RunMode::Production);
}