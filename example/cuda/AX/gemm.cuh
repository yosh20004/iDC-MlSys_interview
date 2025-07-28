#pragma once

#include <vector>
#include <sys/types.h>
#include <cuda_runtime.h>
#include "cpu/AX/common.h"

using f32 = float;

namespace cuda 
{
    
struct CSRGraph_t {
    int* index_pointers;
    int* col_indices;      
    f32* data;           
};

inline cuda::CSRGraph_t host2device(const ::CSRGraph_t& csrA) {
    cuda::CSRGraph_t ret = {nullptr, nullptr, nullptr};
    
    // Allocate memory on device
    cudaMalloc((void**)&ret.col_indices, csrA.indices.size() * sizeof(int));
    cudaMalloc((void**)&ret.index_pointers, csrA.index_pointers.size() * sizeof(int));
    cudaMalloc((void**)&ret.data, csrA.data.size() * sizeof(f32));

    // Copy data from host to device
    cudaMemcpy(ret.col_indices, csrA.indices.data(), 
               csrA.indices.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ret.index_pointers, csrA.index_pointers.data(), 
               csrA.index_pointers.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ret.data, csrA.data.data(), 
               csrA.data.size() * sizeof(f32), cudaMemcpyHostToDevice);

    return ret;
}   

}


namespace cuda {
    enum class RunMode { Bench, Production };

    void launch_kernel_AX(CSRGraph_t d_csrA,
                         const f32* d_X,
                         f32* d_Y,
                         const uint v_num,
                         const uint dim);

    void launch_kernel_AX_Relu(CSRGraph_t d_csrA,
                               const f32* d_X,
                               f32* d_Y,
                               const uint v_num,
                               const uint dim);
}