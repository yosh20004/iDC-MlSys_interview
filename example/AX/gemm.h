#pragma once

#include <cassert>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>

using f32 = float;

struct CSRGraph_t {
    std::vector<int> indices;          // 边的列索引
    std::vector<int> index_pointers;   // 每个节点的邻接边范围
    std::vector<float> data;           // 边权
};


inline f32 edgeNormalization(const int in_degree_dst, const int out_degree_src) {
    if (in_degree_dst == 0 || out_degree_src == 0)
        return 0.0f; 
    return 1.0f / (std::sqrt(static_cast<f32>(in_degree_dst)) * 
                   std::sqrt(static_cast<f32>(out_degree_src)));
}


inline CSRGraph_t RawGraph2CSR(const std::vector<int> &raw_graph,
                             const std::size_t       v_num)      // raw graph 节点数
                  
{
    assert(raw_graph.size() % 2 == 0);
    const uint edge_num = raw_graph.size() / 2;

    std::vector<int> out_degree(v_num, 0); 
    std::vector<int> in_degree(v_num, 0); 
    
    std::vector<int> indices(edge_num); 
    std::vector<int> index_pointers(v_num + 1, 0);
    
    for (uint i = 0; i < edge_num; ++i)
    {
        int src = raw_graph[2 * i];
        int dst = raw_graph[2 * i + 1];
        out_degree[src]++;
        in_degree[dst]++;
    }

    for (uint i = 0; i < v_num; ++i)
    {
        index_pointers[i + 1] = index_pointers[i] + out_degree[i];
    }

    std::vector<int> tmp(v_num, 0);
    std::vector<float> data(edge_num);
    for (uint i = 0; i < edge_num; ++i)
    {
        int src = raw_graph[2 * i];
        int dst = raw_graph[2 * i + 1];
        int index = index_pointers[src] + tmp[src]++;
        indices[index] = dst;

        data[index] = edgeNormalization(in_degree[dst], out_degree[src]);
    }

    return CSRGraph_t{indices, index_pointers, data};
}


void gemm_4_AX(const CSRGraph_t &A_csr, // raw_graph : (v_num * v_num)
               const f32*      X,       // X : (v_num * dim)
               f32*            Y,       // Y : (v_num * dim)
               const uint      dim,
               const uint      v_num) 
{   
    // A行主序遍历
    #pragma omp parallel for
    for (uint i = 0; i < v_num; ++i) {
        const uint start = A_csr.index_pointers[i];
        const uint end = A_csr.index_pointers[i + 1];
        f32* const Y_row = Y + i * dim;
    
        #pragma omp simd
        for (uint j = 0; j < dim; ++j) {
            Y_row[j] = 0.0f;
        }    

        for (uint k = start; k < end; ++k) {
            const uint A_col_index = A_csr.indices[k];
            const f32 A_val = A_csr.data[k];
            const f32* X_row = &X[A_col_index * dim];
            
            #pragma omp simd
            for (uint j = 0; j < dim; ++j) {
                Y_row[j] += A_val * X_row[j];
            }
        }
    }
}


