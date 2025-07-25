#ifndef AX_CUDA_PREPARE_H
#define AX_CUDA_PREPARE_H

#include <cassert>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>

inline uint v_num = 1024; // 图节点数 
inline uint dim = 256; // A:(v_num*v_num) X:(v_num,dim)
using f32 = float;

namespace cpu {
    struct CSRGraph_t {
        std::vector<int> col_indices;      // 边的列索引
        std::vector<int> row_indices;
        std::vector<int> index_pointers;   // 每个节点的邻接边范围
        std::vector<float> data;           // 边权
    };
}

namespace cuda {
    struct CSRGraph_t {
        int* col_indices;      // 边的列索引
        int* row_indices;
        f32* data;             // 边权
    };
}

template<bool Directed = false>
inline std::vector<int> make_raw_graph(std::size_t v_num,          // raw graph 节点数
                                       double sparse_ratio = 0.05, // 稀疏度
                                       unsigned seed = 42)
{
    std::vector<int> raw_graph;

    // 生成自环
    for (std::size_t i = 0; i < v_num; ++i)
    {
        raw_graph.push_back(static_cast<int>(i));
        raw_graph.push_back(static_cast<int>(i));
    }

    // 生成额外边
    std::size_t extraEdgeCnt =
        static_cast<std::size_t>(v_num * sparse_ratio);

    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> dist(0, static_cast<int>(v_num - 1));
    for (std::size_t k = 0; k < extraEdgeCnt; ++k)
    {
        int src = dist(rng);
        int dst = dist(rng);
        raw_graph.push_back(src);
        raw_graph.push_back(dst);

        if constexpr (!Directed)
        {
            // 如果是无向图，添加反向边
            raw_graph.push_back(dst);
            raw_graph.push_back(src);
        }
    }

    // 打乱顺序
    std::shuffle(reinterpret_cast<int(*)[2]>(raw_graph.data()),
                  reinterpret_cast<int(*)[2]>(raw_graph.data() + raw_graph.size()),
                    rng);

    return raw_graph;
}

inline f32 edgeNormalization(const int in_degree_dst, const int out_degree_src) {
    if (in_degree_dst == 0 || out_degree_src == 0)
        return 0.0f; 
    return 1.0f / (std::sqrt(static_cast<f32>(in_degree_dst)) * 
                   std::sqrt(static_cast<f32>(out_degree_src)));
}


inline cpu::CSRGraph_t RawGraph2CSR(const std::vector<int> &raw_graph,
                               const std::size_t       v_num)      // raw graph 节点数
                  
{
    assert(raw_graph.size() % 2 == 0);
    const uint edge_num = raw_graph.size() / 2;

    std::vector<int> out_degree(v_num, 0); 
    std::vector<int> in_degree(v_num, 0); 
    
    std::vector<int> col_indices(edge_num); 
    std::vector<int> row_indices(edge_num);
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
        col_indices[index] = dst;
        row_indices[index] = src;

        data[index] = edgeNormalization(in_degree[dst], out_degree[src]);
    }

    return cpu::CSRGraph_t{col_indices, row_indices, index_pointers, data};
}


template<typename T, bool random = false>
T* alloc(int n) {
    T *p = (float *)aligned_alloc(64, n * sizeof(T));
    if constexpr (random)
        for (int i = 0; i < n; ++i) p[i] = (T)(rand()) / static_cast<f32>(RAND_MAX);
    return p;
}


namespace cpu {
    inline void gemm_4_AX(const cpu::CSRGraph_t &A_csr, // raw_graph : (v_num * v_num)
                          const f32*            X,      // X : (v_num * dim)
                          f32*                  Y,      // Y : (v_num * dim)
                          const uint            dim,
                          const uint            v_num) 
    {   
        // #pragma omp parallel for
        for (uint i = 0; i < v_num; ++i) {
            const uint start = A_csr.index_pointers[i];
            const uint end = A_csr.index_pointers[i + 1];
            f32* const Y_row = Y + i * dim;
        
            #pragma omp simd
            for (uint j = 0; j < dim; ++j) {
                Y_row[j] = 0.0f;
            }    

            for (uint k = start; k < end; ++k) {
                const uint A_col_index = A_csr.col_indices[k];
                const f32 A_val = A_csr.data[k];
                const f32* X_row = &X[A_col_index * dim];
                
                #pragma omp simd
                for (uint j = 0; j < dim; ++j) {
                    Y_row[j] += A_val * X_row[j];
                }
            }
        }
    }
}


#endif // AX_CUDA_PREPARE_H