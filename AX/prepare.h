#ifndef AX_PREPARE_H
#define AX_PREPARE_H

#include <cassert>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include "mkl_spblas.h"

inline uint v_num = 1024; // 图节点数
using f32 = float;

struct CSRGraph {
    std::vector<int> indices;          // 边的列索引
    std::vector<int> index_pointers;   // 每个节点的邻接边范围
    std::vector<float> data;           // 边权
};

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


inline CSRGraph RawGraph2CSR(const std::vector<int> &raw_graph,
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

    return CSRGraph{indices, index_pointers, data};
}


template<typename T, bool random = false>
T* alloc(int n) {
    T *p = (float *)aligned_alloc(64, n * sizeof(T));
    if constexpr (random)
        for (int i = 0; i < n; ++i) p[i] = (T)(rand()) / RAND_MAX;
    return p;
}


inline void gemm_IntelMKL(const CSRGraph &A_csr,
                          const float *X,
                          float *Y,
                          MKL_INT v_num)
{
    // 1. 构造 MKL 稀疏矩阵描述符
    sparse_matrix_t A_mkl = nullptr;
    mkl_sparse_s_create_csr(
        &A_mkl,
        SPARSE_INDEX_BASE_ZERO,        // 0-based 索引
        v_num, v_num,                // rows, cols
        const_cast<MKL_INT *>(A_csr.index_pointers.data()),      // row_start
        const_cast<MKL_INT *>(A_csr.index_pointers.data() + 1),    // row_end
        const_cast<MKL_INT *>(A_csr.indices.data()),             // col_indx
        const_cast<float *>(A_csr.data.data())                   // values
    );

    // 2. 分析并优化矩阵（一次即可）
    mkl_sparse_optimize(A_mkl);

    // 3. 执行 SpMM:  Y = 1.0*A*X + 0.0*Y
    const float alpha = 1.0f, beta = 0.0f;
    struct matrix_descr descrA;
    descrA.type = SPARSE_MATRIX_TYPE_GENERAL;

    // MKL 要求 X, Y 按列主序；我们的 X/Y 是行主序，可以用转置标志
    mkl_sparse_s_mm(
        SPARSE_OPERATION_NON_TRANSPOSE,   // A 不转置
        alpha,
        A_mkl,
        descrA,
        SPARSE_LAYOUT_ROW_MAJOR,            // 稠密矩阵按行主序
        X,
        v_num,                            // 稠密矩阵的列数
        v_num,                            // 稠密矩阵的 ldX (leading dimension)
        beta,
        Y,
        v_num                             // ldY
    );

    // 4. 释放资源
    mkl_sparse_destroy(A_mkl);
}


[[deprecated]] inline void naive_AX(const f32* A,
                     const f32* X,
                     f32* Y,
                     const std::size_t v_num) {
    for (uint i = 0; i < v_num; ++i) {
        for (uint j = 0; j < v_num; ++j) {
            Y[i * v_num + j] = 0.0f;
            for (uint k = 0; k < v_num; ++k) {
                Y[i * v_num + j] += A[i * v_num + k] * X[k * v_num + j];
            }
        }
    }
}


[[deprecated]] inline std::vector<f32> RawGraph2Matrix(const std::vector<int> &raw_graph,
                                        const std::size_t v_num)      // raw graph 节点数
{
    assert(raw_graph.size() % 2 == 0);
    const uint edge_num = raw_graph.size() / 2;
    std::vector<int> in_degree(v_num, 0);
    std::vector<int> out_degree(v_num, 0);

    for (uint i = 0; i < edge_num; ++i)
    {
        int src = raw_graph[2 * i];
        int dst = raw_graph[2 * i + 1];
        out_degree[src]++;
        in_degree[dst]++;
    }

    std::vector<f32> matrix(v_num * v_num, 0.0f);

    for (uint i = 0; i < edge_num; ++i)
    {
        int src = raw_graph[2 * i];
        int dst = raw_graph[2 * i + 1];
        matrix[src * v_num + dst] = edgeNormalization(in_degree[dst], 
                                                     out_degree[src]);
    }

    return matrix;
}





#endif // AX_PREPARE_H