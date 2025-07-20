#include <cassert>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <iostream>

uint v_num = 1024; // 图节点数

struct CSRGraph {
    std::vector<int> indices;          // 边的列索引
    std::vector<int> index_pointers;   // 每个节点的邻接边范围
    std::vector<float> data;           // 边权
};

template<bool Directed = false>
std::vector<int> make_raw_graph(std::size_t v_num,          // raw graph 节点数
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


// 仅用于无向图
CSRGraph RawGraph2CSR(const std::vector<int> &raw_graph,
                     const std::size_t        v_num)      // raw graph 节点数
                  
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

        data[index] = 1.0f / (std::sqrt(in_degree[dst]) * std::sqrt(out_degree[src]));
    }

    return CSRGraph{indices, index_pointers, data};
}



int main() {
    auto raw_graph = make_raw_graph(v_num);
    auto csr_graph = RawGraph2CSR(raw_graph, v_num);
    for (auto i : csr_graph.index_pointers) {
        std::cout << i << " ";
    }
    std::cout << std::endl;
    for (auto i : csr_graph.indices) {
        std::cout << i << " ";  
    }
    std::cout << std::endl;
    for (auto i : csr_graph.data) {
        std::cout << i << " ";
    }
}