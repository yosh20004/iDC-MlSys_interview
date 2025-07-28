#pragma  once


#include <vector>
using f32 = float;

struct CSRGraph_t {
    std::vector<int> indices;          // 边的列索引
    std::vector<int> index_pointers;   // 每个节点的邻接边范围
    std::vector<float> data;           // 边权
};
