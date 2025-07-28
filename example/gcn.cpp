#include <stdio.h>
#include <vector>
#include <fstream>
#include <math.h>
#include <string.h>
#include <omp.h>
#include <iostream>
#include <chrono>
#include "cpu/XW/gemm.h"
#include "cpu/AX/cpu_gemm.h"

#include "cuda/AX/gemm.cuh"
#include "cuda/XW/gemm.h"

using namespace std;
typedef std::chrono::time_point<std::chrono::steady_clock> TimePoint;

int v_num = 0;
int e_num = 0;
int F0 = 0, F1 = 0, F2 = 0;

vector<vector<int>> edge_index;
vector<vector<float>> edge_val;
vector<int> degree;
vector<int> raw_graph;
CSRGraph_t csrGraph;
cuda::CSRGraph_t d_csrA;

float *X0, *W1, *W2, *X1, *X1_inter, *X2, *X2_inter;

// 新增：全局设备指针
float *d_X0 = nullptr;
float *d_W1 = nullptr;
float *d_X1_inter = nullptr;
float *d_X1 = nullptr;
float *d_W2 = nullptr;
float *d_X2_inter = nullptr;

void readGraph(char *fname)
{
	ifstream infile(fname);

	int source;
	int end;

	infile >> v_num >> e_num;

	// raw_graph.resize(e_num * 2);

	while (!infile.eof())
	{
		infile >> source >> end;
		if (infile.peek() == EOF)
			break;
		raw_graph.push_back(source);
		raw_graph.push_back(end);
	}
}

void raw_graph_to_AdjacencyList()
{

	int src;
	int dst;

	edge_index.resize(v_num);
	edge_val.resize(v_num);
	degree.resize(v_num, 0);

	for (int i = 0; i < raw_graph.size() / 2; i++)
	{
		src = raw_graph[2*i];
		dst = raw_graph[2*i + 1];
		edge_index[dst].push_back(src);
		degree[src]++;
	}
}

void edgeNormalization()
{
	for (int i = 0; i < v_num; i++)
	{
		for (int j = 0; j < edge_index[i].size(); j++)
		{
			float val = 1 / sqrt(degree[i]) / sqrt(degree[edge_index[i][j]]);
			edge_val[i].push_back(val);
		}
	}
}

void readFloat(char *fname, float *&dst, int num)
{
	dst = (float *)malloc(num * sizeof(float));
	FILE *fp = fopen(fname, "rb");
	fread(dst, num * sizeof(float), 1, fp);
	fclose(fp);
}

void initFloat(float *&dst, int num)
{
	dst = (float *)malloc(num * sizeof(float));
	memset(dst, 0, num * sizeof(float));
}

void XW(int in_dim, int out_dim, float *in_X, float *out_X, float *W)
{
	// auto tmp_in_X  = reinterpret_cast<float (*)[in_dim]>(in_X);
	// auto tmp_out_X = reinterpret_cast<float (*)[out_dim]>(out_X);
	// auto tmp_W     = reinterpret_cast<float (*)[out_dim]>(W);
	auto tmp_in_X  = in_X;
	auto tmp_out_X = out_X;
	auto tmp_W     = W;
	
	gemm_4_XW(tmp_in_X, tmp_W, tmp_out_X, v_num, in_dim, out_dim);
}

void AX(int dim, float *in_X, float *out_X)
{
	// float(*tmp_in_X)[dim] = (float(*)[dim])in_X;
	auto tmp_in_X = reinterpret_cast<float (*)[dim]>(in_X);
	auto tmp_out_X = reinterpret_cast<float (*)[dim]>(out_X);

	for (int i = 0; i < v_num; i++)
	{
		vector<int> &nlist = edge_index[i];
		for (int j = 0; j < nlist.size(); j++)
		{
			int nbr = nlist[j];
			for (int k = 0; k < dim; k++)
			{
				tmp_out_X[i][k] += tmp_in_X[nbr][k] * edge_val[i][j];
			}
		}
	}
}

void ReLU(int dim, float *X)
{
	for (int i = 0; i < v_num * dim; i++)
		if (X[i] < 0)
			X[i] = 0;
}

void LogSoftmax(int dim, float *X)
{
	auto tmp_X = reinterpret_cast<float (*)[dim]>(X);

#pragma omp parallel for
	for (int i = 0; i < v_num; i++)
	{
		float max = tmp_X[i][0];
		for (int j = 1; j < dim; j++)
		{
			if (tmp_X[i][j] > max)
				max = tmp_X[i][j];
		}

		float sum = 0;
		for (int j = 0; j < dim; j++)
		{
			sum += exp(tmp_X[i][j] - max);
		}
		sum = log(sum);

		for (int j = 0; j < dim; j++)
		{
			tmp_X[i][j] = tmp_X[i][j] - max - sum;
		}
	}
}

float MaxRowSum(float *X, int dim)
{
	auto tmp_X = reinterpret_cast<float (*)[dim]>(X);
	float max = -__FLT_MAX__;

#pragma omp parallel for reduction(max:max)
	for (int i = 0; i < v_num; i++)
	{
		float sum = 0;
		for (int j = 0; j < dim; j++)
		{
			sum += tmp_X[i][j];
		}
		if (sum > max)
			max = sum;
	}
	return max;
}

void freeFloats()
{
	free(X0);
	free(W1);
	free(W2);
	free(X2);
	free(X1_inter);
	free(X2_inter);
}

void somePreprocessing()
{
    csrGraph = RawGraph2CSR(raw_graph, v_num);    // host csr 
    d_csrA   = cuda::host2device(csrGraph);  // device csr

    cudaMalloc(&d_X0, v_num * F0 * sizeof(float));
    cudaMalloc(&d_W1, F0 * F1 * sizeof(float));
    cudaMalloc(&d_W2, F1 * F2 * sizeof(float));

    cudaMemcpy(d_X0, X0, v_num * F0 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W1, W1, F0 * F1 * sizeof(float),  cudaMemcpyHostToDevice);
    cudaMemcpy(d_W2, W2, F1 * F2 * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&d_X1_inter, v_num * F1 * sizeof(float));
    cudaMalloc(&d_X1, v_num * F1 * sizeof(float));
    cudaMemset(d_X1, 0, v_num * F1 * sizeof(float));

    cudaMalloc(&d_X2_inter, v_num * F2 * sizeof(float));

    // Add a synchronization point to warm up the GPU and ensure all initialization is complete
    cudaDeviceSynchronize();
}

void freeDeviceMemory() {
    cudaFree(d_X0);
    cudaFree(d_W1);
    cudaFree(d_W2);
    cudaFree(d_X1_inter);
    cudaFree(d_X1);
    cudaFree(d_X2_inter);
	cudaFree(d_csrA.index_pointers);
	cudaFree(d_csrA.col_indices);
	cudaFree(d_csrA.data);
}

int main(int argc, char **argv)
{
	// Do NOT count the time of reading files, malloc, and memset
	F0 = atoi(argv[1]);
	F1 = atoi(argv[2]);
	F2 = atoi(argv[3]);

	readGraph(argv[4]);
	readFloat(argv[5], X0, v_num * F0);
	readFloat(argv[6], W1, F0 * F1);
	readFloat(argv[7], W2, F1 * F2);

	initFloat(X1_inter, v_num * F1);
	initFloat(X2, v_num * F2);
	initFloat(X2_inter, v_num * F2);

	// Time point at the start of the computation
	TimePoint start = chrono::steady_clock::now();
	TimePoint prepross_start = chrono::steady_clock::now();
	somePreprocessing();  // RawGrpah -> CSR
						  // host -> device
	TimePoint prepross_end = chrono::steady_clock::now();
	chrono::duration<double> prepross_ = prepross_end - prepross_start;
	double prepross_time = prepross_.count() * 1e3;
	printf("prepross_time: %.8lf\n", prepross_time);

	// TimePoint edgeNorm_start = chrono::steady_clock::now();
	// edgeNormalization();
	// TimePoint edgeNorm_end = chrono::steady_clock::now();
	// chrono::duration<double> edgeNorm_ = edgeNorm_end - edgeNorm_start;
	// double edgeNorm_time = edgeNorm_.count() * 1e3;
	// printf("edgeNorm_time: %.8lf\n", edgeNorm_time);


	// printf("Layer1 XW\n");
	TimePoint XW1_start = chrono::steady_clock::now();
	// XW(F0, F1, X0, X1_inter, W1);
    cuda::launch_kernel_XW(v_num, F0, F1, d_X0, d_W1, d_X1_inter);
    // The result d_X1_inter is kept on the device for the next AX operation.
	TimePoint XW1_end = chrono::steady_clock::now();
	chrono::duration<double> XW1_ = XW1_end - XW1_start;
	double XW1_time = XW1_.count() * 1e3;
	printf("XW1_time: %.8lf\n", XW1_time);

	

	// printf("Layer1 AX\n");
	TimePoint Fused_AX1_Relu_start = chrono::steady_clock::now();
	// Launch kernel
	cuda::launch_kernel_AX_Relu(d_csrA, d_X1_inter, d_X1, v_num, F1);
	// Free device memory
	TimePoint Fused_AX1_Relu_end = chrono::steady_clock::now();
	chrono::duration<double> AX1_Relu = Fused_AX1_Relu_end - Fused_AX1_Relu_start;
	double AX1_Relu_time = AX1_Relu.count() * 1e3;

	printf("Fused AX1 + ReLU_time: %.8lf\n", AX1_Relu_time);



	// printf("Layer2 XW\n");	
	TimePoint XW2_start = chrono::steady_clock::now();
	// XW(F1, F2, X1, X2_inter, W2);
    cuda::launch_kernel_XW(v_num, F1, F2, d_X1, d_W2, d_X2_inter);
	TimePoint XW2_end = chrono::steady_clock::now();
	chrono::duration<double> XW2_ = XW2_end - XW2_start;
	double XW2_time = XW2_.count() * 1e3;
	printf("XW2_time: %.8lf\n", XW2_time);



	// printf("Layer2 AX\n");
	TimePoint AX2_start = chrono::steady_clock::now();
    cuda::launch_kernel_AX(d_csrA, d_X2_inter, d_X1, v_num, F2); // 复用d_X1作为输出缓冲区
	// 将结果拷贝回主机端X2
    cudaMemcpy(X2, d_X1, v_num * F2 * sizeof(float), cudaMemcpyDeviceToHost);
    // 释放设备内存
	TimePoint AX2_end = chrono::steady_clock::now();
	chrono::duration<double> AX2_ = AX2_end - AX2_start;
	double AX2_time = AX2_.count() * 1e3;
	printf("AX2_time: %.8lf\n", AX2_time);



	// printf("Layer2 LogSoftmax\n");
	TimePoint LogSoftmax_start = chrono::steady_clock::now();
	LogSoftmax(F2, X2);
	TimePoint LogSoftmax_end = chrono::steady_clock::now();
	chrono::duration<double> LogSoftmax_ = LogSoftmax_end - LogSoftmax_start;
	double LogSoftmax_time = LogSoftmax_.count() * 1e3;
	printf("LogSoftmax_time: %.8lf\n", LogSoftmax_time);



	// You need to compute the max row sum for result verification
	TimePoint max_sum_start = chrono::steady_clock::now();
	float max_sum = MaxRowSum(X2, F2);
	TimePoint max_sum_end = chrono::steady_clock::now();
	chrono::duration<double> max_sum_ = max_sum_end - max_sum_start;
	double max_sum_time = max_sum_.count() * 1e3;
	printf("max_sum_time: %.8lf\n", max_sum_time);

	// Time point at the end of the computation
	TimePoint end = chrono::steady_clock::now();
	chrono::duration<double> l_durationSec = end - start;
	double l_timeMs = l_durationSec.count() * 1e3;

	// Finally, the max row sum and the computing time
	// should be print to the terminal in the following format
	printf("%.8f\n", max_sum);
	printf("total time: %.8lf\n\n", l_timeMs);

	// Remember to free your allocated memory
	freeDeviceMemory();
	freeFloats();
}