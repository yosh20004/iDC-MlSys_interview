#ifndef XW_CUBLAS_CUH
#define XW_CUBLAS_CUH

#include <cublas_v2.h>
#include <cuda_runtime.h>

struct CuBlasHandleRAII {
    cublasHandle_t handle = nullptr;
    CuBlasHandleRAII()  { cublasCreate(&handle); }
    ~CuBlasHandleRAII() { if (handle) cublasDestroy(handle); }
    operator cublasHandle_t() const { return handle; }
};

namespace cuda {

inline void launch_kernel_XW(int rows, int cols, int out,
                             const float *d_X,
                             const float *d_W,
                             float       *d_Y)
{
    static CuBlasHandleRAII handle;

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    cublasSgemm(handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                out,    rows,  cols,
                &alpha,
                d_W,    out,
                d_X,    cols,
                &beta,
                d_Y,    out);
}

}
#endif