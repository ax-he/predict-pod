// complex_gemm.c
// 复杂 GEMM 示例：混合转置/非方阵/不同 lda,ldb,ldc + 手写 batched
// 编译：见文末

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cblas.h>

static double* aligned_alloc_d(size_t n) {
    // 64-byte 对齐以利于 SIMD
    void *p = NULL;
    if (posix_memalign(&p, 64, n * sizeof(double)) != 0) return NULL;
    return (double*)p;
}

static void fill_rand(double *x, size_t n) {
    for (size_t i=0;i<n;i++) x[i] = (double)rand() / RAND_MAX - 0.5;
}

static void print_head(const char* name, const double* M, int r, int c, int ld, int k) {
    // 打印前 k×k 子块，便于快速 sanity check
    int rr = (r<k? r:k), cc=(c<k?c:k);
    printf("%s (top-left %dx%d):\n", name, rr, cc);
    for (int i=0;i<rr;i++) {
        for (int j=0;j<cc;j++) printf("%8.4f ", M[i*ld + j]);
        printf("\n");
    }
}

static void run_one_gemm(
    CBLAS_ORDER order,
    CBLAS_TRANSPOSE ta, CBLAS_TRANSPOSE tb,
    int M, int N, int K,
    double alpha,
    const double* A, int lda,
    const double* B, int ldb,
    double beta,
    double* C, int ldc,
    const char* tag)
{
    clock_t t0 = clock();
    cblas_dgemm(order, ta, tb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    clock_t t1 = clock();
    double secs = (double)(t1 - t0) / CLOCKS_PER_SEC;
    // 理论 FLOPs：2*M*N*K（FMA算 2 次浮点操作）
    double gflops = (2.0 * (double)M * (double)N * (double)K) / 1e9 / (secs > 1e-12 ? secs : 1.0);
    printf("[%-16s] M=%d N=%d K=%d  time=%.4fs  ~%.2f GFLOP/s\n", tag, M, N, K, secs, gflops);
}

// 手写“等步长 batched”：A、B、C 在批间按固定步长推进（最通用的兜底实现）
static void run_strided_batched_gemm(
    CBLAS_ORDER order,
    CBLAS_TRANSPOSE ta, CBLAS_TRANSPOSE tb,
    int M, int N, int K,
    double alpha,
    const double* A, int lda, long long strideA,
    const double* B, int ldb, long long strideB,
    double beta,
    double* C, int ldc, long long strideC,
    int batch_count)
{
    clock_t t0 = clock();
    for (int b=0;b<batch_count;b++) {
        const double* Ab = (const double*)((const char*)A + b*strideA);
        const double* Bb = (const double*)((const char*)B + b*strideB);
        double*       Cb = (double*      )((      char*)C + b*strideC);
        cblas_dgemm(order, ta, tb, M, N, K, alpha, Ab, lda, Bb, ldb, beta, Cb, ldc);
    }
    clock_t t1 = clock();
    double secs = (double)(t1 - t0) / CLOCKS_PER_SEC;
    double total_flops = batch_count * 2.0 * (double)M * (double)N * (double)K;
    double gflops = total_flops / 1e9 / (secs > 1e-12 ? secs : 1.0);
    printf("[strided-batch   ] batch=%d M=%d N=%d K=%d  time=%.4fs  ~%.2f GFLOP/s\n",
           batch_count, M, N, K, secs, gflops);
}

/* 若你使用 Intel oneMKL 且版本支持 CBLAS 扩展（cblas_?gemm_batch），
   可改用官方批量接口（需包含 mkl 的 cblas 头 & 链接行改为 mkl）：
   文档：cblas_?gemm_batch（oneMKL C 参考）:contentReference[oaicite:1]{index=1}
   示例（伪码）：
   const MKL_INT group_count = 1, group_size[1] = {batch_count};
   const CBLAS_TRANSPOSE transA[1] = {CblasNoTrans}, transB[1] = {CblasNoTrans};
   const MKL_INT m[1]={M}, n[1]={N}, k[1]={K}, lda_arr[1]={lda}, ldb_arr[1]={ldb}, ldc_arr[1]={ldc};
   const double alpha_arr[1]={alpha}, beta_arr[1]={beta};
   const double* A_array[batch_count]; const double* B_array[batch_count]; double* C_array[batch_count];
   // 填充 A_array/B_array/C_array 指针后：
   cblas_dgemm_batch(CblasRowMajor, transA, transB, m, n, k, alpha_arr,
                     (const double**)A_array, lda_arr, (const double**)B_array, ldb_arr,
                     beta_arr, C_array, ldc_arr, group_count, group_size);
*/

int main() {
    srand(123);

    // Case 1：RowMajor 非方阵 + 不同 lda/ldb/ldc
    int M1=1024, N1=768, K1=1536;
    int lda1=K1+16, ldb1=N1+8, ldc1=N1+32;  // 故意留边界，模拟实际步长
    double *A1 = aligned_alloc_d((size_t)M1*lda1);
    double *B1 = aligned_alloc_d((size_t)K1*ldb1);
    double *C1 = aligned_alloc_d((size_t)M1*ldc1);
    fill_rand(A1, (size_t)M1*lda1);
    fill_rand(B1, (size_t)K1*ldb1);
    memset(C1, 0, (size_t)M1*ldc1*sizeof(double));
    run_one_gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                 M1, N1, K1, 1.0, A1, lda1, B1, ldb1, 0.0, C1, ldc1, "row nn");

    // Case 2：ColMajor + A 转置、B 不转置
    int M2=512, N2=1200, K2=640;
    int lda2=M2+4, ldb2=K2+0, ldc2=M2+0;   // 刻意设不同 leading dims
    // 在 ColMajor 布局里，分配尺寸要按列主序的思维组织
    double *A2 = aligned_alloc_d((size_t)K2*lda2); // A^T 视角下仍按源矩阵尺寸分配
    double *B2 = aligned_alloc_d((size_t)K2*ldb2);
    double *C2 = aligned_alloc_d((size_t)M2*ldc2);
    fill_rand(A2, (size_t)K2*lda2);
    fill_rand(B2, (size_t)K2*ldb2);
    memset(C2, 0, (size_t)M2*ldc2*sizeof(double));
    run_one_gemm(CblasColMajor, CblasTrans, CblasNoTrans,
                 M2, N2, K2, 0.75, A2, lda2, B2, ldb2, 0.25, C2, ldc2, "col t n");

    // Case 3：手写 Strided Batched（同尺寸多次运算）
    int Mb=384, Nb=384, Kb=384, batch=16;
    int lda=Kb, ldb=Nb, ldc=Nb;
    size_t Asz=(size_t)Mb*lda, Bsz=(size_t)Kb*ldb, Csz=(size_t)Mb*ldc;
    double *Ab = aligned_alloc_d(Asz*batch);
    double *Bb = aligned_alloc_d(Bsz*batch);
    double *Cb = aligned_alloc_d(Csz*batch);
    fill_rand(Ab, Asz*batch);
    fill_rand(Bb, Bsz*batch);
    memset(Cb, 0, Csz*batch*sizeof(double));
    run_strided_batched_gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                             Mb, Nb, Kb, 1.0,
                             Ab, lda, (long long)Asz*sizeof(double),
                             Bb, ldb, (long long)Bsz*sizeof(double),
                             0.0,
                             Cb, ldc, (long long)Csz*sizeof(double),
                             batch);

    // 可选：打印小矩阵头部做校验
    // print_head("C1", C1, M1, N1, ldc1, 4);

    free(A1); free(B1); free(C1);
    free(A2); free(B2); free(C2);
    free(Ab); free(Bb); free(Cb);
    return 0;
}
