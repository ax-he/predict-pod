// complex_gemm.c  —  安全版示例：Row/Col-Major + 转置组合 + Strided Batched
// 关键点：
// 1) 缓冲区长度严格按文档计算：
//    - ColMajor:  len(A) = lda * (TransA ? M : K); len(B) = ldb * (TransB ? K : N); len(C) = ldc * N
//    - RowMajor:  len(A) = lda * (TransA ? K : M); len(B) = ldb * (TransB ? N : K); len(C) = ldc * M
// 2) 下界（assert）严格遵循 BLAS/oneMKL 规则：
//    - ColMajor:  lda >= (TransA ? K : M); ldb >= (TransB ? N : K); ldc >= M
//    - RowMajor:  lda >= (TransA ? M : K); ldb >= (TransB ? K : N); ldc >= N
// 3) 使用 posix_memalign(64B)；计时用 CLOCK_MONOTONIC；FLOPs = 2*M*N*K
//
// 参考（LDA/LDB/LDC/存储规则与 DGEMM 参数说明）：
// • Netlib DGEMM：A 维度 (LDA,ka)，ka=K(TRANSA='N') 否则 M；C(LDC,N)（列主序）。 :contentReference[oaicite:0]{index=0}
// • oneMKL Matrix Storage：通用矩阵一维数组最少大小（列主序 lda*n，行主序 lda*m）。 :contentReference[oaicite:1]{index=1}
// • CBLAS cblas_dgemm 形参与 Row/Col-Major 语义。 :contentReference[oaicite:2]{index=2}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <assert.h>
#include <cblas.h>

static double* aligned_alloc_d(size_t n_elems) {
    void *p = NULL;
    if (posix_memalign(&p, 64, n_elems * sizeof(double)) != 0) return NULL;
    return (double*)p;
}

static void fill_rand(double *x, size_t n) {
    for (size_t i=0;i<n;i++) x[i] = (double)rand() / RAND_MAX - 0.5;
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

// ---- 元素个数（缓冲区长度）与下界计算 ----
static size_t elems_A(CBLAS_ORDER order, CBLAS_TRANSPOSE TA, int M, int K, int lda){
    return (size_t)lda * (size_t)((order==CblasColMajor) ? (TA==CblasNoTrans? K: M)
                                                          : (TA==CblasNoTrans? M: K));
}
static size_t elems_B(CBLAS_ORDER order, CBLAS_TRANSPOSE TB, int N, int K, int ldb){
    return (size_t)ldb * (size_t)((order==CblasColMajor) ? (TB==CblasNoTrans? N: K)
                                                          : (TB==CblasNoTrans? K: N));
}
static size_t elems_C(CBLAS_ORDER order, int M, int N, int ldc){
    return (size_t)ldc * (size_t)((order==CblasColMajor) ? N : M);
}
static int need_lda(CBLAS_ORDER order, CBLAS_TRANSPOSE TA, int M, int K){
    return (order==CblasColMajor) ? (TA==CblasNoTrans? M: K) : (TA==CblasNoTrans? K: M);
}
static int need_ldb(CBLAS_ORDER order, CBLAS_TRANSPOSE TB, int N, int K){
    return (order==CblasColMajor) ? (TB==CblasNoTrans? K: N) : (TB==CblasNoTrans? N: K);
}
static int need_ldc(CBLAS_ORDER order, int M, int N){
    return (order==CblasColMajor) ? M : N;
}
static void assert_ld_ok(CBLAS_ORDER order, CBLAS_TRANSPOSE TA, CBLAS_TRANSPOSE TB,
                         int M,int N,int K,int lda,int ldb,int ldc){
    assert(lda >= need_lda(order,TA,M,K));
    assert(ldb >= need_ldb(order,TB,N,K));
    assert(ldc >= need_ldc(order,M,N));
}

// ---- 跑一次 GEMM 并计时 ----
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
    double t0 = now_sec();
    cblas_dgemm(order, ta, tb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    double t1 = now_sec();
    double secs = t1 - t0;
    double gflops = (2.0 * (double)M * (double)N * (double)K) / 1e9 / (secs > 1e-12 ? secs : 1.0);
    printf("[%-16s] M=%d N=%d K=%d  time=%.4fs  ~%.2f GFLOP/s\n", tag, M, N, K, secs, gflops);
}

// ---- Strided Batched（等步长）----
static void run_strided_batched_gemm(
    CBLAS_ORDER order,
    CBLAS_TRANSPOSE ta, CBLAS_TRANSPOSE tb,
    int M, int N, int K,
    double alpha,
    const double* A, int lda, long long strideA_bytes,
    const double* B, int ldb, long long strideB_bytes,
    double beta,
    double* C, int ldc, long long strideC_bytes,
    int batch_count)
{
    double t0 = now_sec();
    for (int b=0;b<batch_count;b++) {
        const double* Ab = (const double*)((const char*)A + b*strideA_bytes);
        const double* Bb = (const double*)((const char*)B + b*strideB_bytes);
        double*       Cb = (double*      )((      char*)C + b*strideC_bytes);
        cblas_dgemm(order, ta, tb, M, N, K, alpha, Ab, lda, Bb, ldb, beta, Cb, ldc);
    }
    double t1 = now_sec();
    double secs = t1 - t0;
    double total_flops = batch_count * 2.0 * (double)M * (double)N * (double)K;
    double gflops = total_flops / 1e9 / (secs > 1e-12 ? secs : 1.0);
    printf("[strided-batch   ] batch=%d M=%d N=%d K=%d  time=%.4fs  ~%.2f GFLOP/s\n",
           batch_count, M, N, K, secs, gflops);
}

int main() {
    srand(123);

    // ---- Case 1：RowMajor, A/B 都不转置（非方阵 + 预留边界）----
    int M1=1024, N1=768, K1=1536;
    int lda1=K1+16, ldb1=N1+8, ldc1=N1+32; // RowMajor: need lda>=K1, ldb>=N1, ldc>=N1
    assert_ld_ok(CblasRowMajor, CblasNoTrans, CblasNoTrans, M1,N1,K1, lda1,ldb1,ldc1);

    size_t A1_len = elems_A(CblasRowMajor, CblasNoTrans, M1,K1, lda1);
    size_t B1_len = elems_B(CblasRowMajor, CblasNoTrans, N1,K1, ldb1);
    size_t C1_len = elems_C(CblasRowMajor, M1,N1, ldc1);

    double *A1 = aligned_alloc_d(A1_len);
    double *B1 = aligned_alloc_d(B1_len);
    double *C1 = aligned_alloc_d(C1_len);
    fill_rand(A1, A1_len);
    fill_rand(B1, B1_len);
    memset(C1, 0, C1_len*sizeof(double));

    run_one_gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                 M1, N1, K1, 1.0, A1, lda1, B1, ldb1, 0.0, C1, ldc1, "row nn");

    free(A1); free(B1); free(C1);

    // ---- Case 2：ColMajor, A=Trans, B=NoTrans（非方阵）----
    int M2=512, N2=1200, K2=640;
    int lda2=K2, ldb2=K2, ldc2=M2; // ColMajor：A(T)→lda>=K2；B(N)→ldb>=K2；C→ldc>=M2
    assert_ld_ok(CblasColMajor, CblasTrans, CblasNoTrans, M2,N2,K2, lda2,ldb2,ldc2);

    size_t A2_len = elems_A(CblasColMajor, CblasTrans,    M2,K2, lda2);
    size_t B2_len = elems_B(CblasColMajor, CblasNoTrans,  N2,K2, ldb2);
    size_t C2_len = elems_C(CblasColMajor, M2,N2, ldc2);

    double *A2 = aligned_alloc_d(A2_len);
    double *B2 = aligned_alloc_d(B2_len);
    double *C2 = aligned_alloc_d(C2_len);
    fill_rand(A2, A2_len);
    fill_rand(B2, B2_len);
    memset(C2, 0, C2_len*sizeof(double));

    run_one_gemm(CblasColMajor, CblasTrans, CblasNoTrans,
                 M2, N2, K2, 0.75, A2, lda2, B2, ldb2, 0.25, C2, ldc2, "col t n");

    free(A2); free(B2); free(C2);

    // ---- Case 3：RowMajor，手写 Strided Batched（同尺寸多次运算）----
    int Mb=384, Nb=384, Kb=384, batch=16;
    int lda=Kb, ldb=Nb, ldc=Nb; // RowMajor：A(N)→lda>=Kb；B(N)→ldb>=Nb；C→ldc>=Nb
    assert_ld_ok(CblasRowMajor, CblasNoTrans, CblasNoTrans, Mb,Nb,Kb, lda,ldb,ldc);

    size_t Ab_elems = elems_A(CblasRowMajor, CblasNoTrans, Mb,Kb, lda);
    size_t Bb_elems = elems_B(CblasRowMajor, CblasNoTrans, Nb,Kb, ldb);
    size_t Cb_elems = elems_C(CblasRowMajor, Mb,Nb, ldc);

    size_t Ab_bytes = Ab_elems * sizeof(double);
    size_t Bb_bytes = Bb_elems * sizeof(double);
    size_t Cb_bytes = Cb_elems * sizeof(double);

    double *Ab = aligned_alloc_d(Ab_elems * (size_t)batch);
    double *Bb = aligned_alloc_d(Bb_elems * (size_t)batch);
    double *Cb = aligned_alloc_d(Cb_elems * (size_t)batch);

    fill_rand(Ab, Ab_elems * (size_t)batch);
    fill_rand(Bb, Bb_elems * (size_t)batch);
    memset(Cb, 0, Cb_bytes * (size_t)batch);

    run_strided_batched_gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                             Mb, Nb, Kb, 1.0,
                             Ab, lda, (long long)Ab_bytes,
                             Bb, ldb, (long long)Bb_bytes,
                             0.0,
                             Cb, ldc, (long long)Cb_bytes,
                             batch);

    free(Ab); free(Bb); free(Cb);

    return 0;
}
