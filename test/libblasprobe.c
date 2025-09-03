// libblasprobe.c
#define _GNU_SOURCE
#include <dlfcn.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdatomic.h>
#include <cblas.h>   // 仅用于原型声明

static _Atomic int seen_once = 0;

static const char* log_dir() {
    const char* d = getenv("LOG_DIR");
    return (d && d[0]) ? d : "./predict";
}
static void ensure_dir() {
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "mkdir -p '%s' 2>/dev/null", log_dir());
    system(cmd);
}
static void append_log(const char* fmt, ...) {
    ensure_dir();
    char path[512];
    snprintf(path, sizeof(path), "%s/params.log", log_dir());
    int fd = open(path, O_CREAT|O_WRONLY|O_APPEND, 0644);
    if (fd < 0) return;
    char buf[512];
    va_list ap; va_start(ap, fmt);
    int n = vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    if (n > 0) write(fd, buf, (size_t)n);
    close(fd);
}
static void record_and_maybe_abort(const char* op, long M, long N, long K) {
    double flops = 2.0 * (double)M * (double)N * (double)K; // GEMM FLOPs
    append_log("%s M=%ld N=%ld K=%ld FLOPs=%.0f\n", op, M, N, K, flops);
    const char* pre = getenv("PREDICT_ONLY");
    if (pre && pre[0] && atomic_exchange(&seen_once, 1) == 0) {
        _exit(85); // 预检：拿到参数后立刻退出，避免真正计算
    }
}

/* ---- CBLAS dgemm ---- */
void cblas_dgemm(const enum CBLAS_ORDER Order,
  const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB,
  const int M, const int N, const int K,
  const double alpha, const double *A, const int lda,
  const double *B, const int ldb, const double beta, double *C, const int ldc)
{
    typedef void (*real_t)(const enum CBLAS_ORDER, const enum CBLAS_TRANSPOSE, const enum CBLAS_TRANSPOSE,
                           const int, const int, const int, const double,
                           const double*, const int, const double*, const int, const double, double*, const int);
    static real_t real = NULL;
    if (!real) real = (real_t)dlsym(RTLD_NEXT, "cblas_dgemm");
    record_and_maybe_abort("cblas_dgemm", M, N, K);
    real(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

/* ---- CBLAS sgemm ---- */
void cblas_sgemm(const enum CBLAS_ORDER Order,
  const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB,
  const int M, const int N, const int K,
  const float alpha, const float *A, const int lda,
  const float *B, const int ldb, const float beta, float *C, const int ldc)
{
    typedef void (*real_t)(const enum CBLAS_ORDER, const enum CBLAS_TRANSPOSE, const enum CBLAS_TRANSPOSE,
                           const int, const int, const int, const float,
                           const float*, const int, const float*, const int, const float, float*, const int);
    static real_t real = NULL;
    if (!real) real = (real_t)dlsym(RTLD_NEXT, "cblas_sgemm");
    record_and_maybe_abort("cblas_sgemm", M, N, K);
    real(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

/* ---- Fortran dgemm_（常见下划线约定） ---- */
void dgemm_(const char* TRANSA, const char* TRANSB,
            const int* M, const int* N, const int* K,
            const double* ALPHA, const double* A, const int* LDA,
            const double* B, const int* LDB, const double* BETA,
            double* C, const int* LDC)
{
    typedef void (*real_t)(const char*, const char*, const int*, const int*, const int*,
                           const double*, const double*, const int*,
                           const double*, const int*, const double*, double*, const int*);
    static real_t real = NULL;
    if (!real) real = (real_t)dlsym(RTLD_NEXT, "dgemm_");
    record_and_maybe_abort("dgemm_", *M, *N, *K);
    real(TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC);
}
