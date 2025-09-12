#define _GNU_SOURCE
#include <dlfcn.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include <errno.h>
#include <cblas.h>

static _Atomic unsigned long long calls = 0;
static unsigned long long g_max = 0;
static long gM=0,gN=0,gK=0;
static pthread_mutex_t mtx = PTHREAD_MUTEX_INITIALIZER;
static int logfd = -1;

/* 写满为止，处理 EINTR/短写；若写失败直接返回 */
static void write_all(int fd, const char* buf, size_t len) {
    while (len > 0) {
        ssize_t n = write(fd, buf, len);
        if (n > 0) { buf += (size_t)n; len -= (size_t)n; continue; }
        if (n < 0 && errno == EINTR) continue;
        break; // 其它错误/不可进展：放弃
    }
}

static int open_logfd(void){
    if (logfd != -1) return logfd;
    const char* d = getenv("LOG_DIR"); if (!d || !*d) d = ".";
    char p[512];
    int n = snprintf(p, sizeof(p), "%s/max.log", d);
    if (n > 0) {
        /* O_APPEND 让并发写入具有追加原子性 */
        int fd = open(p, O_CREAT|O_WRONLY|O_APPEND, 0644);
        if (fd >= 0) logfd = fd;
    }
    return logfd;
}
static void write_err(const char* msg){
    write_all(2, msg, strlen(msg));
}

static void maybe_record(long M,long N,long K,const char* who){
    unsigned long long f = (unsigned long long)(2.0*(double)M*(double)N*(double)K + 0.5);
    if (f > g_max){
        pthread_mutex_lock(&mtx);
        if (f > g_max){
            g_max=f; gM=M; gN=N; gK=K;
            char buf[256];
            int n = snprintf(buf, sizeof(buf),
                "[max] %s M=%ld N=%ld K=%ld FLOPs=%llu\n", who, M,N,K,f);
            int fd = open_logfd(); if (fd>=0 && n>0) write_all(fd, buf, (size_t)n);
        }
        pthread_mutex_unlock(&mtx);
    }
    atomic_fetch_add(&calls, 1ULL);
}

/* 解析真实符号：先 RTLD_NEXT，再尝试直接从 openblas 句柄找；失败则报错并退出 */
static void *resolve_sym(const char* name){
    void *sym = dlsym(RTLD_NEXT, name);
    if (!sym) {
        void *h = dlopen("libopenblas.so.0", RTLD_LAZY|RTLD_LOCAL);
        if (h) sym = dlsym(h, name);
    }
    return sym;
}
#define RESOLVE_OR_DIE(real_t, var, primary)                        \
    do {                                                            \
        if (!(var)) {                                               \
            void *s = resolve_sym(primary);                         \
            if (!s) {                                               \
                char ebuf[256];                                     \
                int n = snprintf(ebuf,sizeof(ebuf),                 \
                  "[probe] failed to resolve %s\n", primary);       \
                if (n>0) write_err(ebuf);                           \
                const char* de = dlerror();                         \
                if (de){ write_err("[probe] dlerror: "); write_err(de); write_err("\n"); } \
                _exit(86);                                          \
            }                                                       \
            *(void**)(&var) = s;                                    \
        }                                                           \
    } while(0)

/* ---- dgemm (double) ---- */
void cblas_dgemm(const enum CBLAS_ORDER Order,
  const enum CBLAS_TRANSPOSE TA, const enum CBLAS_TRANSPOSE TB,
  const int M, const int N, const int K,
  const double alpha, const double *A, const int lda,
  const double *B, const int ldb, const double beta, double *C, const int ldc)
{
    typedef void (*real_t)(const enum CBLAS_ORDER,const enum CBLAS_TRANSPOSE,const enum CBLAS_TRANSPOSE,
                           const int,const int,const int,const double,
                           const double*,const int,const double*,const int,const double,double*,const int);
    static real_t real = NULL;
    RESOLVE_OR_DIE(real_t, real, "cblas_dgemm");
    maybe_record(M,N,K,"cblas_dgemm");
    real(Order,TA,TB,M,N,K,alpha,A,lda,B,ldb,beta,C,ldc);
}

/* ---- sgemm (float) ---- */
void cblas_sgemm(const enum CBLAS_ORDER Order,
  const enum CBLAS_TRANSPOSE TA, const enum CBLAS_TRANSPOSE TB,
  const int M, const int N, const int K,
  const float alpha, const float *A, const int lda,
  const float *B, const int ldb, const float beta, float *C, const int ldc)
{
    typedef void (*real_t)(const enum CBLAS_ORDER,const enum CBLAS_TRANSPOSE,const enum CBLAS_TRANSPOSE,
                           const int,const int,const int,const float,
                           const float*,const int,const float*,const int,const float,float*,const int);
    static real_t real = NULL;
    RESOLVE_OR_DIE(real_t, real, "cblas_sgemm");
    maybe_record(M,N,K,"cblas_sgemm");
    real(Order,TA,TB,M,N,K,alpha,A,lda,B,ldb,beta,C,ldc);
}
