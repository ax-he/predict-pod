// libblasprobe_roofline_modes.c
#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdarg.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <stdio.h>
#include <cblas.h>

/* 选择统计模式（编译期）：
 *   make MODE=max   => -DPROBE_MODE_MAX
 *   make MODE=total => -DPROBE_MODE_TOTAL
 */
#if !defined(PROBE_MODE_MAX) && !defined(PROBE_MODE_TOTAL)
#  define PROBE_MODE_MAX 1
#endif

static _Atomic unsigned long long g_calls = 0;
static pthread_mutex_t g_mtx = PTHREAD_MUTEX_INITIALIZER;

#if defined(PROBE_MODE_TOTAL)
/* TOTAL 累积 */
static long double g_tot_flops = 0.0L;
static long double g_tot_bytes = 0.0L;
#else
/* MAX 单次最大 */
static unsigned long long g_max_flops = 0;
static long gM=0, gN=0, gK=0;
static int  g_dtype_bytes = 8;
#endif

/* 可靠写（处理短写/EINTR） */
static void write_all(int fd, const char* buf, size_t len){
    while (len > 0){
        ssize_t n = write(fd, buf, len);
        if (n > 0){ buf += (size_t)n; len -= (size_t)n; continue; }
        if (n < 0 && errno == EINTR) continue;
        break;
    }
}
static void wprint(const char* fmt, ...){
    char buf[640];
    va_list ap; va_start(ap, fmt);
    int n = vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    if (n > 0) write_all(2, buf, (size_t)n);
}

/* 解析真实符号 */
static void* resolve_sym(const char* name){
    void* s = dlsym(RTLD_NEXT, name);
    if (!s){
        void* h = dlopen("libopenblas.so.0", RTLD_LAZY|RTLD_LOCAL);
        if (h) s = dlsym(h, name);
    }
    return s;
}
#define RESOLVE_OR_DIE(real_t, var, symbol)                         \
    do{ if(!(var)){                                                 \
        void* s = resolve_sym(symbol);                              \
        if(!s){                                                     \
            const char* de = dlerror();                              \
            wprint("[probe] failed to resolve %s%s%s\n", symbol,     \
                   de?": ":"", de?de:"");                           \
            _exit(86);                                              \
        }                                                           \
        *(void**)(&var) = s;                                        \
    }}while(0)

/* Roofline 辅助 */
static inline unsigned long long flops_u64(long M,long N,long K){
    long double d = 2.0L*(long double)M*(long double)N*(long double)K;
    if (d < 0) d = 0;
    return (unsigned long long)(d + 0.5L);
}
static inline long double bytes_ld(long M,long N,long K,int bpe){
    /* 读A(MK) + 读B(KN) + 读写C(2MN) */
    return (long double)bpe * ( (long double)M*(long double)K +
                                (long double)K*(long double)N +
                          2.0L*(long double)M*(long double)N );
}
static double env_double(const char* key, double defv){
    const char* s = getenv(key);
    if (!s || !*s) return defv;
    char* end = NULL;
    double v = strtod(s, &end);
    if (end==s) return defv;
    return v;
}
/* 解析 budgets（逗号/分号/空格分隔） */
static int parse_budgets(const char* s, double* out, int cap){
    int n=0; if(!s||!*s) return 0;
    char buf[256]; strncpy(buf, s, sizeof(buf)-1); buf[sizeof(buf)-1]='\0';
    const char* delim = ",; ";
    for(char* tok=strtok(buf,delim); tok && n<cap; tok=strtok(NULL,delim)){
        char* end=NULL; double v=strtod(tok,&end);
        if(end!=tok) out[n++]=v;
    }
    return n;
}

/* 记录一次调用 */
static void account(long M,long N,long K,int dtype_bytes){
    unsigned long long f = flops_u64(M,N,K);
    long double       b = bytes_ld(M,N,K,dtype_bytes);
#if defined(PROBE_MODE_TOTAL)
    pthread_mutex_lock(&g_mtx);
    g_tot_flops += (long double)f;
    g_tot_bytes += b;
    pthread_mutex_unlock(&g_mtx);
#else
    if (f > g_max_flops){
        pthread_mutex_lock(&g_mtx);
        if (f > g_max_flops){
            g_max_flops = f; gM=M; gN=N; gK=K; g_dtype_bytes=dtype_bytes;
        }
        pthread_mutex_unlock(&g_mtx);
    }
#endif
    atomic_fetch_add(&g_calls, 1ULL);
}

/* 退出时打印与判定 */
static void summarize_and_decide(void){
#if defined(PROBE_MODE_TOTAL)
    if (g_tot_flops <= 0.0L) return;
    double FLOPs = (double)g_tot_flops;
    double BYTES = (double)g_tot_bytes;
    const char* mode = "TOTAL";
#else
    if (g_max_flops == 0ULL) return;
    double FLOPs = (double)g_max_flops;
    double BYTES = (double)bytes_ld(gM,gN,gK,g_dtype_bytes);
    const char* mode = "MAX";
#endif
    double peak_gflops = env_double("PEAK_GFLOPS", 50.906); // GF/s
    double mem_bw_gbs  = env_double("MEM_BW_GBS",  14.53);  // GB/s

    double t_compute = FLOPs / (peak_gflops * 1e9);
    double t_memory  = BYTES / (mem_bw_gbs * 1e9);
    double t_est     = (t_compute > t_memory ? t_compute : t_memory);
    double ai        = BYTES>0 ? (FLOPs/BYTES) : 0.0;      /* ← 避免与 <complex.h> 的 I 宏冲突 */

#if defined(PROBE_MODE_TOTAL)
    wprint("[probe] mode=%s calls=%llu  TOTAL: FLOPs=%.0f  Bytes=%.0f  AI=%.3f flop/byte\n",
           mode, (unsigned long long)g_calls, FLOPs, BYTES, ai);
#else
    wprint("[probe] mode=%s calls=%llu  MAX: M=%ld N=%ld K=%ld dtype=%s  FLOPs=%.0f  Bytes=%.0f  AI=%.3f\n",
           mode, (unsigned long long)g_calls, gM,gN,gK,
           (g_dtype_bytes==4?"float":"double"), FLOPs, BYTES, ai);
#endif
    wprint("[probe] peak: %.3f GFLOP/s, %.2f GB/s  t_compute=%.6fs  t_memory=%.6fs  => t_est=%.6fs\n",
           peak_gflops, mem_bw_gbs, t_compute, t_memory, t_est);

    /* 预算优先级：MODE 专属 > TIME_BUDGETS > TIME_BUDGET */
    const char* s_mode = getenv(
    #if defined(PROBE_MODE_TOTAL)
        "TIME_BUDGET_TOTAL"
    #else
        "TIME_BUDGET_MAX"
    #endif
    );
    const char* s_list = (s_mode && *s_mode) ? s_mode :
                         (getenv("TIME_BUDGETS") && *getenv("TIME_BUDGETS") ? getenv("TIME_BUDGETS")
                                                                            : getenv("TIME_BUDGET"));

    double budgets[8]; int nb = parse_budgets(s_list, budgets, 8);
    if (nb <= 0){
        wprint("[probe] no TIME_BUDGET provided; set TIME_BUDGET=secs or TIME_BUDGETS=\"0.2,0.5\".\n");
        return;
    }
    for (int i=0;i<nb;i++){
        const char* ans = (t_est <= budgets[i]) ? "yes" : "no";
        wprint("[probe] decision[%s=%.6fs]=%s\n",
               (s_mode&&*s_mode) ? (
#if defined(PROBE_MODE_TOTAL)
                 "TIME_BUDGET_TOTAL"
#else
                 "TIME_BUDGET_MAX"
#endif
               ) : (getenv("TIME_BUDGETS")&&*getenv("TIME_BUDGETS") ? "TIME_BUDGETS" : "TIME_BUDGET"),
               budgets[i], ans);
    }
    if (nb == 1){
        const char* bare = (t_est <= budgets[0]) ? "yes\n" : "no\n";
        write_all(2, bare, strlen(bare));
    }
}

__attribute__((constructor))
static void init_probe(void){ atexit(summarize_and_decide); }

/* CBLAS 包装 */
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
    account(M,N,K,8);
    real(Order,TA,TB,M,N,K,alpha,A,lda,B,ldb,beta,C,ldc);
}

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
    account(M,N,K,4);
    real(Order,TA,TB,M,N,K,alpha,A,lda,B,ldb,beta,C,ldc);
}
