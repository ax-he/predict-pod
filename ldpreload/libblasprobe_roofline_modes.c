// libblasprobe_roofline_modes.c — MAX/TOTAL + DRY-RUN + 逐次/汇总输出 +（新增）LOG_DIR/max.log
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

/* ----- 默认启用 TOTAL；若外部已 -DPROBE_MODE_* 则以外部为准 ----- */
#if !defined(PROBE_MODE_MAX) && !defined(PROBE_MODE_TOTAL)
#  define PROBE_MODE_TOTAL 1
#endif

/* ----- 默认开启 DRY-RUN；若外部已 -DPROBE_DRY_RUN=0 或未定义，可覆盖 ----- */
#ifndef PROBE_DRY_RUN
#  define PROBE_DRY_RUN 1
#endif

static inline int is_dry_run(void){
#ifdef PROBE_DRY_RUN
    return 1;
#else
    const char* s = getenv("PROBE_DRY_RUN");
    return (s && *s == '1');
#endif
}

static _Atomic unsigned long long g_calls = 0;
static pthread_mutex_t g_mtx = PTHREAD_MUTEX_INITIALIZER;

#if defined(PROBE_MODE_TOTAL)
static long double g_tot_flops = 0.0L;
static long double g_tot_bytes = 0.0L;
#else
static unsigned long long g_max_flops = 0;
static long gM=0, gN=0, gK=0;
static int  g_dtype_bytes = 8; /* 8=double, 4=float */
#endif

/* ---- safe write ---- */
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

/* ---- symbol resolve ---- */
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
            const char* de = dlerror();                             \
            wprint("[probe] failed to resolve %s%s%s\n", symbol,    \
                   de?": ":"", de?de:"");                           \
            _exit(86);                                              \
        }                                                           \
        *(void**)(&var) = s;                                        \
    }}while(0)

/* ---- roofline helpers ---- */
static inline unsigned long long flops_u64(long M,long N,long K){
    long double d = 2.0L*(long double)M*(long double)N*(long double)K;
    if (d < 0) d = 0;
    return (unsigned long long)(d + 0.5L);
}
static inline long double bytes_ld(long M,long N,long K,int bpe){
    return (long double)bpe * ( (long double)M*(long double)K +
                                (long double)K*(long double)N +
                          2.0L*(long double)M*(long double)N );
}
static double env_double(const char* key, double defv){
    const char* s = getenv(key);
    if (!s || !*s) return defv;
    char* end = NULL; double v = strtod(s, &end);
    if (end==s) return defv; return v;
}
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

/* ---- per-call log (DRY-RUN only) ---- */
static void log_per_call(long M,long N,long K,int dtype_bytes,const char* who){
    if (!is_dry_run()) return;
    unsigned long long f = flops_u64(M,N,K);
    unsigned long long b = (unsigned long long)(bytes_ld(M,N,K,dtype_bytes) + 0.5L);
    wprint("[probe-call] %s M=%ld N=%ld K=%ld dtype=%s FLOPs=%llu Bytes=%llu\n",
           who, M,N,K, (dtype_bytes==4?"float":"double"), f, b);
}

/* ---- accounting ---- */
static void account(long M,long N,long K,int dtype_bytes){
    unsigned long long f = flops_u64(M,N,K);
#if defined(PROBE_MODE_TOTAL)
    long double       b = bytes_ld(M,N,K,dtype_bytes);
    pthread_mutex_lock(&g_mtx);
    g_tot_flops += (long double)f;
    g_tot_bytes += b;
    pthread_mutex_unlock(&g_mtx);
#else
    if (f > g_max_flops){
        pthread_mutex_lock(&g_mtx);
        if (f > g_max_flops){ g_max_flops = f; gM=M; gN=N; gK=K; g_dtype_bytes=dtype_bytes; }
        pthread_mutex_unlock(&g_mtx);
    }
#endif
    atomic_fetch_add(&g_calls, 1ULL);
}

/* ---- NEW: write [max] line into $LOG_DIR/max.log if set ---- */
static void write_maxlog_if_needed(double FLOPs){
#if !defined(PROBE_MODE_TOTAL)
    const char* dir = getenv("LOG_DIR");
    if (!dir || !*dir) return;
    char path[256]; snprintf(path, sizeof(path), "%s/%s", dir, "max.log");
    FILE* fp = fopen(path, "w");
    if (!fp) return;
    /* listen.py 的正则期望：[max] <word> M= N= K= FLOPs= */
    fprintf(fp, "[max] GEMM M=%ld N=%ld K=%ld FLOPs=%.0f\n", gM, gN, gK, FLOPs);
    fclose(fp);
#endif
}

/* ---- summarize & decide ---- */
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
    double peak_gflops = env_double("PEAK_GFLOPS", 50.906);
    double mem_bw_gbs  = env_double("MEM_BW_GBS",  14.53);

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
    double ai = BYTES>0 ? (FLOPs/BYTES) : 0.0;
    double t_compute = FLOPs / (peak_gflops * 1e9);
    double t_memory  = BYTES / (mem_bw_gbs * 1e9);
    double t_est     = (t_compute > t_memory ? t_compute : t_memory);

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

    if (nb <= 0){
        wprint("[probe] no TIME_BUDGET provided; set TIME_BUDGET=secs or TIME_BUDGETS=\"0.2,0.5\".\n");
#if !defined(PROBE_MODE_TOTAL)
        write_maxlog_if_needed(FLOPs);  /* 即便没预算，也写一行 max.log 供解析 */
#endif
        return;
    }
    for (int i=0;i<nb;i++){
        double Tb = budgets[i];
        double req_gflops = FLOPs / (Tb * 1e9);
        double req_gbps   = BYTES / (Tb * 1e9);
        const char* ok = (req_gflops <= peak_gflops && req_gbps <= mem_bw_gbs) ? "yes" : "no";
        wprint("[probe] require: %.3f GF/s & %.3f GB/s within %.6fs  => %s\n", req_gflops, req_gbps, Tb, ok);
    }
#if !defined(PROBE_MODE_TOTAL)
    write_maxlog_if_needed(FLOPs);
#endif
}

__attribute__((constructor))
static void init_probe(void){ atexit(summarize_and_decide); }

/* ---- wrappers ---- */
void cblas_dgemm(const enum CBLAS_ORDER Order,
  const enum CBLAS_TRANSPOSE TA, const enum CBLAS_TRANSPOSE TB,
  const int M, const int N, const int K,
  const double alpha, const double *A, const int lda,
  const double *B, const int ldb, const double beta, double *C, const int ldc)
{
    account(M,N,K,8);
    log_per_call(M,N,K,8,"cblas_dgemm");
    if (!is_dry_run()){
        typedef void (*real_t)(const enum CBLAS_ORDER,const enum CBLAS_TRANSPOSE,const enum CBLAS_TRANSPOSE,
                               const int,const int,const int,const double,
                               const double*,const int,const double*,const int,const double,double*,const int);
        static real_t real = NULL;
        RESOLVE_OR_DIE(real_t, real, "cblas_dgemm");
        real(Order,TA,TB,M,N,K,alpha,A,lda,B,ldb,beta,C,ldc);
    }
}

void cblas_sgemm(const enum CBLAS_ORDER Order,
  const enum CBLAS_TRANSPOSE TA, const enum CBLAS_TRANSPOSE TB,
  const int M, const int N, const int K,
  const float alpha, const float *A, const int lda,
  const float *B, const int ldb, const float beta, float *C, const int ldc)
{
    account(M,N,K,4);
    log_per_call(M,N,K,4,"cblas_sgemm");
    if (!is_dry_run()){
        typedef void (*real_t)(const enum CBLAS_ORDER,const enum CBLAS_TRANSPOSE,const enum CBLAS_TRANSPOSE,
                               const int,const int,const int,const float,
                               const float*,const int,const float*,const int,const float,float*,const int);
        static real_t real = NULL;
        RESOLVE_OR_DIE(real_t, real, "cblas_sgemm");
        real(Order,TA,TB,M,N,K,alpha,A,lda,B,ldb,beta,C,ldc);
    }
}
