// libfftprobe.c — FFT 版 MAX/TOTAL + DRY-RUN + 多预算评估（FFTW3 / FFTW3F）
// 用法（示例）：
//   gcc -O2 -fPIC -shared -o libfftprobe.so libfftprobe.c -ldl -lpthread
//   LD_PRELOAD=./libfftprobe.so IO_FACTOR=2.0 PEAK_GFLOPS=200 MEM_BW_GBS=50 TIME_BUDGETS="0.02,0.05" ./fft_test
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
#include <math.h>
#include <fftw3.h>   // 同时声明双精度与单精度 API

/* ----- 总/最大 两种聚合模式：默认 TOTAL，可用 -DPROBE_MODE_MAX 覆盖 ----- */
#if !defined(PROBE_MODE_MAX) && !defined(PROBE_MODE_TOTAL)
#  define PROBE_MODE_TOTAL 1
#endif

/* ----- 默认 DRY-RUN 打开；可用 -DPROBE_DRY_RUN=0 或运行时环境覆盖 ----- */
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

/* ---------- 安全写日志 ---------- */
static void write_all(int fd, const char* buf, size_t len){
    while (len > 0){
        ssize_t n = write(fd, buf, len);
        if (n > 0){ buf += (size_t)n; len -= (size_t)n; continue; }
        if (n < 0 && errno == EINTR) continue;
        break;
    }
}
static void wprint(const char* fmt, ...){
    char buf[768];
    va_list ap; va_start(ap, fmt);
    int n = vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    if (n > 0) write_all(2, buf, (size_t)n);
}

/* ---------- 动态符号解析 ---------- */
static void* resolve_sym(const char* name){
    void* s = dlsym(RTLD_NEXT, name);
    if (!s){
        // 常见库名：libfftw3.so / libfftw3f.so
        void* h1 = dlopen("libfftw3.so",  RTLD_LAZY|RTLD_LOCAL);
        void* h2 = dlopen("libfftw3f.so", RTLD_LAZY|RTLD_LOCAL);
        if (h1) s = dlsym(h1, name);
        if (!s && h2) s = dlsym(h2, name);
    }
    return s;
}
#define RESOLVE_OR_DIE(real_t, var, symbol)                         \
    do{ if(!(var)){                                                 \
        void* s = resolve_sym(symbol);                              \
        if(!s){                                                     \
            const char* de = dlerror();                             \
            wprint("[fftprobe] failed to resolve %s%s%s\n", symbol, \
                   de?": ":"", de?de:"");                           \
            _exit(87);                                              \
        }                                                           \
        *(void**)(&var) = s;                                        \
    }}while(0)

/* ---------- 工具函数 ---------- */
static double env_double(const char* key, double defv){
    const char* s = getenv(key);
    if (!s || !*s) return defv;
    char* end=NULL; double v=strtod(s,&end);
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

/* ---------- plan 元数据表（线程安全） ---------- */
typedef struct {
    void*  plan_ptr;
    long   nx, ny, nz;  // 若为 1D/2D 则未用维度置 1
    int    dtype_bytes; // 复数元素字节数: float complex=8, double complex=16
} plan_meta_t;

#define PLAN_CAP 1024
static plan_meta_t g_plans[PLAN_CAP];
static pthread_mutex_t g_plans_mtx = PTHREAD_MUTEX_INITIALIZER;

static void plans_put(void* p, long nx,long ny,long nz,int bpe){
    pthread_mutex_lock(&g_plans_mtx);
    for (int i=0;i<PLAN_CAP;i++){
        if (g_plans[i].plan_ptr==NULL){
            g_plans[i].plan_ptr=p; g_plans[i].nx=nx; g_plans[i].ny=ny; g_plans[i].nz=nz; g_plans[i].dtype_bytes=bpe;
            pthread_mutex_unlock(&g_plans_mtx);
            return;
        }
    }
    pthread_mutex_unlock(&g_plans_mtx);
    wprint("[fftprobe] WARNING: plan table full, metadata dropped.\n");
}
static int plans_get(void* p, long* nx,long* ny,long* nz,int* bpe){
    int ok=0;
    pthread_mutex_lock(&g_plans_mtx);
    for (int i=0;i<PLAN_CAP;i++){
        if (g_plans[i].plan_ptr==p){
            *nx=g_plans[i].nx; *ny=g_plans[i].ny; *nz=g_plans[i].nz; *bpe=g_plans[i].dtype_bytes;
            ok=1; break;
        }
    }
    pthread_mutex_unlock(&g_plans_mtx);
    return ok;
}
static void plans_del(void* p){
    pthread_mutex_lock(&g_plans_mtx);
    for (int i=0;i<PLAN_CAP;i++){
        if (g_plans[i].plan_ptr==p){ g_plans[i].plan_ptr=NULL; break; }
    }
    pthread_mutex_unlock(&g_plans_mtx);
}

/* ---------- 复杂度与流量估算 ---------- */
/* 复数-复数 FFT 常见 FLOPs 估算： 5 * N * (log2 n1 + log2 n2 + ...) */
static inline double flog2(double x){ return log(x)/log(2.0); }

static inline unsigned long long fft_flops(long nx,long ny,long nz){
    long double N = (long double)nx * (long double)ny * (long double)nz;
    long double sum_log = 0.0L;
    if (nx>1) sum_log += flog2((double)nx);
    if (ny>1) sum_log += flog2((double)ny);
    if (nz>1) sum_log += flog2((double)nz);
    long double f = 5.0L * N * sum_log;
    if (f < 0) f = 0;
    return (unsigned long long)(f + 0.5L);
}

/* IO 估算：默认 2.0 * N * bpe（读+写），可用 IO_FACTOR 覆盖 */
static inline long double fft_bytes(long nx,long ny,long nz,int bpe){
    long double N = (long double)nx * (long double)ny * (long double)nz;
    double io_factor = env_double("IO_FACTOR", 2.0); // 可按需要调大（如临时缓冲/非就地等）
    return (long double)bpe * (long double)io_factor * N;
}

/* ---------- 汇总状态 ---------- */
static _Atomic unsigned long long g_calls = 0;
static pthread_mutex_t g_acc_mtx = PTHREAD_MUTEX_INITIALIZER;

#if defined(PROBE_MODE_TOTAL)
static long double g_tot_flops = 0.0L;
static long double g_tot_bytes = 0.0L;
#else
static unsigned long long g_max_flops = 0ULL;
static long gM=1,gN=1,gK=1;   // 这里借 M,N,K 表示 nx,ny,nz
static int  g_dtype_bytes = 16;
#endif

static void log_per_call(long nx,long ny,long nz,int bpe,const char* who){
    if (!is_dry_run()) return;
    unsigned long long f = fft_flops(nx,ny,nz);
    unsigned long long b = (unsigned long long)(fft_bytes(nx,ny,nz,bpe)+0.5L);
    wprint("[fftprobe-call] %s N=(%ld,%ld,%ld) dtype=%s FLOPs=%llu Bytes=%llu\n",
           who, nx,ny,nz, (bpe==8?"cfloat":"cdouble"), f, b);
}

static void account(long nx,long ny,long nz,int bpe){
    unsigned long long f = fft_flops(nx,ny,nz);
#if defined(PROBE_MODE_TOTAL)
    long double       b = fft_bytes(nx,ny,nz,bpe);
    pthread_mutex_lock(&g_acc_mtx);
    g_tot_flops += (long double)f;
    g_tot_bytes += b;
    pthread_mutex_unlock(&g_acc_mtx);
#else
    if (f > g_max_flops){
        pthread_mutex_lock(&g_acc_mtx);
        if (f > g_max_flops){ g_max_flops = f; gM=nx; gN=ny; gK=nz; g_dtype_bytes=bpe; }
        pthread_mutex_unlock(&g_acc_mtx);
    }
#endif
    atomic_fetch_add(&g_calls, 1ULL);
}

/* ---------- 汇总输出与预算评估 ---------- */
static void summarize_and_decide(void){
#if defined(PROBE_MODE_TOTAL)
    if (g_tot_flops <= 0.0L) return;
    double FLOPs = (double)g_tot_flops;
    double BYTES = (double)g_tot_bytes;
    const char* mode = "TOTAL";
#else
    if (g_max_flops == 0ULL) return;
    double FLOPs = (double)g_max_flops;
    double BYTES = (double)fft_bytes(gM,gN,gK,g_dtype_bytes);
    const char* mode = "MAX";
#endif
    double peak_gflops = env_double("PEAK_GFLOPS", 250.91);
    double mem_bw_gbs  = env_double("MEM_BW_GBS",   14.53);

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

#if defined(PROBE_MODE_TOTAL)
    wprint("[fftprobe] mode=%s calls=%llu  TOTAL: FLOPs=%.0f  Bytes=%.0f\n",
           mode, (unsigned long long)g_calls, FLOPs, BYTES);
#else
    wprint("[fftprobe] mode=%s calls=%llu  MAX: N=(%ld,%ld,%ld) dtype=%s  FLOPs=%.0f  Bytes=%.0f\n",
           mode, (unsigned long long)g_calls, gM,gN,gK,
           (g_dtype_bytes==8?"cfloat":"cdouble"), FLOPs, BYTES);
#endif

    double ai = (BYTES>0)?(FLOPs/BYTES):0.0;
    double t_compute = FLOPs / (peak_gflops * 1e9);
    double t_memory  = BYTES / (mem_bw_gbs  * 1e9);
    double t_est     = (t_compute > t_memory ? t_compute : t_memory);
    wprint("[fftprobe] peak: %.3f GF/s, %.2f GB/s  t_compute=%.6fs  t_memory=%.6fs  => t_est=%.6fs  (AI=%.3f)\n",
           peak_gflops, mem_bw_gbs, t_compute, t_memory, t_est, ai);

    if (nb <= 0){
        wprint("[fftprobe] no TIME_BUDGET provided; set TIME_BUDGET=secs or TIME_BUDGETS=\"0.02,0.05\".\n");
        return;
    }
    for (int i=0;i<nb;i++){
        double Tb = budgets[i];
        double req_gflops = FLOPs / (Tb * 1e9);
        double req_gbps   = BYTES / (Tb * 1e9);
        const char* ok = (req_gflops <= peak_gflops && req_gbps <= mem_bw_gbs) ? "yes" : "no";
        wprint("[fftprobe] require: %.3f GF/s & %.3f GB/s within %.6fs  => %s\n", req_gflops, req_gbps, Tb, ok);
    }
}

__attribute__((constructor))
static void init_fftprobe(void){ atexit(summarize_and_decide); }

/* ---------- 包装器：拦截 plan 创建 / 执行 / 销毁 ---------- */
/* 单精度（fftwf_） */
typedef fftwf_plan (*pf_fftwf_plan_dft_1d)(int,fftwf_complex*,fftwf_complex*,int,unsigned);
typedef fftwf_plan (*pf_fftwf_plan_dft_2d)(int,int,fftwf_complex*,fftwf_complex*,int,unsigned);
typedef fftwf_plan (*pf_fftwf_plan_dft_3d)(int,int,int,fftwf_complex*,fftwf_complex*,int,unsigned);
typedef void       (*pf_fftwf_execute)(const fftwf_plan);
typedef void       (*pf_fftwf_destroy_plan)(fftwf_plan);

/* 双精度（fftw_） */
typedef fftw_plan  (*pf_fftw_plan_dft_1d)(int,fftw_complex*,fftw_complex*,int,unsigned);
typedef fftw_plan  (*pf_fftw_plan_dft_2d)(int,int,fftw_complex*,fftw_complex*,int,unsigned);
typedef fftw_plan  (*pf_fftw_plan_dft_3d)(int,int,int,fftw_complex*,fftw_complex*,int,unsigned);
typedef void       (*pf_fftw_execute)(const fftw_plan);
typedef void       (*pf_fftw_destroy_plan)(fftw_plan);

/* ---- 单精度 plan ---- */
fftwf_plan fftwf_plan_dft_1d(int n, fftwf_complex* in, fftwf_complex* out, int sign, unsigned flags){
    static pf_fftwf_plan_dft_1d real=NULL; RESOLVE_OR_DIE(pf_fftwf_plan_dft_1d, real, "fftwf_plan_dft_1d");
    fftwf_plan p = real(n,in,out,sign,flags);
    if (p) plans_put((void*)p, n,1,1, /*cfloat*/ 8);
    return p;
}
fftwf_plan fftwf_plan_dft_2d(int nx,int ny, fftwf_complex* in, fftwf_complex* out, int sign, unsigned flags){
    static pf_fftwf_plan_dft_2d real=NULL; RESOLVE_OR_DIE(pf_fftwf_plan_dft_2d, real, "fftwf_plan_dft_2d");
    fftwf_plan p = real(nx,ny,in,out,sign,flags);
    if (p) plans_put((void*)p, nx,ny,1, 8);
    return p;
}
fftwf_plan fftwf_plan_dft_3d(int nx,int ny,int nz, fftwf_complex* in, fftwf_complex* out, int sign, unsigned flags){
    static pf_fftwf_plan_dft_3d real=NULL; RESOLVE_OR_DIE(pf_fftwf_plan_dft_3d, real, "fftwf_plan_dft_3d");
    fftwf_plan p = real(nx,ny,nz,in,out,sign,flags);
    if (p) plans_put((void*)p, nx,ny,nz, 8);
    return p;
}
void fftwf_execute(const fftwf_plan p){
    long nx=1,ny=1,nz=1; int bpe=8;
    int ok = plans_get((void*)p,&nx,&ny,&nz,&bpe);
    if (ok){ account(nx,ny,nz,bpe); log_per_call(nx,ny,nz,bpe,"fftwf_execute"); }
    if (!is_dry_run()){
        static pf_fftwf_execute real=NULL; RESOLVE_OR_DIE(pf_fftwf_execute, real, "fftwf_execute");
        real(p);
    }
}
void fftwf_destroy_plan(fftwf_plan p){
    plans_del((void*)p);
    static pf_fftwf_destroy_plan real=NULL; RESOLVE_OR_DIE(pf_fftwf_destroy_plan, real, "fftwf_destroy_plan");
    real(p);
}

/* ---- 双精度 plan ---- */
fftw_plan fftw_plan_dft_1d(int n, fftw_complex* in, fftw_complex* out, int sign, unsigned flags){
    static pf_fftw_plan_dft_1d real=NULL; RESOLVE_OR_DIE(pf_fftw_plan_dft_1d, real, "fftw_plan_dft_1d");
    fftw_plan p = real(n,in,out,sign,flags);
    if (p) plans_put((void*)p, n,1,1, /*cdouble*/ 16);
    return p;
}
fftw_plan fftw_plan_dft_2d(int nx,int ny, fftw_complex* in, fftw_complex* out, int sign, unsigned flags){
    static pf_fftw_plan_dft_2d real=NULL; RESOLVE_OR_DIE(pf_fftw_plan_dft_2d, real, "fftw_plan_dft_2d");
    fftw_plan p = real(nx,ny,in,out,sign,flags);
    if (p) plans_put((void*)p, nx,ny,1, 16);
    return p;
}
fftw_plan fftw_plan_dft_3d(int nx,int ny,int nz, fftw_complex* in, fftw_complex* out, int sign, unsigned flags){
    static pf_fftw_plan_dft_3d real=NULL; RESOLVE_OR_DIE(pf_fftw_plan_dft_3d, real, "fftw_plan_dft_3d");
    fftw_plan p = real(nx,ny,nz,in,out,sign,flags);
    if (p) plans_put((void*)p, nx,ny,nz, 16);
    return p;
}
void fftw_execute(const fftw_plan p){
    long nx=1,ny=1,nz=1; int bpe=16;
    int ok = plans_get((void*)p,&nx,&ny,&nz,&bpe);
    if (ok){ account(nx,ny,nz,bpe); log_per_call(nx,ny,nz,bpe,"fftw_execute"); }
    if (!is_dry_run()){
        static pf_fftw_execute real=NULL; RESOLVE_OR_DIE(pf_fftw_execute, real, "fftw_execute");
        real(p);
    }
}
void fftw_destroy_plan(fftw_plan p){
    plans_del((void*)p);
    static pf_fftw_destroy_plan real=NULL; RESOLVE_OR_DIE(pf_fftw_destroy_plan, real, "fftw_destroy_plan");
    real(p);
}
