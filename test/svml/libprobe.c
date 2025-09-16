// libprobe.c  -- drop-in 版，带机器峰值默认
#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// ---------------- TLS 汇总 ----------------
typedef struct {
    unsigned long long exp_d, exp_f, log_d, log_f, pow_d, pow_f;
    unsigned long long sin_d, sin_f, cos_d, cos_f, sqrt_d, sqrt_f;
    unsigned long long svml_expd_elems, svml_expf_elems;
    unsigned long long svml_logd_elems, svml_logf_elems;
    unsigned long long svml_sind_elems, svml_sinf_elems;
    unsigned long long svml_cosd_elems, svml_cosf_elems;
    unsigned long long svml_powd_elems, svml_powf_elems;
    unsigned long long memcpy_bytes;
} probe_tls_t;

static pthread_key_t g_tls_key;
static pthread_once_t g_tls_once = PTHREAD_ONCE_INIT;
static pthread_mutex_t g_sum_mu = PTHREAD_MUTEX_INITIALIZER;
static probe_tls_t g_sum = {0};

static void tls_merge_free(void* p) {
    probe_tls_t* t = (probe_tls_t*)p;
    if (!t) return;
    pthread_mutex_lock(&g_sum_mu);
#define ACC(f) g_sum.f += t->f
    ACC(exp_d); ACC(exp_f); ACC(log_d); ACC(log_f); ACC(pow_d); ACC(pow_f);
    ACC(sin_d); ACC(sin_f); ACC(cos_d); ACC(cos_f); ACC(sqrt_d); ACC(sqrt_f);
    ACC(svml_expd_elems); ACC(svml_expf_elems);
    ACC(svml_logd_elems); ACC(svml_logf_elems);
    ACC(svml_sind_elems); ACC(svml_sinf_elems);
    ACC(svml_cosd_elems); ACC(svml_cosf_elems);
    ACC(svml_powd_elems); ACC(svml_powf_elems);
    ACC(memcpy_bytes);
#undef ACC
    pthread_mutex_unlock(&g_sum_mu);
    free(t);
}
static void make_tls_key(void){ pthread_key_create(&g_tls_key, tls_merge_free); }
static inline probe_tls_t* get_tls(void){
    pthread_once(&g_tls_once, make_tls_key);
    probe_tls_t* t = (probe_tls_t*)pthread_getspecific(g_tls_key);
    if (!t) { t = (probe_tls_t*)calloc(1, sizeof(*t)); pthread_setspecific(g_tls_key, t); }
    return t;
}

// ---------------- 递归保护 ----------------
static __thread int __in = 0;
#define ENTER() do{ if(__in) return 0; __in=1; }while(0)
#define EXIT(rv) do{ __in=0; return (rv);}while(0)
#define ENTERV() do{ if(__in) return; __in=1; }while(0)
#define EXITV() do{ __in=0; return; }while(0)

// ---------------- 真符号指针 ----------------
static double (*real_exp)(double);
static float  (*real_expf)(float);
static double (*real_log)(double);
static float  (*real_logf)(float);
static double (*real_pow)(double,double);
static float  (*real_powf)(float,float);
static double (*real_sin)(double);
static float  (*real_sinf)(float);
static double (*real_cos)(double);
static float  (*real_cosf)(float);
static double (*real_sqrt)(double);
static float  (*real_sqrtf)(float);
static void*  (*real_memcpy)(void*, const void*, size_t);

// ---------------- 可选：SVML ----------------
#if (defined(__x86_64__) || defined(__i386))
  #include <immintrin.h>
  #ifdef __AVX512F__
static __m512d (*real___svml_exp8)(__m512d);
static __m512  (*real___svml_expf16)(__m512);
static __m512d (*real___svml_log8)(__m512d);
static __m512  (*real___svml_logf16)(__m512);
static __m512d (*real___svml_sin8)(__m512d);
static __m512  (*real___svml_sinf16)(__m512);
static __m512d (*real___svml_cos8)(__m512d);
static __m512  (*real___svml_cosf16)(__m512);
static __m512d (*real___svml_pow8)(__m512d, __m512d);
static __m512  (*real___svml_powf16)(__m512, __m512);
  #endif
  #ifdef __AVX__
static __m256d (*real___svml_exp4)(__m256d);
static __m256  (*real___svml_expf8)(__m256);
static __m256d (*real___svml_log4)(__m256d);
static __m256  (*real___svml_logf8)(__m256);
static __m256d (*real___svml_sin4)(__m256d);
static __m256  (*real___svml_sinf8)(__m256);
static __m256d (*real___svml_cos4)(__m256d);
static __m256  (*real___svml_cosf8)(__m256);
static __m256d (*real___svml_pow4)(__m256d, __m256d);
static __m256  (*real___svml_powf8)(__m256, __m256);
  #endif
#endif

// ---------------- CPU 频率 & 成本库 & 机器峰值 ----------------
static double g_cpu_hz = 3.5e9; // 备用值
// 你的机器峰值（默认，仍可被环境覆盖）
static double g_peak_gflops = 50.906; // GFLOP/s
static double g_mem_bw_gbs  = 14.53;  // GB/s
static double g_mem_bw_gbps = 116.24; // Gbit/s（仅展示）
static inline double envd(const char* k, double defv){
    const char* s = getenv(k);
    if(!s||!*s) return defv;
    char* e=0; double v=strtod(s,&e);
    return (e && e!=s) ? v : defv;
}
static void detect_cpu_hz(void){
    FILE* f=fopen("/proc/cpuinfo","r");
    if(f){
        char line[256];
        while(fgets(line,sizeof line,f)){
            double mhz;
            if(sscanf(line,"cpu MHz\t: %lf",&mhz)==1 && mhz>100.0){ g_cpu_hz = mhz*1e6; break; }
        }
        fclose(f);
    }
    g_cpu_hz = envd("PROBE_CPU_HZ", g_cpu_hz);
}

// 标量/向量 CPE 默认值（建议用环境变量覆盖）
static double cpe_expd=6.0, cpe_expf=4.0;
static double cpe_logd=6.0, cpe_logf=4.5;
static double cpe_powd=14.0, cpe_powf=10.0;
static double cpe_sind=5.0, cpe_sinf=4.0, cpe_cosd=5.0, cpe_cosf=4.0;
static double cpe_sqrtd=3.0, cpe_sqrtf=2.5;
static double cpe_svml_expd=4.5, cpe_svml_expf=3.0;
static double cpe_svml_logd=5.0, cpe_svml_logf=3.8;
static double cpe_svml_sind=4.5, cpe_svml_sinf=3.5;
static double cpe_svml_cosd=4.5, cpe_svml_cosf=3.5;
static double cpe_svml_powd=12.0, cpe_svml_powf=8.5;

// 带宽（B/s）默认取你的 14.53 GB/s
static double mem_bw_bps = 14.53 * 1e9;

static void load_costs_and_peaks(void){
#define OV(N,V) V = envd(N,V)
    // 峰值（可外部覆盖）
    OV("PROBE_PEAK_GFLOPS", g_peak_gflops);
    OV("PROBE_MEM_BW_GBS",  g_mem_bw_gbs);
    OV("PROBE_MEM_BW_GBPS", g_mem_bw_gbps);
    // 若用户只给了 GB/s，就据此改写 B/s
    mem_bw_bps = envd("PROBE_MEM_BW_BPS", g_mem_bw_gbs * 1e9);
    // CPE 覆盖
    OV("PROBE_CPE_expd", cpe_expd);   OV("PROBE_CPE_expf", cpe_expf);
    OV("PROBE_CPE_logd", cpe_logd);   OV("PROBE_CPE_logf", cpe_logf);
    OV("PROBE_CPE_powd", cpe_powd);   OV("PROBE_CPE_powf", cpe_powf);
    OV("PROBE_CPE_sind", cpe_sind);   OV("PROBE_CPE_sinf", cpe_sinf);
    OV("PROBE_CPE_cosd", cpe_cosd);   OV("PROBE_CPE_cosf", cpe_cosf);
    OV("PROBE_CPE_sqrtd", cpe_sqrtd); OV("PROBE_CPE_sqrtf", cpe_sqrtf);
    OV("PROBE_CPE_svml_expd", cpe_svml_expd); OV("PROBE_CPE_svml_expf", cpe_svml_expf);
    OV("PROBE_CPE_svml_logd", cpe_svml_logd); OV("PROBE_CPE_svml_logf", cpe_svml_logf);
    OV("PROBE_CPE_svml_sind", cpe_svml_sind); OV("PROBE_CPE_svml_sinf", cpe_svml_sinf);
    OV("PROBE_CPE_svml_cosd", cpe_svml_cosd); OV("PROBE_CPE_svml_cosf", cpe_svml_cosf);
    OV("PROBE_CPE_svml_powd", cpe_svml_powd); OV("PROBE_CPE_svml_powf", cpe_svml_powf);
#undef OV
}

// ---------------- 构造/析构 ----------------
__attribute__((constructor))
static void probe_init(void){
    __in=1;
    detect_cpu_hz(); load_costs_and_peaks();
    real_exp   = dlsym(RTLD_NEXT,"exp");
    real_expf  = dlsym(RTLD_NEXT,"expf");
    real_log   = dlsym(RTLD_NEXT,"log");
    real_logf  = dlsym(RTLD_NEXT,"logf");
    real_pow   = dlsym(RTLD_NEXT,"pow");
    real_powf  = dlsym(RTLD_NEXT,"powf");
    real_sin   = dlsym(RTLD_NEXT,"sin");
    real_sinf  = dlsym(RTLD_NEXT,"sinf");
    real_cos   = dlsym(RTLD_NEXT,"cos");
    real_cosf  = dlsym(RTLD_NEXT,"cosf");
    real_sqrt  = dlsym(RTLD_NEXT,"sqrt");
    real_sqrtf = dlsym(RTLD_NEXT,"sqrtf");
    real_memcpy= dlsym(RTLD_NEXT,"memcpy");
#if (defined(__x86_64__) || defined(__i386))
  #ifdef __AVX512F__
    real___svml_exp8   = dlsym(RTLD_NEXT,"__svml_exp8");
    real___svml_expf16 = dlsym(RTLD_NEXT,"__svml_expf16");
    real___svml_log8   = dlsym(RTLD_NEXT,"__svml_log8");
    real___svml_logf16 = dlsym(RTLD_NEXT,"__svml_logf16");
    real___svml_sin8   = dlsym(RTLD_NEXT,"__svml_sin8");
    real___svml_sinf16 = dlsym(RTLD_NEXT,"__svml_sinf16");
    real___svml_cos8   = dlsym(RTLD_NEXT,"__svml_cos8");
    real___svml_cosf16 = dlsym(RTLD_NEXT,"__svml_cosf16");
    real___svml_pow8   = dlsym(RTLD_NEXT,"__svml_pow8");
    real___svml_powf16 = dlsym(RTLD_NEXT,"__svml_powf16");
  #endif
  #ifdef __AVX__
    real___svml_exp4   = dlsym(RTLD_NEXT,"__svml_exp4");
    real___svml_expf8  = dlsym(RTLD_NEXT,"__svml_expf8");
    real___svml_log4   = dlsym(RTLD_NEXT,"__svml_log4");
    real___svml_logf8  = dlsym(RTLD_NEXT,"__svml_logf8");
    real___svml_sin4   = dlsym(RTLD_NEXT,"__svml_sin4");
    real___svml_sinf8  = dlsym(RTLD_NEXT,"__svml_sinf8");
    real___svml_cos4   = dlsym(RTLD_NEXT,"__svml_cos4");
    real___svml_cosf8  = dlsym(RTLD_NEXT,"__svml_cosf8");
    real___svml_pow4   = dlsym(RTLD_NEXT,"__svml_pow4");
    real___svml_powf8  = dlsym(RTLD_NEXT,"__svml_powf8");
  #endif
#endif
    __in=0;
}

__attribute__((destructor))
static void probe_fini(void){
    tls_merge_free(pthread_getspecific(g_tls_key)); // 合并主线程

    __in=1;
    const double f = (g_cpu_hz>1e6? g_cpu_hz : 3.5e9);
    // 标量
    double t_expd  = (g_sum.exp_d  * 6.0 /*占位*/ ); // 时间统一在下方按 CPE/f 计算
    // 直接用 CPE->时间（把上面“占位”一步到位写成表达式更清晰）：
#define T(calls, cpe) (((double)(calls) * (cpe)) / f)
    double t_expf  = T(g_sum.exp_f,  cpe_expf);
    double t_logd  = T(g_sum.log_d,  cpe_logd);
    double t_logf  = T(g_sum.log_f,  cpe_logf);
    double t_powd  = T(g_sum.pow_d,  cpe_powd);
    double t_powf  = T(g_sum.pow_f,  cpe_powf);
    double t_sind  = T(g_sum.sin_d,  cpe_sind);
    double t_sinf  = T(g_sum.sin_f,  cpe_sinf);
    double t_cosd  = T(g_sum.cos_d,  cpe_cosd);
    double t_cosf  = T(g_sum.cos_f,  cpe_cosf);
    double t_sqrtd = T(g_sum.sqrt_d, cpe_sqrtd);
    double t_sqrtf = T(g_sum.sqrt_f, cpe_sqrtf);
    // 修正 t_expd（上面占位）：
    t_expd = T(g_sum.exp_d, cpe_expd);

    // 向量
    double t_svml_expd = T(g_sum.svml_expd_elems, cpe_svml_expd);
    double t_svml_expf = T(g_sum.svml_expf_elems, cpe_svml_expf);
    double t_svml_logd = T(g_sum.svml_logd_elems, cpe_svml_logd);
    double t_svml_logf = T(g_sum.svml_logf_elems, cpe_svml_logf);
    double t_svml_sind = T(g_sum.svml_sind_elems, cpe_svml_sind);
    double t_svml_sinf = T(g_sum.svml_sinf_elems, cpe_svml_sinf);
    double t_svml_cosd = T(g_sum.svml_cosd_elems, cpe_svml_cosd);
    double t_svml_cosf = T(g_sum.svml_cosf_elems, cpe_svml_cosf);
    double t_svml_powd = T(g_sum.svml_powd_elems, cpe_svml_powd);
    double t_svml_powf = T(g_sum.svml_powf_elems, cpe_svml_powf);
#undef T
    // memcpy 以带宽近似
    double t_memcpy = (g_sum.memcpy_bytes>0 && mem_bw_bps>0)
                      ? ((double)g_sum.memcpy_bytes / mem_bw_bps) : 0.0;

    double t_sum = t_expd+t_expf+t_logd+t_logf+t_powd+t_powf+
                   t_sind+t_sinf+t_cosd+t_cosf+t_sqrtd+t_sqrtf+
                   t_svml_expd+t_svml_expf+t_svml_logd+t_svml_logf+
                   t_svml_sind+t_svml_sinf+t_svml_cosd+t_svml_cosf+
                   t_svml_powd+t_svml_powf + t_memcpy;

    fprintf(stderr,
      "{"
        "\"probe\":\"libm+svml\","
        "\"cpu_hz\":%.0f,"
        "\"machine_peak\":{\"gflops\":%.3f,\"mem_bw_gbs\":%.2f,\"mem_bw_gbps\":%.2f},"
        "\"costs\":{"
          "\"expd\":%.3f,\"expf\":%.3f,\"logd\":%.3f,\"logf\":%.3f,"
          "\"powd\":%.3f,\"powf\":%.3f,\"sind\":%.3f,\"sinf\":%.3f,"
          "\"cosd\":%.3f,\"cosf\":%.3f,\"sqrtd\":%.3f,\"sqrtf\":%.3f,"
          "\"svml_expd\":%.3f,\"svml_expf\":%.3f,\"svml_logd\":%.3f,\"svml_logf\":%.3f,"
          "\"svml_sind\":%.3f,\"svml_sinf\":%.3f,\"svml_cosd\":%.3f,\"svml_cosf\":%.3f,"
          "\"svml_powd\":%.3f,\"svml_powf\":%.3f,"
          "\"mem_bw_bps\":%.0f"
        "},"
        "\"counts\":{"
          "\"scalar\":{"
            "\"exp_d\":%llu,\"exp_f\":%llu,\"log_d\":%llu,\"log_f\":%llu,"
            "\"pow_d\":%llu,\"pow_f\":%llu,\"sin_d\":%llu,\"sin_f\":%llu,"
            "\"cos_d\":%llu,\"cos_f\":%llu,\"sqrt_d\":%llu,\"sqrt_f\":%llu"
          "},"
          "\"svml_elems\":{"
            "\"exp_d\":%llu,\"exp_f\":%llu,\"log_d\":%llu,\"log_f\":%llu,"
            "\"sin_d\":%llu,\"sin_f\":%llu,\"cos_d\":%llu,\"cos_f\":%llu,"
            "\"pow_d\":%llu,\"pow_f\":%llu"
          "},"
          "\"memcpy_bytes\":%llu"
        "},"
        "\"time_s\":{"
          "\"expd\":%.6f,\"expf\":%.6f,\"logd\":%.6f,\"logf\":%.6f,"
          "\"powd\":%.6f,\"powf\":%.6f,\"sind\":%.6f,\"sinf\":%.6f,"
          "\"cosd\":%.6f,\"cosf\":%.6f,\"sqrtd\":%.6f,\"sqrtf\":%.6f,"
          "\"svml_expd\":%.6f,\"svml_expf\":%.6f,\"svml_logd\":%.6f,\"svml_logf\":%.6f,"
          "\"svml_sind\":%.6f,\"svml_sinf\":%.6f,\"svml_cosd\":%.6f,\"svml_cosf\":%.6f,"
          "\"svml_powd\":%.6f,\"svml_powf\":%.6f,"
          "\"memcpy\":%.6f,"
          "\"sum\":%.6f"
        "}"
      "}\n",
      f,
      g_peak_gflops, g_mem_bw_gbs, g_mem_bw_gbps,
      cpe_expd,cpe_expf,cpe_logd,cpe_logf,cpe_powd,cpe_powf,
      cpe_sind,cpe_sinf,cpe_cosd,cpe_cosf,cpe_sqrtd,cpe_sqrtf,
      cpe_svml_expd,cpe_svml_expf,cpe_svml_logd,cpe_svml_logf,
      cpe_svml_sind,cpe_svml_sinf,cpe_svml_cosd,cpe_svml_cosf,
      cpe_svml_powd,cpe_svml_powf,mem_bw_bps,
      g_sum.exp_d,g_sum.exp_f,g_sum.log_d,g_sum.log_f,
      g_sum.pow_d,g_sum.pow_f,g_sum.sin_d,g_sum.sin_f,
      g_sum.cos_d,g_sum.cos_f,g_sum.sqrt_d,g_sum.sqrt_f,
      g_sum.svml_expd_elems,g_sum.svml_expf_elems,g_sum.svml_logd_elems,g_sum.svml_logf_elems,
      g_sum.svml_sind_elems,g_sum.svml_sinf_elems,g_sum.svml_cosd_elems,g_sum.svml_cosf_elems,
      g_sum.svml_powd_elems,g_sum.svml_powf_elems,
      g_sum.memcpy_bytes,
      t_expd,t_expf,t_logd,t_logf,t_powd,t_powf,t_sind,t_sinf,
      t_cosd,t_cosf,t_sqrtd,t_sqrtf,
      t_svml_expd,t_svml_expf,t_svml_logd,t_svml_logf,
      t_svml_sind,t_svml_sinf,t_svml_cosd,t_svml_cosf,
      t_svml_powd,t_svml_powf,
      t_memcpy,t_sum
    );
    __in=0;
}

// ---------------- 标量 libm 包装 ----------------
double exp(double x){ ENTER(); probe_tls_t* t=get_tls(); t->exp_d++; double r=real_exp(x); EXIT(r); }
float  expf(float x){ ENTER(); probe_tls_t* t=get_tls(); t->exp_f++; float  r=real_expf(x); EXIT(r); }
double log(double x){ ENTER(); probe_tls_t* t=get_tls(); t->log_d++; double r=real_log(x); EXIT(r); }
float  logf(float x){ ENTER(); probe_tls_t* t=get_tls(); t->log_f++; float  r=real_logf(x); EXIT(r); }
double pow(double x,double y){ ENTER(); probe_tls_t* t=get_tls(); t->pow_d++; double r=real_pow(x,y); EXIT(r); }
float  powf(float x,float y){ ENTER(); probe_tls_t* t=get_tls(); t->pow_f++; float  r=real_powf(x,y); EXIT(r); }
double sin(double x){ ENTER(); probe_tls_t* t=get_tls(); t->sin_d++; double r=real_sin(x); EXIT(r); }
float  sinf(float x){ ENTER(); probe_tls_t* t=get_tls(); t->sin_f++; float  r=real_sinf(x); EXIT(r); }
double cos(double x){ ENTER(); probe_tls_t* t=get_tls(); t->cos_d++; double r=real_cos(x); EXIT(r); }
float  cosf(float x){ ENTER(); probe_tls_t* t=get_tls(); t->cos_f++; float  r=real_cosf(x); EXIT(r); }
double sqrt(double x){ ENTER(); probe_tls_t* t=get_tls(); t->sqrt_d++; double r=real_sqrt(x); EXIT(r); }
float  sqrtf(float x){ ENTER(); probe_tls_t* t=get_tls(); t->sqrt_f++; float  r=real_sqrtf(x); EXIT(r); }

// ---------------- memcpy 包装 ----------------
void* memcpy(void* dst, const void* src, size_t n){
    if(__in) return real_memcpy(dst,src,n);
    __in=1;
    if(n>0){ probe_tls_t* t=get_tls(); t->memcpy_bytes += n; }
    void* r = real_memcpy(dst,src,n);
    __in=0; return r;
}

// ---------------- SVML 包装（可选） ----------------
#if (defined(__x86_64__) || defined(__i386))
  #ifdef __AVX512F__
__m512d __svml_exp8(__m512d x){ if(__in) return real___svml_exp8(x); __in=1; get_tls()->svml_expd_elems += 8;  __m512d r=real___svml_exp8(x);  __in=0; return r; }
__m512  __svml_expf16(__m512 x){ if(__in) return real___svml_expf16(x); __in=1; get_tls()->svml_expf_elems += 16; __m512  r=real___svml_expf16(x); __in=0; return r; }
__m512d __svml_log8(__m512d x){ if(__in) return real___svml_log8(x); __in=1; get_tls()->svml_logd_elems += 8;  __m512d r=real___svml_log8(x);  __in=0; return r; }
__m512  __svml_logf16(__m512 x){ if(__in) return real___svml_logf16(x); __in=1; get_tls()->svml_logf_elems += 16; __m512  r=real___svml_logf16(x); __in=0; return r; }
__m512d __svml_sin8(__m512d x){ if(__in) return real___svml_sin8(x); __in=1; get_tls()->svml_sind_elems += 8;  __m512d r=real___svml_sin8(x);  __in=0; return r; }
__m512  __svml_sinf16(__m512 x){ if(__in) return real___svml_sinf16(x); __in=1; get_tls()->svml_sinf_elems += 16; __m512  r=real___svml_sinf16(x); __in=0; return r; }
__m512d __svml_cos8(__m512d x){ if(__in) return real___svml_cos8(x); __in=1; get_tls()->svml_cosd_elems += 8;  __m512d r=real___svml_cos8(x);  __in=0; return r; }
__m512  __svml_cosf16(__m512 x){ if(__in) return real___svml_cosf16(x); __in=1; get_tls()->svml_cosf_elems += 16; __m512  r=real___svml_cosf16(x); __in=0; return r; }
__m512d __svml_pow8(__m512d a,__m512d b){ if(__in) return real___svml_pow8(a,b); __in=1; get_tls()->svml_powd_elems += 8;  __m512d r=real___svml_pow8(a,b);  __in=0; return r; }
__m512  __svml_powf16(__m512 a,__m512 b){ if(__in) return real___svml_powf16(a,b); __in=1; get_tls()->svml_powf_elems += 16; __m512  r=real___svml_powf16(a,b); __in=0; return r; }
  #endif
  #ifdef __AVX__
__m256d __svml_exp4(__m256d x){ if(__in) return real___svml_exp4(x); __in=1; get_tls()->svml_expd_elems += 4;  __m256d r=real___svml_exp4(x);  __in=0; return r; }
__m256  __svml_expf8(__m256 x){ if(__in) return real___svml_expf8(x); __in=1; get_tls()->svml_expf_elems += 8;  __m256  r=real___svml_expf8(x);  __in=0; return r; }
__m256d __svml_log4(__m256d x){ if(__in) return real___svml_log4(x); __in=1; get_tls()->svml_logd_elems += 4;  __m256d r=real___svml_log4(x);  __in=0; return r; }
__m256  __svml_logf8(__m256 x){ if(__in) return real___svml_logf8(x); __in=1; get_tls()->svml_logf_elems += 8;  __m256  r=real___svml_logf8(x);  __in=0; return r; }
__m256d __svml_sin4(__m256d x){ if(__in) return real___svml_sin4(x); __in=1; get_tls()->svml_sind_elems += 4;  __m256d r=real___svml_sin4(x);  __in=0; return r; }
__m256  __svml_sinf8(__m256 x){ if(__in) return real___svml_sinf8(x); __in=1; get_tls()->svml_sinf_elems += 8;  __m256  r=real___svml_sinf8(x);  __in=0; return r; }
__m256d __svml_cos4(__m256d x){ if(__in) return real___svml_cos4(x); __in=1; get_tls()->svml_cosd_elems += 4;  __m256d r=real___svml_cos4(x);  __in=0; return r; }
__m256  __svml_cosf8(__m256 x){ if(__in) return real___svml_cosf8(x); __in=1; get_tls()->svml_cosf_elems += 8;  __m256  r=real___svml_cosf8(x);  __in=0; return r; }
__m256d __svml_pow4(__m256d a,__m256d b){ if(__in) return real___svml_pow4(a,b); __in=1; get_tls()->svml_powd_elems += 4;  __m256d r=real___svml_pow4(a,b);  __in=0; return r; }
__m256  __svml_powf8(__m256 a,__m256 b){ if(__in) return real___svml_powf8(a,b); __in=1; get_tls()->svml_powf_elems += 8;  __m256  r=real___svml_powf8(a,b);  __in=0; return r; }
  #endif
#endif
