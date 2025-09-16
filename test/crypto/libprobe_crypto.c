// libprobe_crypto.c  — LD_PRELOAD for OpenSSL SHA256/CMAC/RAND + memcpy
#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

// ─────────── TLS 统计 ───────────
typedef struct {
    unsigned long long sha256_bytes;
    unsigned long long cmac_bytes;    // 累计更新的消息字节
    unsigned long long cmac_blocks;   // 逐上下文 ceil(bytes/16) 的总和
    unsigned long long rng_bytes;
    unsigned long long memcpy_bytes;
    // 简易 CMAC 上下文跟踪（每线程最多 256 个活跃 ctx）
    struct { void* ctx; unsigned long long bytes; unsigned char active; } cmac[256];
} tls_t;

static pthread_key_t gkey;
static pthread_once_t gonce = PTHREAD_ONCE_INIT;
static pthread_mutex_t gsum_mu = PTHREAD_MUTEX_INITIALIZER;
static tls_t gsum = {0};
static __thread int __in=0;

static void tls_dtor(void* p){
    tls_t* t=(tls_t*)p; if(!t) return;
    pthread_mutex_lock(&gsum_mu);
    gsum.sha256_bytes += t->sha256_bytes;
    gsum.cmac_bytes   += t->cmac_bytes;
    gsum.cmac_blocks  += t->cmac_blocks;
    gsum.rng_bytes    += t->rng_bytes;
    gsum.memcpy_bytes += t->memcpy_bytes;
    pthread_mutex_unlock(&gsum_mu);
    free(t);
}
static void make_key(void){ pthread_key_create(&gkey, tls_dtor); }
static inline tls_t* T(){
    pthread_once(&gonce, make_key);
    tls_t* t = (tls_t*)pthread_getspecific(gkey);
    if(!t){ t=(tls_t*)calloc(1,sizeof(*t)); pthread_setspecific(gkey,t); }
    return t;
}
static inline void cmac_ctx_reset(tls_t* t, void* ctx){
    for(int i=0;i<256;i++) if(!t->cmac[i].active){ t->cmac[i].ctx=ctx; t->cmac[i].bytes=0; t->cmac[i].active=1; return; }
}
static inline void cmac_ctx_add(tls_t* t, void* ctx, size_t len){
    for(int i=0;i<256;i++) if(t->cmac[i].active && t->cmac[i].ctx==ctx){ t->cmac[i].bytes += len; return; }
    // 未见 Init 也允许记录（容错）
    cmac_ctx_reset(t, ctx);
    for(int i=0;i<256;i++) if(t->cmac[i].active && t->cmac[i].ctx==ctx){ t->cmac[i].bytes += len; return; }
}
static inline void cmac_ctx_final(tls_t* t, void* ctx){
    for(int i=0;i<256;i++) if(t->cmac[i].active && t->cmac[i].ctx==ctx){
        unsigned long long B = t->cmac[i].bytes;
        t->cmac_bytes += B;
        t->cmac_blocks += (B + 15ULL)/16ULL; // 每条消息 ceil(bytes/16)
        t->cmac[i].active=0; t->cmac[i].ctx=NULL; t->cmac[i].bytes=0;
        return;
    }
}

// ─────────── 真符号（用 void* 原型避免强依赖 OpenSSL 头） ───────────
static unsigned char* (*real_SHA256)(const unsigned char*, size_t, unsigned char*);
static int  (*real_SHA256_Update)(void*, const void*, size_t);
static void*(*real_CMAC_CTX_new)(void);
static int  (*real_CMAC_Init)(void*, const void*, size_t, const void*, void*);
static int  (*real_CMAC_Update)(void*, const void*, size_t);
static int  (*real_CMAC_Final)(void*, unsigned char*, size_t*);
static void (*real_CMAC_CTX_free)(void*);
static int  (*real_RAND_bytes)(unsigned char*, int);
static void*(*real_memcpy)(void*, const void*, size_t);

// ─────────── 频率/成本（ENV 可覆盖） ───────────
static double CPU_HZ = 3.5e9;
static double CPB_SHA256 = 11.0;   // cycles per byte（默认占位，建议校准）
static double CPB_AES128 = 2.0;    // cycles per byte（AES-128 加密，默认占位）
static double MEM_BW_BPS = 14.53e9; // 你的带宽默认（B/s）

static double envd(const char* k, double v){ const char* s=getenv(k); if(!s||!*s) return v; char* e=0; double x=strtod(s,&e); return (e&&e!=s)?x:v; }
static void detect_cpu_hz(void){
    FILE* f=fopen("/proc/cpuinfo","r");
    if(f){ char line[256]; while(fgets(line,sizeof line,f)){ double mhz; if(sscanf(line,"cpu MHz\t: %lf",&mhz)==1){ CPU_HZ = mhz*1e6; break; } } fclose(f); }
    CPU_HZ = envd("PROBE_CPU_HZ", CPU_HZ);
    CPB_SHA256 = envd("PROBE_CPB_SHA256", CPB_SHA256);
    CPB_AES128 = envd("PROBE_CPB_AES128", CPB_AES128);
    MEM_BW_BPS = envd("PROBE_MEM_BW_BPS", MEM_BW_BPS);
}

// ─────────── 构造/析构 ───────────
__attribute__((constructor))
static void init_probe(void){
    __in=1;
    detect_cpu_hz();
    real_SHA256        = dlsym(RTLD_NEXT, "SHA256");
    real_SHA256_Update = dlsym(RTLD_NEXT, "SHA256_Update");
    real_CMAC_CTX_new  = dlsym(RTLD_NEXT, "CMAC_CTX_new");
    real_CMAC_Init     = dlsym(RTLD_NEXT, "CMAC_Init");
    real_CMAC_Update   = dlsym(RTLD_NEXT, "CMAC_Update");
    real_CMAC_Final    = dlsym(RTLD_NEXT, "CMAC_Final");
    real_CMAC_CTX_free = dlsym(RTLD_NEXT, "CMAC_CTX_free");
    real_RAND_bytes    = dlsym(RTLD_NEXT, "RAND_bytes");
    real_memcpy        = dlsym(RTLD_NEXT, "memcpy");
    __in=0;
}

__attribute__((destructor))
static void fini_probe(void){
    tls_dtor(pthread_getspecific(gkey)); // 合并主线程 TLS

    __in=1;
    double t_sha  = (gsum.sha256_bytes * CPB_SHA256) / (CPU_HZ>1?CPU_HZ:1);
    double t_aes  = (gsum.cmac_blocks * 16.0 * CPB_AES128) / (CPU_HZ>1?CPU_HZ:1);
    double t_mem  = (gsum.memcpy_bytes>0 && MEM_BW_BPS>0) ? ((double)gsum.memcpy_bytes / MEM_BW_BPS) : 0.0;
    double t_rng  = 0.0; // 可按需要增加 RAND 的 cpb 模型
    double t_sum  = t_sha + t_aes + t_mem + t_rng;

    fprintf(stderr,
      "{"
        "\"probe\":\"openssl+memcpy\","
        "\"cpu_hz\":%.0f,"
        "\"costs\":{\"cpb_sha256\":%.3f,\"cpb_aes128\":%.3f,\"mem_bw_bps\":%.0f},"
        "\"counts\":{\"sha256_bytes\":%llu,\"cmac_bytes\":%llu,\"cmac_blocks\":%llu,"
                   "\"rand_bytes\":%llu,\"memcpy_bytes\":%llu},"
        "\"time_s\":{\"sha256\":%.6f,\"aes128_cmac\":%.6f,\"memcpy\":%.6f,\"rng\":%.6f,\"sum\":%.6f}"
      "}\n",
      CPU_HZ, CPB_SHA256, CPB_AES128, MEM_BW_BPS,
      gsum.sha256_bytes, gsum.cmac_bytes, gsum.cmac_blocks,
      gsum.rng_bytes, gsum.memcpy_bytes,
      t_sha, t_aes, t_mem, t_rng, t_sum
    );
    __in=0;
}

// ─────────── 拦截实现 ───────────
unsigned char* SHA256(const unsigned char* d, size_t n, unsigned char* md){
    if(__in) return real_SHA256(d,n,md);
    __in=1; if(n) T()->sha256_bytes += n; unsigned char* r = real_SHA256(d,n,md); __in=0; return r;
}
int SHA256_Update(void* ctx, const void* data, size_t len){
    if(__in) return real_SHA256_Update(ctx,data,len);
    __in=1; if(len) T()->sha256_bytes += len; int rv = real_SHA256_Update(ctx,data,len); __in=0; return rv;
}

// CMAC：按 ctx 逐条消息统计
void* CMAC_CTX_new(void){ return real_CMAC_CTX_new(); }
int CMAC_Init(void* ctx, const void* key, size_t keylen, const void* cipher, void* engine){
    if(__in) return real_CMAC_Init(ctx,key,keylen,cipher,engine);
    __in=1; cmac_ctx_reset(T(), ctx); int rv = real_CMAC_Init(ctx,key,keylen,cipher,engine); __in=0; return rv;
}
int CMAC_Update(void* ctx, const void* data, size_t len){
    if(__in) return real_CMAC_Update(ctx,data,len);
    __in=1; if(len) cmac_ctx_add(T(), ctx, len); int rv = real_CMAC_Update(ctx,data,len); __in=0; return rv;
}
int CMAC_Final(void* ctx, unsigned char* out, size_t* outlen){
    if(__in) return real_CMAC_Final(ctx,out,outlen);
    __in=1; cmac_ctx_final(T(), ctx); int rv = real_CMAC_Final(ctx,out,outlen); __in=0; return rv;
}
void CMAC_CTX_free(void* ctx){
    // 若调用方未 Final 就 free，这里也结算一次（保守）
    if(!__in){ __in=1; cmac_ctx_final(T(), ctx); __in=0; }
    real_CMAC_CTX_free(ctx);
}

// RAND_bytes
int RAND_bytes(unsigned char* buf, int num){
    if(__in) return real_RAND_bytes(buf,num);
    __in=1; if(num>0) T()->rng_bytes += (unsigned)num; int rv = real_RAND_bytes(buf,num); __in=0; return rv;
}

// memcpy（带宽近似）
void* memcpy(void* dst, const void* src, size_t n){
    if(__in) return real_memcpy(dst,src,n);
    __in=1; if(n) T()->memcpy_bytes += n; void* r = real_memcpy(dst,src,n); __in=0; return r;
}
