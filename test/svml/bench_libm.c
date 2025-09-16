// bench_libm.c  -- 在大数组上重复调用单个函数，便于用 perf 统计 cycles
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#ifndef N
#define N (1<<26)           // 64M 元素，确保远大于 LLC；可按内存调整
#endif
#ifndef REPEAT
#define REPEAT 4            // 重复次数增加测量稳定度
#endif

static double now_ns(void){
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec*1e9 + ts.tv_nsec;
}

static float  *af; static double *ad;

#define WARM_F(fun) do{ for (size_t i=0;i<N;i++) af[i]=fun(af[i]); }while(0)
#define WARM_D(fun) do{ for (size_t i=0;i<N;i++) ad[i]=fun(ad[i]); }while(0)

typedef enum {F_EXPF, F_LOGF, F_SINF, F_COSF, F_POWF, F_SQRTF,
              F_EXP,  F_LOG,  F_SIN,  F_COS,  F_POW,  F_SQRT} fkind_t;

int main(int argc, char** argv){
    if(argc<2){
        fprintf(stderr,"usage: %s [expf|logf|sinf|cosf|powf|sqrtf|exp|log|sin|cos|pow|sqrt]\n", argv[0]);
        return 1;
    }
    fkind_t k; int isf=1;
    if      (!strcmp(argv[1],"expf")) k=F_EXPF;
    else if (!strcmp(argv[1],"logf")) k=F_LOGF;
    else if (!strcmp(argv[1],"sinf")) k=F_SINF;
    else if (!strcmp(argv[1],"cosf")) k=F_COSF;
    else if (!strcmp(argv[1],"powf")) k=F_POWF;
    else if (!strcmp(argv[1],"sqrtf"))k=F_SQRTF;
    else { isf=0;
      if      (!strcmp(argv[1],"exp")) k=F_EXP;
      else if (!strcmp(argv[1],"log")) k=F_LOG;
      else if (!strcmp(argv[1],"sin")) k=F_SIN;
      else if (!strcmp(argv[1],"cos")) k=F_COS;
      else if (!strcmp(argv[1],"pow")) k=F_POW;
      else if (!strcmp(argv[1],"sqrt"))k=F_SQRT;
      else { fprintf(stderr,"unknown func\n"); return 2; }
    }

    size_t bytes = (isf? sizeof(float):sizeof(double)) * (size_t)N;
    void* buf = aligned_alloc(64, bytes);
    if(!buf){ perror("alloc"); return 3; }
    if(isf){ af=(float*)buf; for(size_t i=0;i<N;i++) af[i]=(float)(i%97)/97.0f; }
    else   { ad=(double*)buf;for(size_t i=0;i<N;i++) ad[i]=(double)(i%97)/97.0; }

    // 预热，驱逐冷启动影响
    switch(k){
      case F_EXPF:  WARM_F(expf); break;
      case F_LOGF:  WARM_F(logf); break;
      case F_SINF:  WARM_F(sinf); break;
      case F_COSF:  WARM_F(cosf); break;
      case F_POWF:  for(size_t i=0;i<N;i++) af[i]=powf(af[i], 1.2345f); break;
      case F_SQRTF: WARM_F(sqrtf); break;
      case F_EXP:   WARM_D(exp);  break;
      case F_LOG:   WARM_D(log);  break;
      case F_SIN:   WARM_D(sin);  break;
      case F_COS:   WARM_D(cos);  break;
      case F_POW:   for(size_t i=0;i<N;i++) ad[i]=pow(ad[i], 1.23456789); break;
      case F_SQRT:  WARM_D(sqrt); break;
    }

    // 正式计时（ns）；cycles 建议用 perf 来测
    double t0 = now_ns();
    for(int r=0;r<REPEAT;r++){
      switch(k){
        case F_EXPF:  WARM_F(expf); break;
        case F_LOGF:  WARM_F(logf); break;
        case F_SINF:  WARM_F(sinf); break;
        case F_COSF:  WARM_F(cosf); break;
        case F_POWF:  for(size_t i=0;i<N;i++) af[i]=powf(af[i], 1.2345f); break;
        case F_SQRTF: WARM_F(sqrtf); break;
        case F_EXP:   WARM_D(exp);  break;
        case F_LOG:   WARM_D(log);  break;
        case F_SIN:   WARM_D(sin);  break;
        case F_COS:   WARM_D(cos);  break;
        case F_POW:   for(size_t i=0;i<N;i++) ad[i]=pow(ad[i], 1.23456789); break;
        case F_SQRT:  WARM_D(sqrt); break;
      }
    }
    double t1 = now_ns();
    double ns = (t1 - t0);
    double elems = (double)N * REPEAT;

    // 输出基本量，便于用 perf 的 cycles 计算 CPE
    printf("FUNC=%s ELEMS=%.0f BYTES=%zu ELAPSED_NS=%.0f\n",
           argv[1], elems, bytes, ns);
    free(buf);
    return 0;
}
