// libopaudit.c — 统计所有经 PLT 的库符号调用次数；在进程退出时输出一行 JSON。
// x86-64 正确原型：第 5 个参数为 La_x86_64_regs*
#define _GNU_SOURCE
#include <link.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N 8192
typedef struct { char *name; unsigned long long calls; } Ent;
static Ent tab[N];
static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;

static unsigned H(const char*s){ unsigned x=5381; for(;*s;s++) x=((x<<5)+x)^*s; return x&(N-1); }
static Ent* upsert(const char*s){
  unsigned i=H(s);
  for(unsigned k=0;k<N;k++){
    unsigned j=(i+k)&(N-1);
    if(!tab[j].name){ tab[j].name=strdup(s); tab[j].calls=0; return &tab[j]; }
    if(strcmp(tab[j].name,s)==0) return &tab[j];
  }
  return NULL;
}

unsigned int la_version(unsigned int v){ (void)v; return LAV_CURRENT; }

unsigned int la_objopen(struct link_map* m, Lmid_t l, uintptr_t* c){
  (void)m; (void)l; (void)c;
  return LA_FLG_BINDTO | LA_FLG_BINDFROM;
}

Elf64_Addr la_x86_64_gnu_pltenter(Elf64_Sym* sym, unsigned int ndx,
  uintptr_t* refcook, uintptr_t* defcook, La_x86_64_regs* regs,
  unsigned int* flags, const char* symname, long int* framesizep)
{
  (void)ndx; (void)refcook; (void)defcook; (void)regs; (void)flags;
  if (framesizep) *framesizep = 0;  // 不为 pltexit 预留栈
  if (symname){
    pthread_mutex_lock(&mu);
    Ent* e = upsert(symname);
    if (e) e->calls++;
    pthread_mutex_unlock(&mu);
  }
  return sym->st_value;
}

__attribute__((destructor))
static void dump(void){
  const char* out = getenv("OPAUDIT_OUT");
  FILE* fp = out && *out ? fopen(out, "w") : stderr;
  if (!fp) fp = stderr;

  fprintf(fp,"{\"audit\":\"plt-calls\",\"items\":[");
  int first=1;
  for(int i=0;i<N;i++) if(tab[i].name && tab[i].calls){
    if(!first) fprintf(fp,","); first=0;
    fprintf(fp,"{\"sym\":\"%s\",\"calls\":%llu}",tab[i].name,tab[i].calls);
  }
  fprintf(fp,"]}\n");
  if (fp!=stderr) fclose(fp);
}
