// test_app.c -- OpenSSL SHA256/CMAC + OpenBLAS GEMM + memcpy/IO + libm
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>

#include <openssl/evp.h>
#include <openssl/cmac.h>
#include <openssl/sha.h>
#include <openssl/rand.h>

#include <cblas.h>   // OpenBLAS / CBLAS

static void die(const char* m){ perror(m); exit(1); }

int main(){
    const size_t SZ = 1<<22;           // 4 MiB payload
    unsigned char* buf = (unsigned char*)aligned_alloc(64, SZ);
    if(!buf) die("alloc");
    RAND_bytes(buf, (int)SZ);

    // —— SHA256: 分块 Update —— //
    unsigned char md[SHA256_DIGEST_LENGTH];
    SHA256_CTX sctx; SHA256_Init(&sctx);
    size_t off=0, CH=8192;
    while(off<SZ){ size_t n = (SZ-off>CH?CH:SZ-off); SHA256_Update(&sctx, buf+off, n); off+=n; }
    SHA256_Final(md, &sctx);

    // —— CMAC(AES-128): 分块 Update —— //
    unsigned char key[16]; RAND_bytes(key, sizeof key);
    CMAC_CTX* cctx = CMAC_CTX_new();
    CMAC_Init(cctx, key, sizeof key, EVP_aes_128_cbc(), NULL);
    off=0;
    while(off<SZ){ size_t n = (SZ-off>CH?CH:SZ-off); CMAC_Update(cctx, buf+off, n); off+=n; }
    unsigned char tag[16]; size_t tlen=0;
    CMAC_Final(cctx, tag, &tlen);
    CMAC_CTX_free(cctx);

    // —— memcpy/memmove —— //
    unsigned char* cp = (unsigned char*)aligned_alloc(64, SZ);
    if(!cp) die("alloc2");
    memcpy(cp, buf, SZ);
    memmove(buf+128, buf, SZ-128);

    // —— I/O: 写入再读出 —— //
    int fd = open("tmp.bin", O_CREAT|O_TRUNC|O_RDWR, 0644);
    if(fd<0) die("open");
    if(write(fd, buf, SZ)!=(ssize_t)SZ) die("write");
    lseek(fd, 0, SEEK_SET);
    if(read(fd, cp, SZ)!=(ssize_t)SZ) die("read");
    close(fd);

    // —— libm：若干 transcendentals —— //
    volatile float acc=0.f;
    for(int i=0;i<10*1024*1024;i++){ float x=(float)(i%1000)/1000.f; acc += sinf(x)*expf(x); }
    fprintf(stderr,"acc=%.3f\n",acc);

    // —— BLAS GEMM: C = alpha*A*B + beta*C —— //
    int N=512; double alpha=1.0, beta=0.0;
    double *A=(double*)aligned_alloc(64, sizeof(double)*N*N);
    double *B=(double*)aligned_alloc(64, sizeof(double)*N*N);
    double *C=(double*)aligned_alloc(64, sizeof(double)*N*N);
    if(!A||!B||!C) die("alloc gemm");
    for(long i=0;i<(long)N*N;i++){ A[i]=((i%97)-48)/50.0; B[i]=((i%89)-44)/40.0; C[i]=0.0; }
    // Row-major, no-transpose
cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            N, N, N,               // M,N,K
            alpha,
            A, /*lda=*/N,          // 若改成任意 M,N,K，通用写法应是 lda=K
            B, /*ldb=*/N,          // 行主存下 B 的步长是列数 N
            beta,
            C, /*ldc=*/N);
    remove("tmp.bin");
    free(A);free(B);free(C); free(cp); free(buf);
    return 0;
}
