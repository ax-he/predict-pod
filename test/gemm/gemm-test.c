// gemm_test.c
#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>

int main() {
    int M=2048, N=2048, K=2048;
    double *A = (double*)aligned_alloc(64, (size_t)M*K*sizeof(double));
    double *B = (double*)aligned_alloc(64, (size_t)K*N*sizeof(double));
    double *C = (double*)aligned_alloc(64, (size_t)M*N*sizeof(double));
    for (long i=0;i<(long)M*K;i++) A[i] = 1.0;
    for (long i=0;i<(long)K*N;i++) B[i] = 1.0;
    for (long i=0;i<(long)M*N;i++) C[i] = 0.0;

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, 1.0, A, K, B, N, 0.0, C, N);
    printf("done\n");
    free(A); free(B); free(C);
    return 0;
}
