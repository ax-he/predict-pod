// sgemm_test.c
#include <cblas.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  int M=512,N=512,K=512;
  if (argc==4) { M=atoi(argv[1]); N=atoi(argv[2]); K=atoi(argv[3]); }
  float *A = (float*)malloc((size_t)M*K*sizeof(float));
  float *B = (float*)malloc((size_t)K*N*sizeof(float));
  float *C = (float*)malloc((size_t)M*N*sizeof(float));
  for (size_t i=0;i<(size_t)M*K;i++) A[i]=1.0f;
  for (size_t i=0;i<(size_t)K*N;i++) B[i]=1.0f;

  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
  printf("C[0]=%f\n", C[0]);
  free(A); free(B); free(C);
  return 0;
}
