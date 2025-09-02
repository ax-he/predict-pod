#include <stdio.h>
#include "mkl.h"              // 包含 cblas_* 和 mkl_service.h

int main() {
    // 打印 MKL 版本
    char ver[198];
    mkl_get_version_string(ver, sizeof(ver));
    printf("MKL Version: %s\n", ver);

    // 做一次极小的 DGEMM：C = alpha*A*B + beta*C
    const int n = 2;
    double A[4] = {1,2,3,4};
    double B[4] = {5,6,7,8};
    double C[4] = {0,0,0,0};
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                n, n, n, 1.0, A, n, B, n, 0.0, C, n);
    printf("C = [%.1f %.1f; %.1f %.1f]\n", C[0], C[1], C[2], C[3]);
    return 0;
}
