// fft_test.c — 用 FFTW3/FFTW3F 生成多组 1D/2D/3D 计算，触发探针统计
#include <stdio.h>
#include <stdlib.h>
#include <fftw3.h>

static void run_fftw_double_1d(int n){
    fftw_complex *in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*n);
    fftw_complex *out= (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*n);
    for (int i=0;i<n;i++){ in[i][0]=i; in[i][1]=0; }
    fftw_plan p = fftw_plan_dft_1d(n, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(p);
    fftw_destroy_plan(p);
    fftw_free(in); fftw_free(out);
}

static void run_fftw_double_2d(int nx,int ny){
    fftw_complex *in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*nx*ny);
    fftw_complex *out= (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*nx*ny);
    for (int i=0;i<nx*ny;i++){ in[i][0]=i%7; in[i][1]=0.0; }
    fftw_plan p = fftw_plan_dft_2d(nx,ny, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(p);
    fftw_destroy_plan(p);
    fftw_free(in); fftw_free(out);
}

static void run_fftw_double_3d(int nx,int ny,int nz){
    fftw_complex *in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*nx*ny*nz);
    fftw_complex *out= (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*nx*ny*nz);
    for (int i=0;i<nx*ny*nz;i++){ in[i][0]=(i%5)*0.1; in[i][1]=0.0; }
    fftw_plan p = fftw_plan_dft_3d(nx,ny,nz, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(p);
    fftw_destroy_plan(p);
    fftw_free(in); fftw_free(out);
}

static void run_fftw_float_1d(int n){
    fftwf_complex *in = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex)*n);
    fftwf_complex *out= (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex)*n);
    for (int i=0;i<n;i++){ in[i][0]=i*0.01f; in[i][1]=0.0f; }
    fftwf_plan p = fftwf_plan_dft_1d(n, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftwf_execute(p);
    fftwf_destroy_plan(p);
    fftwf_free(in); fftwf_free(out);
}

int main(void){
    printf("Running several FFTs...\n");
    run_fftw_double_1d(1<<12);   // 4096
    run_fftw_double_2d(512,512);
    run_fftw_double_3d(64,64,32);
    run_fftw_float_1d(1<<14);    // 16384
    printf("Done.\n");
    return 0;
}
