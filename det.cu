#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

#define epsilon 1e-9
#define MOD 1000000007.0

__device__ static inline int matIndex(int i, int j, int n){
    return i*n+j;
}

__global__ void parallel(double *mat, int n, int n1, int k){
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    int stride = blockDim.x*gridDim.x;
    for(int i = k+1+idx; i < n1; i += stride){
        for(int j = k+1; j < n1; j++){
            mat[matIndex(i, j, n1)] = fmod((mat[matIndex(k, k, n1)]*mat[matIndex(i, j, n1)] - mat[matIndex(i, k, n1)]*mat[matIndex(k, j, n1)])/mat[matIndex(k-1, k-1, n1)], MOD);
        }
    }
}

void det(double *mat, float *time, int n, double *cpu_mat, unsigned long long size){
    int sms;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    printf("SMS: %d\n", sms);
    int n1 = n+1;
    float elapsed;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for(int k = 1; k < n1; k++){
        cudaEventRecord(start, 0);
        parallel<<<sms*32, 256>>>(mat, n, n1, k); 
        cudaDeviceSynchronize();       
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        *time += elapsed;
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main(){
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    int n, clockRate;
    cudaDeviceGetAttribute(&clockRate, cudaDevAttrClockRate, 0);
    clockRate *= 1e3;
    scanf("%d", &n);
    int n1 = n + 1;

    float time, det_time = 0;
    unsigned long long size = sizeof(double)*n1*n1;
    double *mat = (double*)calloc((n1)*(n1), sizeof(double));
    
    mat[0] = 1;
    for(int i = 1; i < n1; i++){
        for(int j = 1; j < n1; j++){
            scanf("%lf", mat+(i*n1+j));
        }
    }
    
    double *gpu_mat;
    cudaMalloc((void **)&gpu_mat, size);
    cudaMemcpy(gpu_mat, mat, size, cudaMemcpyHostToDevice);

    det(gpu_mat, &det_time, n, mat, size);
    cudaMemcpy(mat, gpu_mat, size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    
    printf("Time (DET): %.12lf\n", det_time/clockRate);
    printf("%lf\n", mat[n*n1+n]);
    printf("Time (Total): %.6lf\n", time/1000);
    free(mat);
    cudaFree(gpu_mat);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventSynchronize(stop);
    
    return 0;
}