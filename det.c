#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <omp.h>
#include <math.h>

#define MOD 1000000007.0

static inline int matIndex(int i, int j, int n){
    return i*n+j;
}

double det(double *mat, int n){
    double tot_time = 0;
    int n1 = n+1;
    int detSign = 1;
    if(n == 0) return 1;
    if(n == 1) return mat[matIndex(1, 1, n1)];
    double prev = 1, curr;
    for(int k = 1; k < n1; k++){
        curr = mat[matIndex(k, k, n1)];
        
        double start_time = omp_get_wtime();
        #pragma omp parallel for num_threads(8)
        for(int i = k+1; i < n1; i++){
            for(int j = k+1; j < n1; j++){
                mat[matIndex(i, j, n1)] = fmod((curr*mat[matIndex(i, j, n1)] - mat[matIndex(i, k, n1)]*mat[matIndex(k, j, n1)])/prev, MOD);
            }
        }
        tot_time += omp_get_wtime()-start_time;
        prev = curr;
    }
    printf("Time (DET): %.6lf\n", tot_time);
    return mat[matIndex(n, n, n1)]*detSign;
}

int main(){
    double start_time = omp_get_wtime();
    int n;
    double r1, r2;
    scanf("%d", &n);
    int n1 = n + 1;
    double *mat = (double*)calloc((n1)*(n1), sizeof(double));
    mat[0] = 1;
    for(int i = 1; i < n1; i++){
        for(int j = 1; j < n1; j++){
            scanf("%lf", mat+matIndex(i, j, n1));
        }
    }
    r2 = det(mat, n);
    printf("%lf\n", r2);
    free(mat);
    printf("Time (TOT): %.6lf\n", omp_get_wtime()-start_time);
    return 0;
}