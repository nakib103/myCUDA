#include <iostream>

/*
kernel              - function that runs on GPU
__global__ keyword  - it tells CUDA C++ compiler that the function
runs in GPU and can be called from CPU code
*/
__global__ void vectorAdd(int N, float* x, float* y){
    for(int i = 0; i < N; i++){
        y[i] = x[i] + y[i];
    }
}

int main(){
    int N = 1<<20;

    //float *x = new float[N];
    //float *y = new float[N];
    /*
    We need to keep the data in memory which is accessible by GPU.
    CUDA provides "Unified Memory" space which is accessible by both CPU and GPU
    cudaMallocManaged       - allocate memory in Unified Memory space
    */
    float *x, *y;
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));

    // initialize x and y vectors on host
    for(int i = 0; i < N; i++){
        x[i] = float(i);
        y[i] = float(i);
    }

    //vectorAdd(N, x, y);
    /* 
    launcing kernel on GPU are done using <<<>>>
    */
    vectorAdd <<<1, 1>>> (N, x, y);

    /*
    cudaDeviceSyncronize        - block CPU execution until kernel execution is done
    */
    cudaDeviceSyncronize();
    
    // verify the operation is successful
    float maxError = 0.0f;
    for(int i = 0; i < N; i++){
        float error = (y[i] - 2*x[i]) > 0 ? y[i] - 2*x[i] : 2*x[i] - y[i];
        maxError = maxError > error ? maxError : error;
    }
    std::cout << "Maximum error " << maxError << std::endl;

    //delete [] x;
    //delete [] y;
    /* 
    cudaFree        - free memory from Unified Memory space
    */
    cudaFree(x);
    cudaFree(y);

    return 0;
}