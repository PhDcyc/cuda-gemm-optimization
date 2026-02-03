#include <stdio.h>
#include <cuda_runtime.h>

#define N 8192  // 在 4090 上跑这个规模才能拉开差距
#define TILE_WIDTH 32
#define RX 8 
#define RY 8 

__global__ void __launch_bounds__(256) 
matrixMulDoubleBuffer(float *A, float *B, float *C, int width) {
    float accum[RY][RX];
    #pragma unroll
    for (int i = 0; i < RY; i++)
        for (int j = 0; j < RX; j++) accum[i][j] = 0.0f;

    float reg_A[RY], reg_B[RX];
    __shared__ float ds_A[2][TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_B[2][TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x; int ty = threadIdx.y;
    int row = blockIdx.y * TILE_WIDTH + ty * RY;
    int col = blockIdx.x * TILE_WIDTH + tx * RX;

    int phase = 0;
    #pragma unroll
    for (int i = 0; i < RY; i++) ds_A[phase][ty * RY + i][tx] = A[(row + i) * width + tx];
    #pragma unroll
    for (int i = 0; i < RX; i++) ds_B[phase][ty][tx * RX + i] = B[ty * width + (col + i)];
    __syncthreads();

    for (int m = 1; m < width / TILE_WIDTH; ++m) {
        int next_phase = 1 - phase;
        #pragma unroll
        for (int i = 0; i < RY; i++) ds_A[next_phase][ty * RY + i][tx] = A[(row + i) * width + (m * TILE_WIDTH + tx)];
        #pragma unroll
        for (int i = 0; i < RX; i++) ds_B[next_phase][ty][tx * RX + i] = B[(m * TILE_WIDTH + ty) * width + (col + i)];

        #pragma unroll
        for (int k = 0; k < TILE_WIDTH; k++) {
            #pragma unroll
            for (int i = 0; i < RY; i++) reg_A[i] = ds_A[phase][ty * RY + i][k];
            #pragma unroll
            for (int i = 0; i < RX; i++) reg_B[i] = ds_B[phase][k][tx * RX + i];
            #pragma unroll
            for (int i = 0; i < RY; i++)
                for (int j = 0; j < RX; j++) accum[i][j] += reg_A[i] * reg_B[j];
        }
        phase = next_phase;
        __syncthreads();
    }
    #pragma unroll
    for (int k = 0; k < TILE_WIDTH; k++) {
        #pragma unroll
        for (int i = 0; i < RY; i++) reg_A[i] = ds_A[phase][ty * RY + i][k];
        #pragma unroll
        for (int i = 0; i < RX; i++) reg_B[i] = ds_B[phase][k][tx * RX + i];
        #pragma unroll
        for (int i = 0; i < RY; i++)
            for (int j = 0; j < RX; j++) accum[i][j] += reg_A[i] * reg_B[j];
    }
    #pragma unroll
    for (int i = 0; i < RY; i++)
        for (int j = 0; j < RX; j++) C[(row + i) * width + (col + j)] = accum[i][j];
}

int main() {
    size_t size = (size_t)N * N * sizeof(float);
    float *h_A = (float*)malloc(size), *h_B = (float*)malloc(size);
    for (int i = 0; i < N * N; i++) { h_A[i] = 1.0f; h_B[i] = 2.0f; }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size); cudaMalloc(&d_B, size); cudaMalloc(&d_C, size);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_WIDTH / RX, TILE_WIDTH / RY); 
    dim3 dimGrid(N / TILE_WIDTH, N / TILE_WIDTH);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    matrixMulDoubleBuffer<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float msec = 0;
    cudaEventElapsedTime(&msec, start, stop);
    printf("N=%d Performance: %.3f GFLOPS\n", N, (2.0 * N * N * N) / (msec * 1e6));

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B);
    return 0;
}