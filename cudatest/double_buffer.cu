#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

// 4090 的核心要在 8192 以上规模才能完全吃饱
#define N 8192 

// --- 核心参数调优 ---
// 块大小扩大到 128x128，这是高端显卡的甜点位
#define BM 128
#define BN 128
#define BK 8   // K 维度步进，配合 float4 刚好
#define TM 8   // 每个线程负责 8x8 的计算 (寄存器压力适中)
#define TN 8

// 错误检查宏
#define CHECK_CUDA(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        printf("Error: %s:%d, reason: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
        exit(1); \
    } \
}

// 核心 Kernel
__global__ void __launch_bounds__(256) 
sgemm_vec4_128x128(float * __restrict__ A, float * __restrict__ B, float * __restrict__ C, int K) {
    // 计算当前线程在 Block 中的行列
    // blockDim.x = (BM * BN) / (TM * TN) = 128*128 / 64 = 256 线程
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y; // 这里的 ty 实际上没用到，是一维线程块
    const int tid = threadIdx.x; // 一维索引

    // 1. 寄存器分配
    // accum 存储 C 的局部结果 (8x8 = 64 个寄存器)
    float accum[TM][TN] = {0.0f};
    
    // reg_A, reg_B 用于从 Shared Memory 加载到寄存器 (用于计算)
    float reg_A[2][TM]; // 双缓冲
    float reg_B[2][TN]; 

    // load_a_reg, load_b_reg 用于从 Global Memory 加载到寄存器 (用于搬运)
    // 每个线程需要搬运 (BM*BK)/256 = 4 个 float = 1 个 float4
    float4 load_a_reg; 
    float4 load_b_reg;

    // 2. 共享内存分配 (Double Buffer)
    // 这里的尺寸是 [2][BK][BM] 和 [2][BK][BN] 为了避免 Bank Conflict 可能会做转置，
    // 但为了代码简洁和 float4 直读，我们直接用 [2][BK][BM]
    __shared__ float As[2][BK][BM];
    __shared__ float Bs[2][BK][BN];

    // --- 预处理：计算加载 Global Memory 的索引 ---
    // A 的加载：每个 Block 需要加载 128行 * 8列
    // 线程视角的加载索引
    // A_row = tid / (BK/4)  -> A 是列主序还是行主序？我们假设标准行主序 A[M][K]
    // 向量化加载：每次加载 float4 (4个float)
    // 每个 Block 加载 BM*BK = 128*8 = 1024 float
    // 256 个线程，每个线程加载 4 个 float (正好 1 个 float4)
    
    // A 的 Global 指针偏移
    // 线程负责 A 的哪一行? load_a_row = tid / (BK/4); 
    // 线程负责 A 的哪一列? load_a_col = (tid % (BK/4)) * 4;
    int load_a_row = tid >> 1; // tid / 2
    int load_a_col = (tid & 1) << 2; // (tid % 2) * 4
    
    // B 的 Global 指针偏移
    // B 是 KxN。加载 BK行 * BN列 = 8 * 128
    // load_b_row = tid / (BN/4);
    // load_b_col = (tid % (BN/4)) * 4;
    int load_b_row = tid >> 5; // tid / 32
    int load_b_col = (tid & 31) << 2; 

    // 初始指针
    const float* A_ptr = A + (by * BM + load_a_row) * K + load_a_col;
    const float* B_ptr = B + load_b_row * K /*这里 B 应当是 K*N，假设 B 已知 N*/ + (bx * BN + load_b_col);
    // 修正：B 的索引计算。如果 B 是 KxN (Row Major)
    // 我们要加载 B[row][col]。row 从 0..7, col 从 0..127.
    // B_ptr 基础位置：B + (load_b_row) * N + (bx * BN + load_b_col);
    // 但是这里 K 是动态的，我们假设 K=N
    const float* A_src = A + (by * BM + load_a_row) * N + load_a_col;
    const float* B_src = B + (load_b_row) * N + (bx * BN + load_b_col);

    // --- 3. Prologue: 加载第一个 Tile 到 Shared Memory ---
    // 必须用 reinterpret_cast 强转为 float4* 进行加载
    load_a_reg = reinterpret_cast<const float4*>(A_src)[0]; 
    load_b_reg = reinterpret_cast<const float4*>(B_src)[0];

    // 写入 Shared Memory (这步要把 float4 拆开或者是直接写)
    // As[0][col][row] -> 转置存储以便后续计算时利用 LDS.128 ?
    // 为了简单且高效，我们直接存。计算时再看。
    // As [load_a_col][load_a_row] ? 
    // 我们计算时需要 A 的一行 和 B 的一列。
    // 优化：A 转置存，B 不转置。
    As[0][load_a_col + 0][load_a_row] = load_a_reg.x;
    As[0][load_a_col + 1][load_a_row] = load_a_reg.y;
    As[0][load_a_col + 2][load_a_row] = load_a_reg.z;
    As[0][load_a_col + 3][load_a_row] = load_a_reg.w;
    
    // B 也转置存？ Bs[load_b_row][load_b_col]
    reinterpret_cast<float4*>(&Bs[0][load_b_row][load_b_col])[0] = load_b_reg;

    __syncthreads();

    // --- 4. Main Loop ---
    int write_stage_idx = 1;
    int load_stage_idx = 0;
    
    // 这里的 tid 用于计算，和加载的 tid 含义不同
    // 计算线程索引：ty = tid / (BN/TN) = tid / 16
    // tx = tid % 16
    int thread_row = (tid / 16) * TM;
    int thread_col = (tid % 16) * TN;

    // 预加载第一波寄存器 (Double Buffer on Register level)
    #pragma unroll
    for(int i=0; i<TM; i+=4) {
        float4 tmp = reinterpret_cast<float4*>(&As[0][0][thread_row + i])[0];
        reg_A[0][i] = tmp.x; reg_A[0][i+1] = tmp.y; reg_A[0][i+2] = tmp.z; reg_A[0][i+3] = tmp.w;
    }
    #pragma unroll
    for(int i=0; i<TN; i+=4) {
        float4 tmp = reinterpret_cast<float4*>(&Bs[0][0][thread_col + i])[0];
        reg_B[0][i] = tmp.x; reg_B[0][i+1] = tmp.y; reg_B[0][i+2] = tmp.z; reg_B[0][i+3] = tmp.w;
    }

    int loop_k_limit = N / BK;
    
    for (int k = 1; k < loop_k_limit; k++) {
        // --- Pipeline Stage 1: Load Global to Register (Next Tile) ---
        // 移动指针
        A_src += BK; // A 向右移 8
        B_src += BK * N; // B 向下移 8 行
        
        load_a_reg = reinterpret_cast<const float4*>(A_src)[0];
        load_b_reg = reinterpret_cast<const float4*>(B_src)[0];

        // --- Pipeline Stage 2: Compute (Current Tile) ---
        // 这里的 loop 是 BK 维度 (0..7)
        #pragma unroll
        for (int dot_idx = 0; dot_idx < BK; ++dot_idx) {
            // 加载下一轮计算需要的寄存器 (Ping-Pong)
            int next_dot = (dot_idx + 1) & 7; // % 8
            int load_buf = (dot_idx + 1) & 1; // 这里的 buffer 是指寄存器层面的 buffer 吗？
            // 简化：为了代码极简，我们不做寄存器级双缓冲，只做 Shared Memory 双缓冲
            // 但为了性能，必须预取。
            
            // 我们直接暴力计算当前 dot_idx
            #pragma unroll
            for (int m = 0; m < TM; ++m) {
                #pragma unroll
                for (int n = 0; n < TN; ++n) {
                    accum[m][n] += reg_A[0][m] * reg_B[0][n];
                }
            }
            
            // 如果不是最后一次迭代，加载下一个 dot_idx 的数据到寄存器
            if (dot_idx < BK - 1) {
                #pragma unroll
                for(int i=0; i<TM; i+=4) {
                     float4 tmp = reinterpret_cast<float4*>(&As[load_stage_idx][dot_idx+1][thread_row + i])[0];
                     reg_A[0][i] = tmp.x; reg_A[0][i+1] = tmp.y; reg_A[0][i+2] = tmp.z; reg_A[0][i+3] = tmp.w;
                }
                #pragma unroll
                for(int i=0; i<TN; i+=4) {
                     float4 tmp = reinterpret_cast<float4*>(&Bs[load_stage_idx][dot_idx+1][thread_col + i])[0];
                     reg_B[0][i] = tmp.x; reg_B[0][i+1] = tmp.y; reg_B[0][i+2] = tmp.z; reg_B[0][i+3] = tmp.w;
                }
            }
        }
        
        // --- Pipeline Stage 3: Store Register to Shared (Next Tile) ---
        // 写入下一块 Shared Memory Buffer
        As[write_stage_idx][load_a_col + 0][load_a_row] = load_a_reg.x;
        As[write_stage_idx][load_a_col + 1][load_a_row] = load_a_reg.y;
        As[write_stage_idx][load_a_col + 2][load_a_row] = load_a_reg.z;
        As[write_stage_idx][load_a_col + 3][load_a_row] = load_a_reg.w;
        
        reinterpret_cast<float4*>(&Bs[write_stage_idx][load_b_row][load_b_col])[0] = load_b_reg;
        
        __syncthreads();
        
        // 切换 Buffer 指针
        write_stage_idx = 1 - write_stage_idx;
        load_stage_idx = 1 - load_stage_idx;

        // 加载新 Tile 的第 0 个 dot_idx 数据到寄存器，为下一次大循环做准备
        #pragma unroll
        for(int i=0; i<TM; i+=4) {
             float4 tmp = reinterpret_cast<float4*>(&As[load_stage_idx][0][thread_row + i])[0];
             reg_A[0][i] = tmp.x; reg_A[0][i+1] = tmp.y; reg_A[0][i+2] = tmp.z; reg_A[0][i+3] = tmp.w;
        }
        #pragma unroll
        for(int i=0; i<TN; i+=4) {
             float4 tmp = reinterpret_cast<float4*>(&Bs[load_stage_idx][0][thread_col + i])[0];
             reg_B[0][i] = tmp.x; reg_B[0][i+1] = tmp.y; reg_B[0][i+2] = tmp.z; reg_B[0][i+3] = tmp.w;
        }
    }

    // --- Epilogue: 计算最后一个 Tile (不需要再加载 Global) ---
    #pragma unroll
    for (int dot_idx = 0; dot_idx < BK; ++dot_idx) {
        #pragma unroll
        for (int m = 0; m < TM; ++m) {
            #pragma unroll
            for (int n = 0; n < TN; ++n) {
                accum[m][n] += reg_A[0][m] * reg_B[0][n];
            }
        }
        if (dot_idx < BK - 1) {
            #pragma unroll
            for(int i=0; i<TM; i+=4) {
                 float4 tmp = reinterpret_cast<float4*>(&As[load_stage_idx][dot_idx+1][thread_row + i])[0];
                 reg_A[0][i] = tmp.x; reg_A[0][i+1] = tmp.y; reg_A[0][i+2] = tmp.z; reg_A[0][i+3] = tmp.w;
            }
            #pragma unroll
            for(int i=0; i<TN; i+=4) {
                 float4 tmp = reinterpret_cast<float4*>(&Bs[load_stage_idx][dot_idx+1][thread_col + i])[0];
                 reg_B[0][i] = tmp.x; reg_B[0][i+1] = tmp.y; reg_B[0][i+2] = tmp.z; reg_B[0][i+3] = tmp.w;
            }
        }
    }

    // --- 5. 写回 Global Memory ---
    // C[row][col] = accum
    // 每个线程负责 8x8 区域
    int c_row = by * BM + thread_row;
    int c_col = bx * BN + thread_col;
    
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; j+=4) { // 向量化写回
             float4 tmp;
             tmp.x = accum[i][j]; tmp.y = accum[i][j+1]; tmp.z = accum[i][j+2]; tmp.w = accum[i][j+3];
             reinterpret_cast<float4*>(&C[(c_row + i) * N + (c_col + j)])[0] = tmp;
        }
    }
}

int main() {
    size_t size = (size_t)N * N * sizeof(float);
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    // 初始化
    for (int i = 0; i < N * N; i++) { h_A[i] = 1.0f * (rand()%10); h_B[i] = 1.0f * (rand()%10); }

    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, size));
    CHECK_CUDA(cudaMalloc(&d_B, size));
    CHECK_CUDA(cudaMalloc(&d_C, size));
    
    CHECK_CUDA(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Block = 256 threads (1D)
    dim3 dimBlock(256); 
    // Grid = N/128, N/128
    dim3 dimGrid(N / BM, N / BN);

    // Warmup
    sgemm_vec4_128x128<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    
    // Benchmark
    cudaEventRecord(start);
    int iter = 5; // 跑5次取平均，稳一点
    for(int i=0; i<iter; i++)
        sgemm_vec4_128x128<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float msec = 0;
    cudaEventElapsedTime(&msec, start, stop);
    msec /= iter; // 平均时间
    
    double gflops = (2.0 * N * N * N) / (msec * 1e6);
    printf("N=%d [float4 + 128x128 Tile] Performance: %.3f GFLOPS\n", N, gflops);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B);
    return 0;
}