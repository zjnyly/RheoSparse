#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <random>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <string>

using namespace std;

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if(err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)


#define WARP_SIZE 32
#define GROUP_SIZE 128 // 线程块大小
#define IC_PACK_SIZE 64 // IC 维度上的分块大小，对应 x 向量加载的粒度

__device__ __host__ inline int make_divisible(int c, int divisor) {
    return (c + divisor - 1) / divisor;
}

__global__ void spmv_kernel_g128_fp16(
    __half *activations, // x vector (input)
    const __half *packed_weight, // Sparse A matrix (FP16 weights)
    const uint64_t *bitmask,
    float *outputs, // y vector (output)
    const int IC, // Input Channels (K dimension)
    const int OC,
    // , // Output Channels (M dimension)
    const int SPARSITY)
{
    const int oc_idx = blockIdx.y * GROUP_SIZE + threadIdx.x;

    if (oc_idx >= OC) return;

    const int packed_ic_blocks = make_divisible(IC, IC_PACK_SIZE);
    const int ic_block_idx = blockIdx.x;

    extern __shared__ float smem_x[64];
    __half *activations_block_ptr = activations + ic_block_idx * IC_PACK_SIZE;

    const unsigned int warp_id = threadIdx.x / WARP_SIZE;
    const unsigned int lane_id = threadIdx.x % WARP_SIZE;

    if (warp_id <= 1) {
        int idx = warp_id * 32 + lane_id;
        smem_x[idx] = __half2float(activations_block_ptr[idx]);
    }
        
    __syncthreads();

    const int chunkIdx = blockIdx.y * gridDim.x + blockIdx.x;

    const int bitmask_base_offset = chunkIdx * GROUP_SIZE + threadIdx.x;
    uint64_t local_bitmask = bitmask[bitmask_base_offset];
    float psum_local_to_thread = 0.0f;
    
    // const int weightChunkSize = GROUP_SIZE * 32 / 8; // 32 is per row element, 8 is packed data
    // const int weight_base_offset = chunkIdx * weightChunkSize;
    // const float4 *local_packed_weight = ((float4*)packed_weight + weight_base_offset);

    const int weightChunkSize = GROUP_SIZE * SPARSITY; // 32 is per row element, 8 is packed data
    const int weight_base_offset = chunkIdx * weightChunkSize;
    const __half *local_packed_weight = packed_weight + weight_base_offset;

    // if(SPARSITY >> 5 & 1){
    //     const float4 * local_packed_weight_32 = (float4 *)(local_packed_weight + threadIdx.x * 32);
    //     for (int i = 0; i < 4; i++) {
    //         __half warp_packed_weight[8];
    //         *((float4*)warp_packed_weight) = *(local_packed_weight_32 + i);
    //         for (int j = 0; j < 8; j++) {
    //             int pos = __ffsll(local_bitmask) - 1;
    //             local_bitmask &= (local_bitmask - 1); 
    //             psum_local_to_thread += smem_x[pos] * __half2float(warp_packed_weight[j]);
    //         }
    //     }
    //     local_packed_weight += 32 * GROUP_SIZE;
    // }
    
    // if(SPARSITY >> 4 & 1){
    //     const float4 * local_packed_weight_16 = (float4 *)(local_packed_weight + threadIdx.x * 16);
    //     for (int i = 0; i < 2; i++) {
    //         __half warp_packed_weight[8];
    //         *((float4*)warp_packed_weight) = *(local_packed_weight_16 + i);
    //         for (int j = 0; j < 8; j++) {
    //             int pos = __ffsll(local_bitmask) - 1;
    //             local_bitmask &= (local_bitmask - 1); 
    //             psum_local_to_thread += smem_x[pos] * __half2float(warp_packed_weight[j]);
    //         }
    //     }
    //     local_packed_weight += 16 * GROUP_SIZE;
    // }
    
    // if(SPARSITY >> 3 & 1){
    //     const float4 * local_packed_weight_8 = (float4 *)(local_packed_weight + threadIdx.x * 8);
    //     __half warp_packed_weight[8];
    //     *((float4*)warp_packed_weight) = *(local_packed_weight_8);
    //     for (int j = 0; j < 8; j++) {
    //         int pos = __ffsll(local_bitmask) - 1;
    //         local_bitmask &= (local_bitmask - 1); 
    //         psum_local_to_thread += smem_x[pos] * __half2float(warp_packed_weight[j]);
    //     }
    //     local_packed_weight += 8 * GROUP_SIZE;
    // }
    
    // if(SPARSITY >> 2 & 1){
    //     const float2 * local_packed_weight_4 = (float2 *)(local_packed_weight + threadIdx.x * 4);
    //     __half warp_packed_weight[4];
    //     *((float2*)warp_packed_weight) = *(local_packed_weight_4);
    //     for (int j = 0; j < 4; j++) {
    //         int pos = __ffsll(local_bitmask) - 1;
    //         local_bitmask &= (local_bitmask - 1); 
    //         psum_local_to_thread += smem_x[pos] * __half2float(warp_packed_weight[j]);
    //     }
    //     local_packed_weight += 4 * GROUP_SIZE;
    // }
    
    // if(SPARSITY >> 1 & 1){
    //     const float * local_packed_weight_2 = (float *)(local_packed_weight + threadIdx.x * 2);
    //     __half warp_packed_weight[2];
    //     *((float*)warp_packed_weight) = *(local_packed_weight_2);
    //     for (int j = 0; j < 2; j++) {
    //         int pos = __ffsll(local_bitmask) - 1;
    //         local_bitmask &= (local_bitmask - 1); 
    //         psum_local_to_thread += smem_x[pos] * __half2float(warp_packed_weight[j]);
    //     }
    //     local_packed_weight += 2 * GROUP_SIZE;
    // }
    
    // if(SPARSITY & 1){
    //     __half warp_packed_weight = *(local_packed_weight + threadIdx.x);
    //     int pos = __ffsll(local_bitmask) - 1;
    //     local_bitmask &= (local_bitmask - 1); 
    //     psum_local_to_thread += smem_x[pos] * __half2float(warp_packed_weight);
    //     local_packed_weight += 1 * GROUP_SIZE;
    // }
    

    if(SPARSITY >> 5 & 1){
        const __half * local_packed_weight_32 = local_packed_weight + threadIdx.x * 32;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 8; j++) {
                int pos = __ffsll(local_bitmask) - 1;
                local_bitmask &= (local_bitmask - 1); 
                psum_local_to_thread += smem_x[pos] * __half2float(local_packed_weight_32[i * 8 + j]);
            }
        }
        local_packed_weight += 32 * GROUP_SIZE;
    }
    
    if(SPARSITY >> 4 & 1){
        const __half * local_packed_weight_16 = local_packed_weight + threadIdx.x * 16;
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 8; j++) {
                int pos = __ffsll(local_bitmask) - 1;
                local_bitmask &= (local_bitmask - 1); 
                psum_local_to_thread += smem_x[pos] * __half2float(local_packed_weight_16[i * 8 + j]);
            }
        }
        local_packed_weight += 16 * GROUP_SIZE;
    }
    
    if(SPARSITY >> 3 & 1){
        const __half * local_packed_weight_8 = local_packed_weight + threadIdx.x * 8;
        for (int j = 0; j < 8; j++) {
            int pos = __ffsll(local_bitmask) - 1;
            local_bitmask &= (local_bitmask - 1); 
            psum_local_to_thread += smem_x[pos] * __half2float(local_packed_weight_8[j]);
        }
        local_packed_weight += 8 * GROUP_SIZE;
    }
    
    if(SPARSITY >> 2 & 1){
        const __half * local_packed_weight_4 = local_packed_weight + threadIdx.x * 4;
        for (int j = 0; j < 4; j++) {
            int pos = __ffsll(local_bitmask) - 1;
            local_bitmask &= (local_bitmask - 1); 
            psum_local_to_thread += smem_x[pos] * __half2float(local_packed_weight_4[j]);
        }
        local_packed_weight += 4 * GROUP_SIZE;
    }
    
    if(SPARSITY >> 1 & 1){
        const __half * local_packed_weight_2 = local_packed_weight + threadIdx.x * 2;
        for (int j = 0; j < 2; j++) {
            int pos = __ffsll(local_bitmask) - 1;
            local_bitmask &= (local_bitmask - 1); 
            psum_local_to_thread += smem_x[pos] * __half2float(local_packed_weight_2[j]);
        }
        local_packed_weight += 2 * GROUP_SIZE;
    }
    
    if(SPARSITY & 1){
        __half warp_packed_weight = *(local_packed_weight + threadIdx.x);
        int pos = __ffsll(local_bitmask) - 1;
        local_bitmask &= (local_bitmask - 1); 
        psum_local_to_thread += smem_x[pos] * __half2float(warp_packed_weight);
        local_packed_weight += 1 * GROUP_SIZE;
    }
    
    atomicAdd(&outputs[oc_idx], psum_local_to_thread);
}




int main() {
    const int OC = 4096;
    const int IC = 4096;
    const int total_packed = OC * IC;

    std::vector<half> h_input(IC, __float2half(1.0f));
    for (int i = 0; i < IC; ++i) {
    h_input[i] = __float2half(static_cast<float>(rand()) / RAND_MAX);
}
    std::vector<half> h_weights(total_packed, 0.1);
    std::vector<uint64_t> h_bitmask(IC * OC / 64);
    std::vector<float> h_output_gpu(OC, 0.0f);
    std::vector<float> h_output_cpu(OC, 0.0f);

    std::mt19937 rng(42);
    for (int row = 0; row < OC; ++row) {
        for (int blk = 0; blk < 64; ++blk) {
            std::vector<int> positions;
            std::uniform_int_distribution<int> dist_blk(0, 63);
            while (positions.size() < 32) {
                int val = dist_blk(rng);
                if (std::find(positions.begin(), positions.end(), val) == positions.end()) {
                    positions.push_back(val);
                }
            }
            std::sort(positions.begin(), positions.end());
            uint64_t mask = 0;
            for (int p : positions) mask |= (1ULL << p);
            h_bitmask[row * 64 + blk] = mask;
        }
    }

    half *d_input;
    half *d_weights;
    uint64_t *d_bitmask;
    float *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, IC * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_weights, total_packed * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_bitmask, IC * OC / 64 * sizeof(uint64_t)));
    CHECK_CUDA(cudaMalloc(&d_output, OC * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), IC * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_weights, h_weights.data(), total_packed * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_bitmask, h_bitmask.data(), IC * OC / 64 * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_output, 0, OC * sizeof(float)));

    dim3 blockDim(GROUP_SIZE);
    dim3 gridDim(IC / IC_PACK_SIZE, (OC + blockDim.x - 1) / blockDim.x);
    size_t sharedMem = IC_PACK_SIZE * sizeof(float);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < 1000; i++) {
        spmv_kernel_g128_fp16<<<gridDim, blockDim, sharedMem>>>(
            d_input, d_weights, d_bitmask, d_output, IC, OC, 32
        );
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds_gpu = 0.0f;
    cudaEventElapsedTime(&milliseconds_gpu, start, stop);
    milliseconds_gpu /= 1000;
    CHECK_CUDA(cudaMemcpy(h_output_gpu.data(), d_output, OC * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "GPU Kernel execution time: " << milliseconds_gpu << " ms" << std::endl;
    std::cout << "GPU Sample output[0]: " << h_output_gpu[0] << std::endl;
    bool correct = true;
    for (int i = 0; i < OC; i++) {
        if (std::abs(h_output_gpu[i] - h_output_cpu[i]) > 1e-3) {
            correct = false;
            break;
        }
    }
    std::cout << "Results match: " << (correct ? "Yes" : "No") << std::endl;

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_bitmask);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

