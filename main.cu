#include <iostream>
#include <cuda_runtime.h>

#include "benchmark.h"

int main() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    int runtimeVersion = 0;
    int driverVersion = 0;
    cudaRuntimeGetVersion(&runtimeVersion);
    cudaDriverGetVersion(&driverVersion);
    std::cout << "CUDA Runtime Version: " << runtimeVersion << std::endl;
    std::cout << "CUDA Driver Version: " << driverVersion << std::endl;

    if (error != cudaSuccess) {
        std::cout << "Failed to get device count: " << cudaGetErrorString(error) << std::endl;
        return 1;
    }

    if (deviceCount == 0) {
        std::cout << "\nNo NVIDIA GPU found." << std::endl;
    } else {
        std::cout << "\nFound " << deviceCount << " NVIDIA GPU(s)." << std::endl;

        // Print information about each GPU
        for (int i = 0; i < deviceCount; ++i) {
            cudaDeviceProp deviceProp;
            error = cudaGetDeviceProperties(&deviceProp, i);

            if (error == cudaSuccess) {
                std::cout << "\nGPU " << i << " Information:" << std::endl;
                std::cout << "Name: " << deviceProp.name << std::endl;
                std::cout << "Compute Capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
                std::cout << "Total Global Memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
                std::cout << "Multiprocessor Count: " << deviceProp.multiProcessorCount << std::endl;
            } else {
                std::cout << "Failed to get device properties for GPU " << i << ": " << cudaGetErrorString(error) << std::endl;
            }
        }
    }

    std::pair<float, float> bandwidthValues = measureBandwidth();
    float hostToDeviceBandwidth = bandwidthValues.first;
    float deviceToHostBandwidth = bandwidthValues.second;

    std::cout << "\nHost to Device Bandwidth: " << hostToDeviceBandwidth << " MiB/s" << std::endl;
    std::cout << "Device to Host Bandwidth: " << deviceToHostBandwidth << " MiB/s" << std::endl;

    return 0;
}
