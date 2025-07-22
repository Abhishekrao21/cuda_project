#ifndef HISTOGRAM_UTILS_H
#define HISTOGRAM_UTILS_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in file " << __FILE__ << " at line " << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
}

__global__ void computeHistogramKernel(unsigned char* d_image, int width, int height, int* d_histogram);

void computeHistogramCPU(const unsigned char* image, int width, int height, int* histogram);

std::vector<std::string> getImagePaths(const std::string& folderPath);

void writeHistogramToCSV(const std::string& filename, const int* histogram);

#endif // HISTOGRAM_UTILS_H
