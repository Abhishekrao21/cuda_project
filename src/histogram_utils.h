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

#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t e = call;                                                \
        if (e != cudaSuccess) {                                              \
            std::cerr << "CUDA error: " << cudaGetErrorString(e)             \
                      << " (" << __FILE__ << ':' << __LINE__ << ")\n";       \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

constexpr int HISTOGRAM_SIZE = 256;

/* ---------- kernel & helpers ---------- */
__global__
void computeHistogramKernel(const unsigned char* d_img,
                            int width, int height,
                            unsigned int* d_hist);

void  computeHistogramCPU(const cv::Mat& img, unsigned int* hist);
void  writeHistogramCSV (const std::string& file, const unsigned int* hist);
std::vector<std::string> getImagePaths(const std::string& folder);

#endif // HISTOGRAM_UTILS_H
