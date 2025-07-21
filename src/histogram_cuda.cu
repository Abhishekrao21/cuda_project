#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <fstream>
#include <filesystem>

#define HISTOGRAM_SIZE 256
#define THREADS_PER_BLOCK 256
#define SHARED_MEMORY_BANKS 32

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)

// CUDA kernel for computing histogram of a single image
__global__ void computeHistogramKernel(const unsigned char* image, 
                                     unsigned int* histogram, 
                                     int width, int height) {

    // Shared memory for local histogram accumulation
    __shared__ unsigned int shared_hist[HISTOGRAM_SIZE];

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int global_tid = bid * blockDim.x + tid;

    // Initialize shared memory histogram
    if (tid < HISTOGRAM_SIZE) {
        shared_hist[tid] = 0;
    }
    __syncthreads();

    // Each thread processes multiple pixels to ensure all pixels are covered
    int pixels_per_image = width * height;
    int stride = blockDim.x * gridDim.x;

    for (int i = global_tid; i < pixels_per_image; i += stride) {
        unsigned char pixel_value = image[i];
        atomicAdd(&shared_hist[pixel_value], 1);
    }

    __syncthreads();

    // Reduce shared memory histogram to global memory
    if (tid < HISTOGRAM_SIZE) {
        atomicAdd(&histogram[tid], shared_hist[tid]);
    }
}

// CUDA kernel for batch processing multiple images
__global__ void batchHistogramKernel(const unsigned char* images, 
                                   unsigned int* histograms,
                                   int width, int height, 
                                   int num_images) {

    int image_idx = blockIdx.y;
    if (image_idx >= num_images) return;

    // Shared memory for local histogram accumulation
    __shared__ unsigned int shared_hist[HISTOGRAM_SIZE];

    int tid = threadIdx.x;
    int block_id = blockIdx.x;
    int global_tid = block_id * blockDim.x + tid;

    // Initialize shared memory histogram for this image
    if (tid < HISTOGRAM_SIZE) {
        shared_hist[tid] = 0;
    }
    __syncthreads();

    // Calculate image offset
    int pixels_per_image = width * height;
    const unsigned char* current_image = images + image_idx * pixels_per_image;
    unsigned int* current_histogram = histograms + image_idx * HISTOGRAM_SIZE;

    // Each thread processes multiple pixels
    int stride = blockDim.x * gridDim.x;

    for (int i = global_tid; i < pixels_per_image; i += stride) {
        unsigned char pixel_value = current_image[i];
        atomicAdd(&shared_hist[pixel_value], 1);
    }

    __syncthreads();

    // Reduce shared memory histogram to global memory
    if (tid < HISTOGRAM_SIZE) {
        atomicAdd(&current_histogram[tid], shared_hist[tid]);
    }
}

class HistogramProcessor {
private:
    std::vector<cv::Mat> images;
    std::vector<std::string> image_paths;
    std::vector<std::vector<unsigned int>> histograms;

public:
    bool loadImages(const std::string& directory_path) {
        images.clear();
        image_paths.clear();

        if (!std::filesystem::exists(directory_path)) {
            std::cerr << "Directory does not exist: " << directory_path << std::endl;
            return false;
        }

        for (const auto& entry : std::filesystem::directory_iterator(directory_path)) {
            if (entry.is_regular_file()) {
                std::string ext = entry.path().extension().string();
                if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
                    cv::Mat img = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);
                    if (!img.empty()) {
                        images.push_back(img);
                        image_paths.push_back(entry.path().filename().string());
                        std::cout << "Loaded: " << entry.path().filename().string() << std::endl;
                    }
                }
            }
        }

        std::cout << "Total images loaded: " << images.size() << std::endl;
        return !images.empty();
    }

    void computeHistogramsCPU() {
        auto start = std::chrono::high_resolution_clock::now();

        histograms.clear();
        histograms.resize(images.size());

        for (size_t i = 0; i < images.size(); i++) {
            histograms[i].resize(HISTOGRAM_SIZE, 0);

            cv::Mat& img = images[i];
            for (int y = 0; y < img.rows; y++) {
                for (int x = 0; x < img.cols; x++) {
                    unsigned char pixel = img.at<unsigned char>(y, x);
                    histograms[i][pixel]++;
                }
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "CPU histogram computation time: " << duration.count() << " ms" << std::endl;
    }

    void computeHistogramsGPU() {
        if (images.empty()) {
            std::cerr << "No images loaded!" << std::endl;
            return;
        }

        auto start = std::chrono::high_resolution_clock::now();

        // Assume all images have the same dimensions (resize if needed)
        int width = images[0].cols;
        int height = images[0].rows;
        int num_images = images.size();
        int pixels_per_image = width * height;

        // Resize images if they have different dimensions
        for (auto& img : images) {
            if (img.cols != width || img.rows != height) {
                cv::resize(img, img, cv::Size(width, height));
            }
        }

        // Prepare host data
        std::vector<unsigned char> host_images(num_images * pixels_per_image);
        for (int i = 0; i < num_images; i++) {
            std::memcpy(host_images.data() + i * pixels_per_image, 
                       images[i].data, pixels_per_image);
        }

        // Allocate device memory
        unsigned char* d_images;
        unsigned int* d_histograms;

        CUDA_CHECK(cudaMalloc(&d_images, num_images * pixels_per_image * sizeof(unsigned char)));
        CUDA_CHECK(cudaMalloc(&d_histograms, num_images * HISTOGRAM_SIZE * sizeof(unsigned int)));

        // Initialize histograms to zero
        CUDA_CHECK(cudaMemset(d_histograms, 0, num_images * HISTOGRAM_SIZE * sizeof(unsigned int)));

        // Copy images to device
        CUDA_CHECK(cudaMemcpy(d_images, host_images.data(), 
                             num_images * pixels_per_image * sizeof(unsigned char), 
                             cudaMemcpyHostToDevice));

        // Configure kernel launch parameters
        dim3 blockDim(THREADS_PER_BLOCK);
        dim3 gridDim((pixels_per_image + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, num_images);

        // Launch kernel
        batchHistogramKernel<<<gridDim, blockDim>>>(d_images, d_histograms, width, height, num_images);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy results back to host
        std::vector<unsigned int> host_histograms(num_images * HISTOGRAM_SIZE);
        CUDA_CHECK(cudaMemcpy(host_histograms.data(), d_histograms, 
                             num_images * HISTOGRAM_SIZE * sizeof(unsigned int), 
                             cudaMemcpyDeviceToHost));

        // Convert to vector of vectors
        histograms.clear();
        histograms.resize(num_images);
        for (int i = 0; i < num_images; i++) {
            histograms[i].resize(HISTOGRAM_SIZE);
            for (int j = 0; j < HISTOGRAM_SIZE; j++) {
                histograms[i][j] = host_histograms[i * HISTOGRAM_SIZE + j];
            }
        }

        // Cleanup
        CUDA_CHECK(cudaFree(d_images));
        CUDA_CHECK(cudaFree(d_histograms));

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "GPU histogram computation time: " << duration.count() << " ms" << std::endl;
    }

    void saveHistograms(const std::string& output_dir) {
        if (histograms.empty()) {
            std::cerr << "No histograms computed!" << std::endl;
            return;
        }

        // Save individual histograms
        for (size_t i = 0; i < histograms.size(); i++) {
            std::string filename = output_dir + "/histogram_" + std::to_string(i) + "_" + image_paths[i] + ".csv";
            std::ofstream file(filename);

            file << "Intensity,Count\n";
            for (int j = 0; j < HISTOGRAM_SIZE; j++) {
                file << j << "," << histograms[i][j] << "\n";
            }
            file.close();
        }

        // Save summary statistics
        std::string summary_file = output_dir + "/histogram_summary.csv";
        std::ofstream summary(summary_file);

        summary << "Image,Mean,Std,Min,Max,Total_Pixels\n";

        for (size_t i = 0; i < histograms.size(); i++) {
            double mean = 0.0, variance = 0.0;
            unsigned int total_pixels = 0;
            unsigned int min_intensity = 255, max_intensity = 0;

            // Calculate total pixels
            for (int j = 0; j < HISTOGRAM_SIZE; j++) {
                total_pixels += histograms[i][j];
            }

            // Calculate mean
            for (int j = 0; j < HISTOGRAM_SIZE; j++) {
                mean += j * histograms[i][j];
            }
            mean /= total_pixels;

            // Calculate variance and find min/max intensities
            for (int j = 0; j < HISTOGRAM_SIZE; j++) {
                if (histograms[i][j] > 0) {
                    variance += histograms[i][j] * (j - mean) * (j - mean);
                    if (j < min_intensity) min_intensity = j;
                    if (j > max_intensity) max_intensity = j;
                }
            }
            variance /= total_pixels;
            double std_dev = sqrt(variance);

            summary << image_paths[i] << "," << mean << "," << std_dev << "," 
                   << min_intensity << "," << max_intensity << "," << total_pixels << "\n";
        }

        summary.close();
        std::cout << "Histograms and summary saved to: " << output_dir << std::endl;
    }

    void printStatistics() {
        if (histograms.empty()) {
            std::cerr << "No histograms computed!" << std::endl;
            return;
        }

        std::cout << "\n=== Histogram Statistics ===" << std::endl;
        std::cout << "Number of images processed: " << histograms.size() << std::endl;

        // Calculate aggregate statistics
        std::vector<double> means, std_devs;

        for (size_t i = 0; i < histograms.size(); i++) {
            double mean = 0.0, variance = 0.0;
            unsigned int total_pixels = 0;

            for (int j = 0; j < HISTOGRAM_SIZE; j++) {
                total_pixels += histograms[i][j];
            }

            for (int j = 0; j < HISTOGRAM_SIZE; j++) {
                mean += j * histograms[i][j];
            }
            mean /= total_pixels;
            means.push_back(mean);

            for (int j = 0; j < HISTOGRAM_SIZE; j++) {
                variance += histograms[i][j] * (j - mean) * (j - mean);
            }
            variance /= total_pixels;
            std_devs.push_back(sqrt(variance));

            std::cout << "Image " << i << " (" << image_paths[i] << "): "
                     << "Mean=" << mean << ", Std=" << sqrt(variance) << std::endl;
        }
    }
};

int main(int argc, char* argv[]) {
    std::cout << "CUDA Parallel Histogram Calculation" << std::endl;
    std::cout << "====================================" << std::endl;

    // Check CUDA availability
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return -1;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "Using GPU: " << prop.name << std::endl;
    std::cout << "CUDA Capability: " << prop.major << "." << prop.minor << std::endl;

    std::string data_dir = "data";
    std::string output_dir = "output";

    if (argc >= 2) {
        data_dir = argv[1];
    }
    if (argc >= 3) {
        output_dir = argv[2];
    }

    HistogramProcessor processor;

    // Load images
    if (!processor.loadImages(data_dir)) {
        std::cerr << "Failed to load images from: " << data_dir << std::endl;
        return -1;
    }

    // Compute histograms using GPU
    processor.computeHistogramsGPU();

    // Optionally compute using CPU for comparison
    std::cout << "\nComputing CPU baseline for comparison..." << std::endl;
    processor.computeHistogramsCPU();

    // Print statistics
    processor.printStatistics();

    // Save results
    processor.saveHistograms(output_dir);

    std::cout << "\nProcessing complete!" << std::endl;
    return 0;
}
