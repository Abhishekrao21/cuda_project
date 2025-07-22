#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <fstream>
#include <numeric>
#include <cstdio>
#include <filesystem>
#include <iomanip>

#define HISTOGRAM_SIZE       256
#define THREADS_PER_BLOCK    512

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)         \
                      << " (" << __FILE__ << ':' << __LINE__ << ")\n";     \
            std::exit(EXIT_FAILURE);                                       \
        }                                                                  \
    } while (0)

class Logger {
    std::ofstream file_;
public:
    explicit Logger(const std::string& path) {
        file_.open(path, std::ios::out | std::ios::app);
    }
    template<typename T>
    void log(const T& msg) {
        auto now  = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss; ss << std::put_time(std::localtime(&time), "%F %T");
        std::string line = "[" + ss.str() + "] " + msg;
        std::cout << line << '\n';
        if (file_.is_open()) file_ << line << '\n';
    }
};

__global__
void batchHistogramKernelOpt(const unsigned char *images,
                             unsigned int       *histograms,
                             int W, int H,
                             int nImg)
{
    const int imgIdx = blockIdx.y;
    if (imgIdx >= nImg) return;

    __shared__ unsigned int local[HISTOGRAM_SIZE];

    const int tid    = threadIdx.x;
    const int pixels = W * H;
    const int stride = gridDim.x * blockDim.x;
    const int gid    = blockIdx.x * blockDim.x + tid;

    if (tid < HISTOGRAM_SIZE) local[tid] = 0;
    __syncthreads();

    const unsigned char *img = images + imgIdx * pixels;
    for (int i = gid; i < pixels; i += stride)
        atomicAdd(&local[img[i]], 1U);

    __syncthreads();

    if (tid < HISTOGRAM_SIZE)
        atomicAdd(&histograms[imgIdx * HISTOGRAM_SIZE + tid], local[tid]);
}

class HistogramProcessor {
    std::vector<cv::Mat>                    images_;
    std::vector<std::string>                names_;
    std::vector<std::vector<unsigned int>>  hist_;
    Logger                                   *logger_;
public:
    explicit HistogramProcessor(Logger *lg) : logger_(lg) {}

    bool load(const std::string &dir) {
        images_.clear(); names_.clear();
        if (!std::filesystem::exists(dir)) {
            logger_->log("Directory not found: " + dir);
            return false;
        }
        for (auto &e : std::filesystem::directory_iterator(dir)) {
            if (!e.is_regular_file()) continue;
            auto ext = e.path().extension().string();
            if (ext==".png"||ext==".jpg"||ext==".bmp"||ext==".jpeg") {
                cv::Mat img = cv::imread(e.path().string(), cv::IMREAD_GRAYSCALE);
                if (!img.empty()) {
                    images_.push_back(img);
                    names_.push_back(e.path().filename().string());
                    logger_->log("Loaded: " + names_.back() + "  (" +
                                 std::to_string(img.cols) + "x" +
                                 std::to_string(img.rows) + ')');
                }
            }
        }
        logger_->log("Total images loaded: " + std::to_string(images_.size()));
        return !images_.empty();
    }

    void runGPU() {
        if (images_.empty()) return;
        const int W   = images_[0].cols;
        const int H   = images_[0].rows;
        const int n   = images_.size();
        const int pix = W * H;

        for (auto &im : images_)
            if (im.cols!=W || im.rows!=H) cv::resize(im, im, {W,H});

        const size_t imgBytes  = 1ULL * n * pix;
        const size_t histBytes = 1ULL * n * HISTOGRAM_SIZE * sizeof(unsigned int);

        unsigned char *hPinned = nullptr;
        CUDA_CHECK(cudaHostAlloc(&hPinned, imgBytes, cudaHostAllocDefault));
        for (int i = 0; i < n; ++i)
            std::memcpy(hPinned + i*pix, images_[i].data, pix);

        unsigned char *dImg  = nullptr;
        unsigned int  *dHist = nullptr;
        CUDA_CHECK(cudaMalloc(&dImg,  imgBytes));
        CUDA_CHECK(cudaMalloc(&dHist, histBytes));
        CUDA_CHECK(cudaMemset(dHist, 0, histBytes));

        cudaStream_t stream; CUDA_CHECK(cudaStreamCreate(&stream));

        auto t0 = std::chrono::high_resolution_clock::now();
        CUDA_CHECK(cudaMemcpyAsync(dImg, hPinned, imgBytes,
                                   cudaMemcpyHostToDevice, stream));

        const int blocks = (pix + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        dim3 grid(blocks, n);
        batchHistogramKernelOpt<<<grid, THREADS_PER_BLOCK, 0, stream>>>(
            dImg, dHist, W, H, n);

        std::vector<unsigned int> hHist(n*HISTOGRAM_SIZE);
        CUDA_CHECK(cudaMemcpyAsync(hHist.data(), dHist, histBytes,
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        auto t1 = std::chrono::high_resolution_clock::now();

        logger_->log("GPU kernel blocks: " + std::to_string(blocks) +
                     "  threads/block: " + std::to_string(THREADS_PER_BLOCK));
        logger_->log("GPU time: " +
                     std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count())
                     + " ms");

        hist_.assign(n, std::vector<unsigned int>(HISTOGRAM_SIZE));
        for (int i=0;i<n;++i)
            std::copy_n(hHist.data()+i*HISTOGRAM_SIZE, HISTOGRAM_SIZE, hist_[i].begin());

        CUDA_CHECK(cudaFreeHost(hPinned));
        CUDA_CHECK(cudaFree(dImg));
        CUDA_CHECK(cudaFree(dHist));
        CUDA_CHECK(cudaStreamDestroy(stream));
    }

    void runCPU() {
        auto t0 = std::chrono::high_resolution_clock::now();
        hist_.assign(images_.size(), std::vector<unsigned int>(HISTOGRAM_SIZE,0));
        for (size_t i=0;i<images_.size();++i)
            for (int r=0;r<images_[i].rows;++r)
                for (int c=0;c<images_[i].cols;++c)
                    ++hist_[i][ images_[i].at<uchar>(r,c) ];
        auto t1 = std::chrono::high_resolution_clock::now();
        logger_->log("CPU time: " +
            std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count()) + " ms");
    }

    void saveHistograms(const std::string &outDir) {
        if (hist_.empty()) return;
        std::filesystem::create_directories(outDir);
        for (size_t i=0;i<hist_.size();++i) {
            std::ofstream f(outDir + "/histogram_" + std::to_string(i) +
                            "_" + names_[i] + ".csv");
            f << "Intensity,Count\n";
            for (int j=0;j<HISTOGRAM_SIZE;++j) f << j << ',' << hist_[i][j] << '\n';
        }
        logger_->log("Histogram CSV files written to " + outDir + '/');
    }
};

int main(int argc, char* argv[])
{
    std::string dataDir = "data", outDir = "output";
    bool saveCSV = false;

    for (int i=1;i<argc;++i){
        std::string a=argv[i];
        if ((a=="-d"||a=="--data")   && i+1<argc) dataDir=argv[++i];
        else if((a=="-o"||a=="--output")&& i+1<argc) outDir = argv[++i];
        else if(a=="-s"||a=="--save") saveCSV=true;
        else if(a=="-h"||a=="--help"){
            std::puts("Usage: ./histogram_cuda [options]\n"
                      "  -d, --data   DIR   input directory (default: data)\n"
                      "  -o, --output DIR  output directory (default: output)\n"
                      "  -s, --save        write histogram CSVs\n"
                      "  -h, --help        show this message");
            return 0;
        }
    }

    std::filesystem::create_directories(outDir);
    Logger logger(outDir + "/execution_log.txt");
    logger.log("=== CUDA Parallel Histogram Calculation ===");

    HistogramProcessor hp(&logger);
    if(!hp.load(dataDir)) return 1;

    hp.runGPU();
    hp.runCPU();
    if(saveCSV) hp.saveHistograms(outDir);

    logger.log("Processing complete.");
    return 0;
}


