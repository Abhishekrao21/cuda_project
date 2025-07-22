/*******************************************************
 * CUDA Parallel Histogram Calculation (Optimised)
 * -----------------------------------------------------
 * 1. 512-thread blocks for higher occupancy.
 * 2. One shared-memory histogram per block
 *    → only 256 global atomics per block.
 * 3. Pinned host buffer + single stream to overlap
 *    host↔device copies with kernel execution.
 * 4. All images are batched in one launch.
 * 5. CSV output is optional: add -s / --save flag.
 *******************************************************/

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <fstream>
#include <numeric>
#include <filesystem>

#define HISTOGRAM_SIZE     256
#define THREADS_PER_BLOCK  512

/* ----- CUDA error macro ------------------------------------------------ */
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)         \
                      << " (" << __FILE__ << ':' << __LINE__ << ")\n";     \
            std::exit(EXIT_FAILURE);                                       \
        }                                                                  \
    } while (0)

/* ----- CUDA kernel: shared-memory reduction ---------------------------- */
__global__
void batchHistogramKernelOpt(const unsigned char *images,
                             unsigned int       *histograms,
                             int width, int height,
                             int numImages)
{
    const int imgIdx = blockIdx.y;
    if (imgIdx >= numImages) return;

    __shared__ unsigned int local[HISTOGRAM_SIZE];

    const int tid    = threadIdx.x;
    const int pixels = width * height;
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

/* =======================================================================
                               Host class
   =======================================================================*/
class HistogramProcessor {
    std::vector<cv::Mat>                    images_;
    std::vector<std::string>                names_;
    std::vector<std::vector<unsigned int>>  hist_;

public:
    /* ---------- load grayscale images from a folder ---------- */
    bool load(const std::string &dir) {
        images_.clear(); names_.clear();
        if (!std::filesystem::exists(dir)) {
            std::cerr << "Directory not found: " << dir << '\n';
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
                    std::cout << "Loaded: " << names_.back() << '\n';
                }
            }
        }
        std::cout << "Total images loaded: " << images_.size() << '\n';
        return !images_.empty();
    }

    /* ---------- GPU histogram (optimised) ---------- */
    void runGPU() {
        if (images_.empty()) { std::cerr<<"No images!\n"; return; }

        const int W = images_[0].cols, H = images_[0].rows;
        for (auto &im : images_)
            if (im.cols!=W || im.rows!=H) cv::resize(im, im, {W,H});

        const int  nImg      = images_.size();
        const int  pix       = W * H;
        const size_t imgB    = 1ULL * nImg * pix;
        const size_t histB   = 1ULL * nImg * HISTOGRAM_SIZE * sizeof(unsigned int);

        unsigned char *hImgPinned = nullptr;
        CUDA_CHECK(cudaHostAlloc(&hImgPinned, imgB, cudaHostAllocDefault));
        for (int i = 0; i < nImg; ++i)
            std::memcpy(hImgPinned + i*pix, images_[i].data, pix);

        unsigned char *dImg = nullptr;
        unsigned int  *dHist= nullptr;
        CUDA_CHECK(cudaMalloc(&dImg,  imgB));
        CUDA_CHECK(cudaMalloc(&dHist, histB));
        CUDA_CHECK(cudaMemset(dHist, 0, histB));

        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));

        auto t0 = std::chrono::high_resolution_clock::now();

        CUDA_CHECK(cudaMemcpyAsync(dImg, hImgPinned, imgB,
                                   cudaMemcpyHostToDevice, stream));

        const int blocks = (pix + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        dim3 grid(blocks, nImg);
        batchHistogramKernelOpt<<<grid, THREADS_PER_BLOCK, 0, stream>>>(
            dImg, dHist, W, H, nImg);

        std::vector<unsigned int> hHist(nImg * HISTOGRAM_SIZE);
        CUDA_CHECK(cudaMemcpyAsync(hHist.data(), dHist, histB,
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        auto t1 = std::chrono::high_resolution_clock::now();
        std::cout << "GPU histogram time: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
                  << " ms\n";

        hist_.assign(nImg, std::vector<unsigned int>(HISTOGRAM_SIZE));
        for (int i = 0; i < nImg; ++i)
            std::copy_n(hHist.data() + i*HISTOGRAM_SIZE, HISTOGRAM_SIZE, hist_[i].begin());

        CUDA_CHECK(cudaFreeHost(hImgPinned));
        CUDA_CHECK(cudaFree(dImg));
        CUDA_CHECK(cudaFree(dHist));
        CUDA_CHECK(cudaStreamDestroy(stream));
    }

    /* ---------- CPU baseline (naïve) ---------- */
    void runCPU() {
        auto t0 = std::chrono::high_resolution_clock::now();
        hist_.assign(images_.size(), std::vector<unsigned int>(HISTOGRAM_SIZE, 0));
        for (size_t i = 0; i < images_.size(); ++i)
            for (int r = 0; r < images_[i].rows; ++r)
                for (int c = 0; c < images_[i].cols; ++c)
                    ++hist_[i][ images_[i].at<uchar>(r, c) ];
        auto t1 = std::chrono::high_resolution_clock::now();
        std::cout << "CPU histogram time: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
                  << " ms\n";
    }

    /* ---------- optional CSV writer ---------- */
    void saveHistograms(const std::string& outDir) {
        if (hist_.empty()) { std::cerr<<"No histograms.\n"; return; }
        std::filesystem::create_directories(outDir);

        /* per-image CSVs */
        for (size_t i = 0; i < hist_.size(); ++i) {
            std::ofstream f(outDir + "/histogram_" + std::to_string(i) +
                            "_" + names_[i] + ".csv");
            f << "Intensity,Count\n";
            for (int j = 0; j < HISTOGRAM_SIZE; ++j)
                f << j << ',' << hist_[i][j] << '\n';
        }

        /* summary CSV */
        std::ofstream s(outDir + "/histogram_summary.csv");
        s << "Image,Mean,Std,Min,Max,Total_Pixels\n";

        for (size_t i = 0; i < hist_.size(); ++i) {
            const auto& h = hist_[i];
            unsigned total = std::accumulate(h.begin(), h.end(), 0U);

            double mean = 0.0;
            for (int j = 0; j < HISTOGRAM_SIZE; ++j) mean += j * h[j];
            mean /= total;

            double var = 0.0;
            for (int j = 0; j < HISTOGRAM_SIZE; ++j)
                var += h[j] * (j - mean) * (j - mean);
            var /= total;

            unsigned minI = 255, maxI = 0;
            for (int j = 0; j < HISTOGRAM_SIZE; ++j)
                if (h[j]) { minI = std::min(minI, (unsigned)j);
                            maxI = std::max(maxI, (unsigned)j); }

            s << names_[i] << ',' << mean << ',' << std::sqrt(var) << ','
              << minI << ',' << maxI << ',' << total << '\n';
        }
        std::cout << "CSV files saved to " << outDir << "/\n";
    }
};

/* =======================================================================
                                   main
   =======================================================================*/
int main(int argc, char* argv[])
{
    std::string dataDir   = "data";
    std::string outDir    = "output";
    bool        saveCSV   = false;

    /* ---------- simple CLI ---------- */
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if ((arg=="-d"||arg=="--data") && i+1<argc)   dataDir = argv[++i];
        else if ((arg=="-o"||arg=="--output")&& i+1<argc) outDir = argv[++i];
        else if (arg=="-s"||arg=="--save")            saveCSV = true;
        else if (arg=="-h"||arg=="--help") {
            std::cout <<
              "Usage: ./histogram_cuda [options]\n"
              "  -d, --data   <dir>   input directory (default: data)\n"
              "  -o, --output <dir>   output directory (default: output)\n"
              "  -s, --save          write CSV outputs\n"
              "  -h, --help          show this help\n";
            return 0;
        }
    }

    std::cout << "CUDA Parallel Histogram Calculation\n";

    HistogramProcessor hp;
    if (!hp.load(dataDir)) return 1;

    hp.runGPU();
    std::cout << "\nCPU baseline for comparison:\n";
    hp.runCPU();

    if (saveCSV) hp.saveHistograms(outDir);
    return 0;
}
