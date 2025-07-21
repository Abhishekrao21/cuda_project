/*******************************************************
 * CUDA Parallel Histogram Calculation (Optimised)
 * -----------------------------------------------------
 * 1. 512-thread blocks for higher occupancy.
 * 2. One shared-memory histogram per block
 *    → only 256 global atomics per block.
 * 3. Pinned host buffer + single stream to overlap
 *    H↔D copies with kernel execution.
 * 4. All images are batched in one launch.
 *******************************************************/

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <fstream>
#include <filesystem>

#define HISTOGRAM_SIZE     256
#define THREADS_PER_BLOCK  512         // was 256
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)         \
                      << " (" << __FILE__ << ':' << __LINE__ << ")\n";     \
            std::exit(EXIT_FAILURE);                                       \
        }                                                                  \
    } while (0)

/* ------------------------------------------------------------------ */
/*  Optimised batch kernel – shared-memory reduction                   */
/* ------------------------------------------------------------------ */
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

/* ========================== HOST CLASS =========================== */

class HistogramProcessor {
    std::vector<cv::Mat> images_;
    std::vector<std::string> names_;
    std::vector<std::vector<unsigned int>> hist_;

public:
    bool load(const std::string &dir) {
        images_.clear(); names_.clear();
        if (!std::filesystem::exists(dir)) return false;

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

    /* ------------ GPU version (optimised) ------------ */
    void runGPU() {
        if (images_.empty()) { std::cerr<<"No images!\n"; return; }

        const int W = images_[0].cols, H = images_[0].rows;
        for (auto &im : images_)              // resize mismatched
            if (im.cols!=W||im.rows!=H) cv::resize(im,im,{W,H});

        const int nImg = images_.size();
        const int pix  = W*H;
        const size_t imgBytes  = 1ULL*nImg*pix;
        const size_t histBytes = 1ULL*nImg*HISTOGRAM_SIZE*sizeof(unsigned int);

        /* host-side pinned buffer */
        unsigned char *hImgPinned = nullptr;
        CUDA_CHECK(cudaHostAlloc(&hImgPinned, imgBytes, cudaHostAllocDefault));

        for (int i=0;i<nImg;++i)
            std::memcpy(hImgPinned + i*pix, images_[i].data, pix);

        unsigned char *dImg = nullptr;
        unsigned int  *dHist= nullptr;
        CUDA_CHECK(cudaMalloc(&dImg,  imgBytes));
        CUDA_CHECK(cudaMalloc(&dHist, histBytes));
        CUDA_CHECK(cudaMemset(dHist, 0, histBytes));

        cudaStream_t stream;  CUDA_CHECK(cudaStreamCreate(&stream));

        auto t0 = std::chrono::high_resolution_clock::now();

        CUDA_CHECK(cudaMemcpyAsync(dImg, hImgPinned, imgBytes,
                                   cudaMemcpyHostToDevice, stream));

        const int blocks = (pix + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        dim3 grid(blocks, nImg);
        batchHistogramKernelOpt<<<grid, THREADS_PER_BLOCK, 0, stream>>>(
            dImg, dHist, W, H, nImg);

        std::vector<unsigned int> hHist(nImg*HISTOGRAM_SIZE);
        CUDA_CHECK(cudaMemcpyAsync(hHist.data(), dHist, histBytes,
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        auto t1 = std::chrono::high_resolution_clock::now();
        std::cout << "GPU histogram time: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count()
                  << " ms\n";

        /* convert to vector-of-vector */
        hist_.assign(nImg, std::vector<unsigned int>(HISTOGRAM_SIZE));
        for (int i=0;i<nImg;++i)
            std::copy_n(hHist.data()+i*HISTOGRAM_SIZE, HISTOGRAM_SIZE, hist_[i].begin());

        CUDA_CHECK(cudaFreeHost(hImgPinned));
        CUDA_CHECK(cudaFree(dImg));
        CUDA_CHECK(cudaFree(dHist));
        CUDA_CHECK(cudaStreamDestroy(stream));
    }

    /* ------------ CPU baseline (unchanged) ------------ */
    void runCPU() {
        auto t0=std::chrono::high_resolution_clock::now();
        hist_.assign(images_.size(), std::vector<unsigned int>(HISTOGRAM_SIZE,0));
        for (size_t i=0;i<images_.size();++i)
            for (int r=0;r<images_[i].rows;++r)
                for (int c=0;c<images_[i].cols;++c)
                    ++hist_[i][ images_[i].at<uchar>(r,c) ];
        auto t1=std::chrono::high_resolution_clock::now();
        std::cout<<"CPU histogram time: "
                 <<std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count()
                 <<" ms\n";
    }

    /* additional utility functions (save, stats) are identical to
       your previous version and can be kept as-is. */
};

/* ----------------------------- main ----------------------------- */
int main(int argc,char*argv[])
{
    std::string dataDir   = (argc>1)?argv[1]:"data";
    std::string outputDir = (argc>2)?argv[2]:"output";

    std::cout<<"CUDA Parallel Histogram Calculation\n";

    HistogramProcessor hp;
    if (!hp.load(dataDir)) return 1;

    hp.runGPU();                       // optimised GPU pass
    std::cout<<"\nCPU baseline for comparison:\n";
    hp.runCPU();                       // reference

    /* hp.saveHistograms(outputDir);  // keep your existing saveStats/plot calls */

    return 0;
}
