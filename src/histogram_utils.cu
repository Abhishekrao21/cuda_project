#include "histogram_utils.h"

/* ====================================================== */
/*  KERNEL – one image per block, 512 threads per block    */
/* ====================================================== */
__global__
void computeHistogramKernel(const unsigned char* d_img,
                            int W, int H,
                            unsigned int* d_hist)
{
    __shared__ unsigned int local[HISTOGRAM_SIZE];
    const int tid = threadIdx.x;
    const int pix = W * H;
    if (tid < HISTOGRAM_SIZE) local[tid] = 0;
    __syncthreads();

    for (int i = tid; i < pix; i += blockDim.x)
        atomicAdd(&local[d_img[i]], 1U);
    __syncthreads();

    if (tid < HISTOGRAM_SIZE)
        atomicAdd(&d_hist[tid], local[tid]);
}

/* ---------------- CPU reference ---------------- */
void computeHistogramCPU(const cv::Mat& img, unsigned int* hist)
{
    std::fill(hist, hist + HISTOGRAM_SIZE, 0U);
    for (int r = 0; r < img.rows; ++r)
        for (int c = 0; c < img.cols; ++c)
            ++hist[ img.at<uchar>(r,c) ];
}

/* ---------------- CSV writer ------------------- */
void writeHistogramCSV(const std::string& file, const unsigned int* hist)
{
    std::ofstream f(file);
    f << "Intensity,Count\n";
    for (int i = 0; i < HISTOGRAM_SIZE; ++i)
        f << i << ',' << hist[i] << '\n';
}

/* -------------- image file discovery ----------- */
std::vector<std::string> getImagePaths(const std::string& folder)
{
    std::vector<std::string> paths;
    for (auto& e : std::filesystem::directory_iterator(folder))
        if (e.is_regular_file()) {
            auto ext = e.path().extension().string();
            if (ext==".png"||ext==".jpg"||ext==".bmp"||ext==".jpeg")
                paths.push_back(e.path().string());
        }
    return paths;
}
