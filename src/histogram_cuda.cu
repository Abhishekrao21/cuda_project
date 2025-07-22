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

#define HISTOGRAM_SIZE 256
#define THREADS_PER_BLOCK 512
#define CUDA_CHECK(call) do{cudaError_t err=call; if(err!=cudaSuccess){std::cerr<<cudaGetErrorString(err)<<"\n"; std::exit(EXIT_FAILURE);}}while(0)

__global__ void batchHistogramKernelOpt(const unsigned char* images, unsigned int* histograms, int W, int H, int nImg) {
    int imgIdx = blockIdx.y;
    if(imgIdx>=nImg) return;
    __shared__ unsigned int local[HISTOGRAM_SIZE];
    int tid = threadIdx.x;
    int pix = W*H;
    int stride = gridDim.x*blockDim.x;
    int gid = blockIdx.x*blockDim.x + tid;
    if(tid<HISTOGRAM_SIZE) local[tid]=0;
    __syncthreads();
    const unsigned char* img = images + imgIdx*pix;
    for(int i=gid;i<pix;i+=stride) atomicAdd(&local[img[i]],1u);
    __syncthreads();
    if(tid<HISTOGRAM_SIZE) atomicAdd(&histograms[imgIdx*HISTOGRAM_SIZE+tid], local[tid]);
}

class HistogramProcessor {
    std::vector<cv::Mat> images_;
    std::vector<std::string> names_;
    std::vector<std::vector<unsigned int>> hist_;
public:
    bool load(const std::string &dir) {
        images_.clear(); names_.clear();
        if(!std::filesystem::exists(dir)) return false;
        for(auto &e: std::filesystem::directory_iterator(dir)) {
            auto ext=e.path().extension().string();
            if(e.is_regular_file()&&(ext==".png"||ext==".jpg"||ext==".bmp"||ext==".jpeg")) {
                cv::Mat img=cv::imread(e.path().string(), cv::IMREAD_GRAYSCALE);
                if(!img.empty()){images_.push_back(img); names_.push_back(e.path().filename().string());}
            }
        }
        return !images_.empty();
    }
    void runGPU() {
        if(images_.empty()) return;
        int W=images_[0].cols, H=images_[0].rows, nImg=images_.size(), pix=W*H;
        for(auto &im:images_) if(im.cols!=W||im.rows!=H) cv::resize(im,im,{W,H});
        size_t imgB=1ULL*nImg*pix;
        size_t histB=1ULL*nImg*HISTOGRAM_SIZE*sizeof(unsigned int);
        unsigned char *hImgPinned, *dImg;
        unsigned int *dHist;
        CUDA_CHECK(cudaHostAlloc(&hImgPinned,imgB,cudaHostAllocDefault));
        for(int i=0;i<nImg;++i) std::memcpy(hImgPinned+i*pix,images_[i].data,pix);
        CUDA_CHECK(cudaMalloc(&dImg,imgB));
        CUDA_CHECK(cudaMalloc(&dHist,histB));
        CUDA_CHECK(cudaMemset(dHist,0,histB));
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));
        auto t0=std::chrono::high_resolution_clock::now();
        CUDA_CHECK(cudaMemcpyAsync(dImg,hImgPinned,imgB,cudaMemcpyHostToDevice,stream));
        int blocks=(pix+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK;
        dim3 grid(blocks,nImg);
        batchHistogramKernelOpt<<<grid,THREADS_PER_BLOCK,0,stream>>>(dImg,dHist,W,H,nImg);
        std::vector<unsigned int> hHist(nImg*HISTOGRAM_SIZE);
        CUDA_CHECK(cudaMemcpyAsync(hHist.data(),dHist,histB,cudaMemcpyDeviceToHost,stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        auto t1=std::chrono::high_resolution_clock::now();
        std::cout<<"GPU time:"<<std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count()<<"ms\n";
        hist_.assign(nImg,std::vector<unsigned int>(HISTOGRAM_SIZE));
        for(int i=0;i<nImg;++i) std::copy_n(hHist.data()+i*HISTOGRAM_SIZE,HISTOGRAM_SIZE,hist_[i].begin());
        CUDA_CHECK(cudaFreeHost(hImgPinned)); CUDA_CHECK(cudaFree(dImg)); CUDA_CHECK(cudaFree(dHist)); CUDA_CHECK(cudaStreamDestroy(stream));
    }
    void runCPU() {
        auto t0=std::chrono::high_resolution_clock::now();
        hist_.assign(images_.size(),std::vector<unsigned int>(HISTOGRAM_SIZE));
        for(size_t i=0;i<images_.size();++i)
            for(int r=0;r<images_[i].rows;++r)
                for(int c=0;c<images_[i].cols;++c)
                    ++hist_[i][images_[i].at<uchar>(r,c)];
        auto t1=std::chrono::high_resolution_clock::now();
        std::cout<<"CPU time:"<<std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count()<<"ms\n";
    }
    void saveHistograms(const std::string &out) {
        if(hist_.empty()) return;
        std::filesystem::create_directories(out);
        for(size_t i=0;i<hist_.size();++i) {
            std::ofstream f(out+"/histogram_"+std::to_string(i)+"_"+names_[i]+".csv");
            f<<"Intensity,Count\n";
            for(int j=0;j<HISTOGRAM_SIZE;++j) f<<j<<','<<hist_[i][j]<<"\n";
        }
    }
};

int main(int argc,char*argv[]){
    std::string dataDir="data", outDir="output";
    bool save=false;
    for(int i=1;i<argc;++i){std::string a=argv[i];
        if((a=="-d"||a=="--data")&&i+1<argc) dataDir=argv[++i];
        else if((a=="-o"||a=="--output")&&i+1<argc) outDir=argv[++i];
        else if(a=="-s"||a=="--save") save=true;
    }
    HistogramProcessor hp;
    if(!hp.load(dataDir)) return 1;
    hp.runGPU(); hp.runCPU();
    if(save) hp.saveHistograms(outDir);
    return 0;
}

