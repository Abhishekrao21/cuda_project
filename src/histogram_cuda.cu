#include "histogram_utils.h"
#include <iomanip>
#include <chrono>

/* ---------- simple logger ---------- */
class Logger {
    std::ofstream f_;
public:
    explicit Logger(const std::string& path) { f_.open(path); }
    template<class T>
    void log(const T& msg) {
        auto t = std::chrono::system_clock::to_time_t(
                     std::chrono::system_clock::now());
        std::stringstream ss; ss<< std::put_time(std::localtime(&t),"%F %T");
        std::string line = "["+ss.str()+"] "+msg;
        std::cout << line << '\n';
        if (f_.is_open()) f_ << line << '\n';
    }
};

/* ====================================================== */
int main(int argc,char**argv)
{
    std::string datDir="data", outDir="output";
    bool save=false;
    for(int i=1;i<argc;++i){
        std::string a=argv[i];
        if((a=="-d"||a=="--data")&&i+1<argc)   datDir = argv[++i];
        else if((a=="-o"||a=="--output")&&i+1<argc) outDir=argv[++i];
        else if(a=="-s"||a=="--save") save=true;
    }
    std::filesystem::create_directories(outDir);
    Logger log(outDir+"/execution_log.txt");
    log.log("Starting batch histogram run");

    /* -------- load images -------- */
    auto paths = getImagePaths(datDir);
    if(paths.empty()){ log.log("No images found."); return 1; }
    std::vector<cv::Mat> imgs;
    for(auto& p:paths) imgs.emplace_back(cv::imread(p,cv::IMREAD_GRAYSCALE));
    int W=imgs[0].cols, H=imgs[0].rows;
    for(auto& im:imgs) if(im.cols!=W||im.rows!=H) cv::resize(im,im,{W,H});

    /* -------- allocate device -------- */
    const int n=imgs.size(), pix=W*H;
    size_t imgB=1ULL*n*pix, histB=1ULL*n*HISTOGRAM_SIZE*sizeof(unsigned int);
    unsigned char *dImg;  unsigned int *dHist;
    CHECK_CUDA(cudaMalloc(&dImg,  imgB));
    CHECK_CUDA(cudaMalloc(&dHist, histB));
    CHECK_CUDA(cudaMemset(dHist,0,histB));

    /* -------- copy host→device -------- */
    std::vector<unsigned char> hAll(imgB);
    for(int i=0;i<n;++i) std::memcpy(hAll.data()+i*pix,imgs[i].data,pix);
    CHECK_CUDA(cudaMemcpy(dImg,hAll.data(),imgB,cudaMemcpyHostToDevice));

    /* -------- launch kernel ---------- */
    int blocks=(pix+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK;
    dim3 grid(blocks,n);
    auto t0=std::chrono::high_resolution_clock::now();
    batchHistogramKernelOpt<<<grid,THREADS_PER_BLOCK>>>(dImg,dHist,W,H,n);
    CHECK_CUDA(cudaDeviceSynchronize());
    auto t1=std::chrono::high_resolution_clock::now();
    log.log("GPU time: "
        +std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count())+" ms");

    /* -------- retrieve one example histogram ---------- */
    std::vector<unsigned int> first(HISTOGRAM_SIZE);
    CHECK_CUDA(cudaMemcpy(first.data(), dHist, HISTOGRAM_SIZE*sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));
    log.log("Example pixel count check (image0): "
            +std::to_string(std::accumulate(first.begin(),first.end(),0U)));

    /* -------- optional CSV ---------- */
    if(save){
        for(int i=0;i<n;++i){
            std::vector<unsigned int> h(HISTOGRAM_SIZE);
            CHECK_CUDA(cudaMemcpy(h.data(), dHist+i*HISTOGRAM_SIZE,
                      HISTOGRAM_SIZE*sizeof(unsigned int), cudaMemcpyDeviceToHost));
            writeHistogramCSV(outDir+"/histogram_"+std::to_string(i)+".csv", h.data());
        }
        log.log("CSV files written.");
    }

    CHECK_CUDA(cudaFree(dImg)); CHECK_CUDA(cudaFree(dHist));
    log.log("Processing complete.");
    return 0;
}
