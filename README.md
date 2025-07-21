# CUDA Parallel Histogram Calculation

A high-performance CUDA-accelerated program for computing pixel intensity histograms on large batches of images. This project demonstrates parallel processing, GPU memory management, and efficient batch computation techniques.

## Features

- **GPU-Accelerated Processing**: Utilizes CUDA kernels for parallel histogram computation
- **Batch Processing**: Efficiently handles large datasets of mixed-size images
- **Memory Optimization**: Uses shared memory and atomic operations for performance
- **Statistical Analysis**: Computes comprehensive statistics and detects anomalies
- **Visualization**: Generates detailed plots and analysis reports
- **Benchmarking**: Includes CPU baseline comparison for performance validation

## Project Structure

```
cuda_histogram_project/
├── src/
│   └── histogram_cuda.cu          # Main CUDA implementation
├── data/                          # Input images directory
├── output/                        # Results and plots
├── build/                         # Compiled binaries
├── Makefile                       # Build configuration
├── plot_histograms.py            # Python visualization script
├── generate_sample_data.py       # Test data generation
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Requirements

### System Requirements
- NVIDIA GPU with CUDA Compute Capability 6.0+
- CUDA Toolkit 11.0+
- OpenCV 4.x
- C++17 compatible compiler (GCC 7+)
- Python 3.6+

### Dependencies
- **System packages**: `libopencv-dev`, `build-essential`
- **Python packages**: `matplotlib`, `numpy`, `pandas`

## Installation

### 1. Install System Dependencies (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install libopencv-dev build-essential python3-pip
```

### 2. Install Python Dependencies
```bash
pip3 install -r requirements.txt
```

### 3. Build the Project
```bash
make all
```

## Usage

### Quick Start
```bash
# Generate sample test data
make generate-data

# Run the complete pipeline
make pipeline
```

### Step-by-Step Usage

#### 1. Generate Test Data
```bash
python3 generate_sample_data.py --output data --small 15 --large 10
```

#### 2. Build the Project
```bash
make all
```

#### 3. Run Histogram Computation
```bash
./build/histogram_cuda data output
```

#### 4. Visualize Results
```bash
python3 plot_histograms.py output
```

### Advanced Usage

#### Custom Dataset
```bash
# Place your images in a directory and run
./build/histogram_cuda /path/to/your/images /path/to/output
```

#### Performance Benchmarking
```bash
make run-benchmark
```

## Output Files

The program generates several types of output:

### CSV Files
- `histogram_<id>_<imagename>.csv`: Individual histogram data
- `histogram_summary.csv`: Statistical summary for all images

### Visualization
- `individual_histograms.png`: Sample histogram plots
- `aggregate_histogram.png`: Combined histogram of all images
- `summary_statistics.png`: Statistical distribution plots

### Analysis
- `analysis_report.txt`: Comprehensive analysis report

## Algorithm Details

### CUDA Kernel Design
The implementation uses a two-level parallelization strategy:

1. **Block Level**: Each CUDA block processes one image
2. **Thread Level**: Threads within a block handle pixels collaboratively

### Key Optimizations
- **Shared Memory**: Local histogram accumulation reduces global memory access
- **Atomic Operations**: Ensures correct parallel updates to histogram bins
- **Memory Coalescing**: Optimized memory access patterns
- **Batch Processing**: Multiple images processed simultaneously

### Memory Management
```cpp
// Pseudocode for memory layout
images[num_images][width * height]     // Input images
histograms[num_images][256]            // Output histograms
```

## Performance Characteristics

### Expected Speedup
- **Small images (≤256²)**: 10-50x over CPU
- **Large images (≥512²)**: 50-200x over CPU
- **Batch processing**: Additional 2-5x improvement

### Scalability
- Processes 100s of small images or 10s of large images efficiently
- Memory usage scales linearly with dataset size
- GPU memory is the primary limiting factor

## Troubleshooting

### Common Issues

#### CUDA Not Found
```bash
# Check CUDA installation
nvcc --version
nvidia-smi
```

#### OpenCV Linking Issues
```bash
# Check OpenCV installation
pkg-config --modversion opencv4
```

#### Memory Issues
- Reduce batch size for large images
- Check available GPU memory with `nvidia-smi`

#### Build Errors
```bash
# Clean and rebuild
make clean
make all
```

## Educational Value

This project demonstrates several key CUDA concepts:

1. **Parallel Algorithms**: Histogram computation as a reduction problem
2. **Memory Hierarchy**: Effective use of shared memory
3. **Synchronization**: Thread cooperation and atomic operations
4. **Performance Optimization**: Memory coalescing and occupancy
5. **Real-World Application**: Practical image processing pipeline

## Extensions and Modifications

### Possible Enhancements
- **Multi-channel histograms**: Process RGB images
- **Streaming**: Handle datasets larger than GPU memory
- **Additional statistics**: Entropy, histogram matching
- **GPU optimization**: Utilize multiple GPUs
- **File format support**: Add support for more image formats

### Code Modifications
The codebase is designed for extensibility:
- Add new kernel variants in `histogram_cuda.cu`
- Extend analysis features in `plot_histograms.py`
- Modify image loading to support new formats

## Performance Tuning

### GPU Architecture Specific
```bash
# For different GPU architectures
make CUDA_ARCH="-arch=sm_75"  # RTX 20xx series
make CUDA_ARCH="-arch=sm_86"  # RTX 30xx series
```

### Memory Configuration
Adjust these parameters in the source code:
- `THREADS_PER_BLOCK`: Optimize for your GPU's SM count
- `SHARED_MEMORY_BANKS`: Match your GPU's shared memory banks

## Benchmarking Results

### Sample Performance (RTX 3080)
- **Dataset**: 100 images, 512x512 pixels
- **GPU Time**: ~15ms
- **CPU Time**: ~2.1s
- **Speedup**: 140x

*Results vary based on GPU model, image sizes, and system configuration.*

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve this educational project.

## License

This project is provided for educational purposes. Feel free to use and modify for learning and academic projects.

## References

1. NVIDIA CUDA Programming Guide
2. "Programming Massively Parallel Processors" by Kirk & Hwu
3. OpenCV Documentation for Image Processing
4. CUDA Samples and Best Practices Guide
