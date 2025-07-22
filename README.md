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

#### 1. Generate Test Data
```bash
python3 generate_sample_data.py --output data --small 15 --large 500
```

#### 2. Build the Project
```bash
make all
```

#### 3. Run Histogram Computation
```bash
./histogram_cuda  # run without CSV output

./histogram_cuda -s   # run with CSV output
```

#### 4. Visualize Results
```bash
python3 plot_histograms.py output
```

### Advanced Usage

#### Custom Dataset
```bash
# Place your images in output directory and run
./histogram_cuda -d ~/pictures/batch1 -o ~/results -s
```

## Output Files

The program generates several types of output:

### CSV Files
- `histogram_<id>_<imagename>.csv`: Individual histogram data
- `histogram_summary.csv`: Statistical summary for all images

### Visualization
- `individual_histograms.png`: Sample histogram plots
- `aggregate_histogram.png`: Combined histogram of all images
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


## Performance Characteristics

### Expected Speedup
- **Small images (≤256²)**: 10-50x over CPU
- **Large images (≥512²)**: 50-200x over CPU
- **Batch processing**: Additional 2-5x improvement

### Scalability
- Processes 100s of small images or 1000s of large images efficiently
- Memory usage scales linearly with dataset size
- GPU memory is the primary limiting factor


## Educational Value

This project demonstrates several key CUDA concepts:

1. **Parallel Algorithms**: Histogram computation as a reduction problem
2. **Memory Hierarchy**: Effective use of shared memory
3. **Synchronization**: Thread cooperation and atomic operations
4. **Performance Optimization**: Memory coalescing and occupancy
5. **Real-World Application**: Practical image processing pipeline
