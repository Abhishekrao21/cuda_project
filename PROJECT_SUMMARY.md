# CUDA Parallel Histogram Calculation - Project Overview

## Quick Reference

### Project Goals
- Demonstrate CUDA programming proficiency
- Process large batches of images efficiently
- Show GPU acceleration benefits
- Implement parallel reduction algorithms

### Key Features
✅ GPU-accelerated histogram computation
✅ Batch processing of mixed-size images
✅ Memory optimization with shared memory
✅ Statistical analysis and anomaly detection
✅ Performance benchmarking vs CPU
✅ Comprehensive visualization

### File Overview

| File/Directory | Purpose |
|---------------|---------|
| `src/histogram_cuda.cu` | Main CUDA implementation |
| `Makefile` | Build configuration |
| `plot_histograms.py` | Results visualization |
| `generate_sample_data.py` | Test data creation |
| `build.sh` | Convenience build script |
| `data/` | Input images directory |
| `output/` | Results and analysis |
| `build/` | Compiled binaries |

# CUDA Parallel Histogram Calculation

This project implements a **high-performance**, **batched histogram computation** for grayscale images using **CUDA**. It also includes a CPU baseline implementation to compare performance, showcasing efficient GPU programming techniques.

---

## 🧭 Overview

- **Purpose**: Compute pixel intensity histograms (0–255) for a batch of grayscale images.
- **Significance**: Demonstrates key CUDA strategies like:
  - Shared memory reduction
  - Atomic operations
  - Asynchronous data transfer
  - GPU resource optimization

---

## 📈 Algorithm Breakdown

### 1. Data Load & Resize

- Load all grayscale images from a given directory (`.png`, `.jpg`, `.bmp`, `.jpeg`).
- Resize each image to match the dimensions of the first one for uniform GPU processing.

### 2. Pinned Memory & Asynchronous Transfer

- Allocate **pinned host memory** (`cudaHostAlloc`) for all images in a contiguous batch.
- Use `cudaMemcpyAsync` and a **single CUDA stream** to **overlap** memory copy (H→D) with kernel execution.

### 3. Kernel: `batchHistogramKernelOpt`

#### Grid Design:
- **Grid X**: Enough blocks to cover `W * H` pixels per image (with strided thread access).
- **Grid Y**: One block row per image (`nImg` images in batch).

#### Block:
- 512 threads per block.

#### Shared Memory:
- 256-bin local histogram in shared memory per block (`__shared__ unsigned int local[256]`).

#### Thread Work:
- Each thread processes a strided set of pixels from one image.
- Updates the local histogram using `atomicAdd(&local[img[i]], 1)`.

#### Global Write:
- After local accumulation, threads cooperatively write results to **global memory** using `atomicAdd(&histograms[imgIdx * 256 + bin], local[bin])`.

### 4. Host Retrieval & Optional Output

- Histograms are copied back to host asynchronously.
- If `-s` or `--save` is passed, per-image `.csv` histogram files are written to the `output/` directory.

---

