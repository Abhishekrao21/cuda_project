# CUDA Parallel Histogram Calculation Makefile

# Compiler settings
NVCC = nvcc
CXX = g++

# CUDA architecture (adjust based on your GPU)
CUDA_ARCH = -arch=sm_60

# OpenCV settings
OPENCV_CFLAGS = $(shell pkg-config --cflags opencv4)
OPENCV_LIBS = $(shell pkg-config --libs opencv4)

# Directories
SRC_DIR = src
BUILD_DIR = build
OUTPUT_DIR = output
DATA_DIR = data

# Compiler flags
NVCC_FLAGS = -std=c++17 $(CUDA_ARCH) -O3 -Xcompiler -fopenmp
CXX_FLAGS = -std=c++17 -O3 -fopenmp

# Include directories
INCLUDES = -I/usr/local/cuda/include

# Library directories
LIBDIRS = -L/usr/local/cuda/lib64

# Libraries
LIBS = -lcuda -lcudart $(OPENCV_LIBS) -lm

# Target executable
TARGET = $(BUILD_DIR)/histogram_cuda

# Source files
CUDA_SRC = $(SRC_DIR)/histogram_cuda.cu

# Default target
all: directories $(TARGET)

# Create necessary directories
directories:
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(OUTPUT_DIR)
	@mkdir -p $(DATA_DIR)

# Build the main executable
$(TARGET): $(CUDA_SRC)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(CUDA_SRC) -o $@ $(LIBDIRS) $(LIBS)

# Clean build files
clean:
	rm -rf $(BUILD_DIR)/*
	rm -rf $(OUTPUT_DIR)/*

# Install dependencies (Ubuntu/Debian)
install-deps:
	sudo apt update
	sudo apt install -y libopencv-dev python3-pip python3-matplotlib python3-numpy python3-pandas
	pip3 install -r requirements.txt

# Generate sample data
generate-data:
	python3 generate_sample_data.py

# Run the program
run: $(TARGET)
	./$(TARGET) $(DATA_DIR) $(OUTPUT_DIR)

# Run with timing
run-benchmark: $(TARGET)
	time ./$(TARGET) $(DATA_DIR) $(OUTPUT_DIR)

# Plot results
plot:
	python3 plot_histograms.py $(OUTPUT_DIR)

# Full pipeline: build, generate data, run, and plot
pipeline: all generate-data run plot

# Help target
help:
	@echo "Available targets:"
	@echo "  all           - Build the project"
	@echo "  directories   - Create necessary directories"
	@echo "  clean         - Clean build files"
	@echo "  install-deps  - Install system dependencies"
	@echo "  generate-data - Generate sample test images"
	@echo "  run           - Run the histogram computation"
	@echo "  run-benchmark - Run with timing information"
	@echo "  plot          - Generate histogram plots"
	@echo "  pipeline      - Full build and run pipeline"
	@echo "  help          - Show this help message"

.PHONY: all directories clean install-deps generate-data run run-benchmark plot pipeline help