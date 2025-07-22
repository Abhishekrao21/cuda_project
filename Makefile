NVCC := nvcc
CUDA_ARCH := -arch=sm_75
OPENCV_CFLAGS := $(shell pkg-config --cflags opencv4)
OPENCV_LIBS := $(shell pkg-config --libs opencv4)

NVCC_FLAGS := -std=c++17 $(CUDA_ARCH) -O3 -Xcompiler -fopenmp $(OPENCV_CFLAGS) -I/usr/local/cuda/include
LDLIBS := -lcuda -lcudart $(OPENCV_LIBS) -lm

SRC_DIR := src
OUTPUT_DIR := output
DATA_DIR := data
CUDA_UTILS_SRC := $(SRC_DIR)/histogram_utils.cu
CUDA_SRC := $(SRC_DIR)/histogram_cuda.cu
TARGET := histogram_cuda

all: $(TARGET)

$(TARGET): $(CUDA_UTILS_SRC) $(CUDA_SRC)
@mkdir -p $(OUTPUT_DIR) $(DATA_DIR)
$(NVCC) $(NVCC_FLAGS) $^ -o $@ $(LDLIBS)

clean:
rm -f $(TARGET)
rm -rf $(OUTPUT_DIR)/*

run: $(TARGET)
./$(TARGET) $(DATA_DIR) $(OUTPUT_DIR) -s

.PHONY: all clean run
