# Debug
DEBUG?=0

# Compiler paths
HOST_COMPILER ?= g++
NVCC          ?= nvcc

# Compiler/linker flags
# MD is --generate-dependencies-with-compile
NVCCFLAGS   ?= -std=c++20 -MD
CCFLAGS     ?=
LDFLAGS     ?=

# Autodetect native arch (only on newer nvcc versions, CUDA 11.6 or greater)
GENCODE_FLAGS ?= --gpu-architecture=native
# Turing
# GENCODE_FLAGS ?= --gpu-architecture=sm_75 -gencode=arch=compute_75,code=sm_75
# Ampere
# GENCODE_FLAGS ?= --gpu-architecture=sm_86 -gencode=arch=compute_86,code=sm_86

# Debug build flags
# -g is --debug, -G is --device-debug, -Xptxas passes options to the PTX assembler
ifeq ($(DEBUG),1)
        NVCCFLAGS += -DDEBUG -g -G -src-in-ptx -keep -keep-dir $(BUILD_DIR) -Xptxas -O0,-v
        CCFLAGS += -Wall -Og -rdynamic
        BUILD_TYPE := debug
$(info DEBUG configuration enabled)
else
        NVCCFLAGS += -Xptxas -O3
        CCFLAGS += -Wall -O2 -march=native -mtune=native
        BUILD_TYPE := release
endif

# Project directories
SRC_DIR := src
BUILD_DIR ?= build
TARGET := stNTT

# External includes and libraries
INCLUDES  := -I$(SRC_DIR)
LIBRARIES := 

# Force host compiler with -ccbin, and add flags for nvcc
# Also pass host compiler options to NVCC (-Xcompiler or --compiler-options) and linker (-Xlinker or --linker-options)
NVCCARGS := -ccbin $(HOST_COMPILER) $(NVCCFLAGS) $(if $(CCFLAGS),-Xcompiler $(subst $() ,$(strip ,),$(CCFLAGS))) $(if $(LDFLAGS),-Xlinker $(subst $() ,$(strip ,),$(LDFLAGS)))

# Get versions from git and build env, embed important compiler flags
VERFLAGS := -DVERSION='"$(shell git describe --dirty --broken --always --tags)"' -DCOMPILEOPTS='"$(NVCCARGS)"' -DCOMPILEVER='"$(shell $(HOST_COMPILER) --version | head -n1), $(shell $(NVCC) --version | head -n5 | tail -n1)"'

# NTT
SRCS := ntt/ntt_cpu.cpp ntt/ntt_util.cpp ntt/cuda/st_ntt.cu
# Tests
# SRCS += tests/tests.cpp tests/benchmarks.cpp
# Util
# SRCS += util/cudahelpers.cu util/debug.cpp util/signals.cpp util/consolehelpers.cpp
# Main
SRCS += main.cpp

.PHONY: all clean run ptx
# Default, make executable target
all: $(BUILD_DIR)/$(TARGET)

# Objects
OBJS := $(addprefix $(BUILD_DIR)/, $(SRCS:%=%.o))

# Dependency files
-include $(OBJS:%.o=%.d)

# Final executable build
$(BUILD_DIR)/$(TARGET): $(OBJS)
		$(EXEC) $(NVCC) $(NVCCARGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

# Other source files
$(OBJS): $(BUILD_DIR)/%.o: $(SRC_DIR)/%
		mkdir -p $(dir $@)
		$(EXEC) $(NVCC) $(INCLUDES) $(NVCCARGS) $(GENCODE_FLAGS) $(VERFLAGS) -o $@ -c $<

# Extract PTX for debug
ptx:: $(OBJS:%.cu.o=%.ptx)
$(BUILD_DIR)/%.ptx: $(BUILD_DIR)/%.cu.o
		cuobjdump -ptx $< > $@

# Run main target executable
run:: $(BUILD_DIR)/$(TARGET)
		$(EXEC) $(BUILD_DIR)/$(TARGET)

# Clean build files
clean::
		rm -rf $(BUILD_DIR)