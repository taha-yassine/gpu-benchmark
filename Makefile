# Compiler
NVCC = nvcc

# Compiler flags
NVCC_FLAGS += -std=c++11

# Linker flags
LDFLAGS += $(addprefix -Xlinker ,-rpath /run/opengl-driver/lib)

# Target executable
TARGET = main

# Source files
SRCS = main.cu benchmark.cu

# Header files
HDRS = benchmark.h

# Object files
OBJS = $(SRCS:.cu=.o)

# Rule to build the target executable
$(TARGET): $(OBJS)
	$(NVCC) $(LDFLAGS) -o $@ $^

# Rule to compile CUDA source files
%.o: %.cu $(HDRS)
	$(NVCC) $(NVCC_FLAGS) -c $<

# Rule to clean the build artifacts
clean:
	rm -f $(OBJS) $(TARGET)
