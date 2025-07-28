# Makefile for SQLite AI Extension
# Supports compilation for Linux, macOS, Windows, Android and iOS

# customize sqlite3 executable with 
# make test SQLITE3=/opt/homebrew/Cellar/sqlite/3.49.1/bin/sqlite3
SQLITE3 ?= sqlite3

# Set default platform if not specified
ifeq ($(OS),Windows_NT)
	PLATFORM := windows
	HOST := windows
	CPUS := $(shell powershell -Command "[Environment]::ProcessorCount")
else
	HOST = $(shell uname -s | tr '[:upper:]' '[:lower:]')
	ifeq ($(HOST),darwin)
		PLATFORM := macos
		CPUS := $(shell sysctl -n hw.ncpu)
	else
		PLATFORM := $(HOST)
		CPUS := $(shell nproc)
	endif
endif

# Speed up builds by using all available CPU cores
MAKEFLAGS += -j$(CPUS)

# Directories
SRC_DIR = src
DIST_DIR = dist
VPATH = $(SRC_DIR)
BUILD_DIR = build
LLAMA_DIR = modules/llama.cpp
WHISPER_DIR = modules/whisper.cpp
MINIAUDIO_DIR = modules/miniaudio
BUILD_LLAMA = $(BUILD_DIR)/llama.cpp
BUILD_WHISPER = $(BUILD_DIR)/whisper.cpp
BUILD_MINIAUDIO = $(BUILD_DIR)/miniaudio

# Compiler and flags
ifeq ($(PLATFORM),windows)
	# Check if this is a CUDA or SYCL build that needs MSVC
	ifneq (,$(findstring CUDA,$(LLAMA)))
		USE_MSVC = 1
	endif
	ifneq (,$(findstring SYCL,$(LLAMA)))
		USE_MSVC = 1
	endif
	# HIP builds use their own Clang compiler, not MSVC
	
	ifeq ($(USE_MSVC),1)
		CC = cl
		CXX = cl
		CFLAGS = /nologo /I$(SRC_DIR) /I$(LLAMA_DIR)/ggml/include /I$(LLAMA_DIR)/include /I$(WHISPER_DIR)/include /I$(MINIAUDIO_DIR)
		OBJ_EXT = .obj
		LIB_EXT = .lib
	else
		CC = gcc
		CXX = g++
		CFLAGS = -Wall -Wextra -Wno-unused-parameter -I$(SRC_DIR) -I$(LLAMA_DIR)/ggml/include -I$(LLAMA_DIR)/include -I$(WHISPER_DIR)/include -I$(MINIAUDIO_DIR)
		OBJ_EXT = .o
		LIB_EXT = .a
	endif
else
	CC = gcc
	CXX = g++
	CFLAGS = -Wall -Wextra -Wno-unused-parameter -I$(SRC_DIR) -I$(LLAMA_DIR)/ggml/include -I$(LLAMA_DIR)/include -I$(WHISPER_DIR)/include -I$(MINIAUDIO_DIR)
	OBJ_EXT = .o
	LIB_EXT = .a
endif
LLAMA_OPTIONS = $(LLAMA) -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF -DGGML_RPC=OFF
WHISPER_OPTIONS = $(WHISPER) -DBUILD_SHARED_LIBS=OFF -DWHISPER_BUILD_EXAMPLES=OFF -DWHISPER_BUILD_TESTS=OFF -DWHISPER_BUILD_SERVER=OFF -DWHISPER_RPC=OFF
MINIAUDIO_OPTIONS = $(MINIAUDIO) -DBUILD_SHARED_LIBS=OFF -DMINIAUDIO_BUILD_EXAMPLES=OFF -DMINIAUDIO_BUILD_TESTS=OFF
# MinGW produces .a files without lib prefix, use -l:filename.a syntax
ifeq ($(PLATFORM),windows)
L = -l:
A = .a
else
L = -l
endif
# Module-specific linker flags
ifeq ($(USE_MSVC),1)
	# MSVC uses .lib files directly, no -L/-l flags needed for modules
	LLAMA_LDFLAGS = 
	WHISPER_LDFLAGS = 
	MINIAUDIO_LDFLAGS = 
else
	# MinGW/GCC uses -L/-l flags
	LLAMA_LDFLAGS = -L./$(BUILD_LLAMA)/common -L./$(BUILD_LLAMA)/ggml/src -L./$(BUILD_LLAMA)/src -lcommon -lllama $(L)ggml$(A) $(L)ggml-base$(A) $(L)ggml-cpu$(A)
	WHISPER_LDFLAGS = -L./$(BUILD_WHISPER)/src -lwhisper
	MINIAUDIO_LDFLAGS = -L./$(BUILD_MINIAUDIO) -lminiaudio
endif
LDFLAGS = $(LLAMA_LDFLAGS) $(WHISPER_LDFLAGS) $(MINIAUDIO_LDFLAGS)

# Files
SRC_FILES = $(wildcard $(SRC_DIR)/*.c)
OBJ_FILES = $(patsubst %.c, $(BUILD_DIR)/%$(OBJ_EXT), $(notdir $(SRC_FILES)))
ifeq ($(USE_MSVC),1)
	LLAMA_LIBS = $(BUILD_LLAMA)/common/Release/common.lib $(BUILD_LLAMA)/ggml/src/Release/ggml.lib $(BUILD_LLAMA)/ggml/src/Release/ggml-base.lib $(BUILD_LLAMA)/ggml/src/Release/ggml-cpu.lib $(BUILD_LLAMA)/src/Release/llama.lib
	WHISPER_LIBS = $(BUILD_WHISPER)/src/Release/whisper.lib
	MINIAUDIO_LIBS = $(BUILD_MINIAUDIO)/Release/miniaudio.lib
else
	LLAMA_LIBS = $(BUILD_LLAMA)/common/libcommon.a $(BUILD_LLAMA)/ggml/src/libggml.a $(BUILD_LLAMA)/ggml/src/libggml-base.a $(BUILD_LLAMA)/ggml/src/libggml-cpu.a $(BUILD_LLAMA)/src/libllama.a
	WHISPER_LIBS = $(BUILD_WHISPER)/src/libwhisper.a
	MINIAUDIO_LIBS = $(BUILD_MINIAUDIO)/libminiaudio.a
endif

# Platform-specific settings
ifeq ($(PLATFORM),windows)
	TARGET := $(DIST_DIR)/ai.dll
	ifeq ($(USE_MSVC),1)
		LDFLAGS += /DLL
		MSVC_LIBS += bcrypt.lib
		DEF_FILE := $(BUILD_DIR)/ai.def
		STRIP = echo "No stripping needed for MSVC"
	else
		LDFLAGS += -shared -lbcrypt -lgomp -lstdc++
		DEF_FILE := $(BUILD_DIR)/ai.def
		STRIP = strip --strip-unneeded $@
	endif
else ifeq ($(PLATFORM),macos)
	TARGET := $(DIST_DIR)/ai.dylib
	LLAMA_LIBS += $(BUILD_LLAMA)/ggml/src/ggml-metal/libggml-metal.a $(BUILD_LLAMA)/ggml/src/ggml-blas/libggml-blas.a
	LDFLAGS += -arch x86_64 -arch arm64 -L./$(BUILD_LLAMA)/ggml/src/ggml-metal -lggml-metal -L./$(BUILD_LLAMA)/ggml/src/ggml-blas -lggml-blas -framework Metal -framework Foundation -framework CoreFoundation -framework QuartzCore -dynamiclib -undefined dynamic_lookup
	CFLAGS += -arch x86_64 -arch arm64
	LLAMA_OPTIONS += -DCMAKE_OSX_ARCHITECTURES="x86_64;arm64"
	WHISPER_OPTIONS += -DCMAKE_OSX_ARCHITECTURES="x86_64;arm64"
	MINIAUDIO_OPTIONS += -DCMAKE_OSX_ARCHITECTURES="x86_64;arm64"
	STRIP = strip -x -S $@
else ifeq ($(PLATFORM),android)
	# Set ARCH to find Android NDK's Clang compiler, the user should set the ARCH
	ifeq ($(filter %,$(ARCH)),)
		$(error "Android ARCH must be set to ARCH=x86_64 or ARCH=arm64-v8a")
	endif
	# Set ANDROID_NDK path to find android build tools
	# e.g. on MacOS: export ANDROID_NDK=/Users/username/Library/Android/sdk/ndk/25.2.9519653
	ifeq ($(filter %,$(ANDROID_NDK)),)
		$(error "Android NDK must be set")
	endif

	BIN = $(ANDROID_NDK)/toolchains/llvm/prebuilt/$(HOST)-x86_64/bin
	PATH := $(BIN):$(PATH)

	ifneq (,$(filter $(ARCH),arm64 arm64-v8a))
		override ARCH := aarch64
	endif

	CC = $(BIN)/$(ARCH)-linux-android26-clang
	CXX = $(CC)++
	TARGET := $(DIST_DIR)/ai.so
	LDFLAGS += -static-libstdc++ -shared
	ANDROID_OPTIONS = -DCMAKE_TOOLCHAIN_FILE=$(ANDROID_NDK)/build/cmake/android.toolchain.cmake -DANDROID_ABI=$(if $(filter aarch64,$(ARCH)),arm64-v8a,$(ARCH)) -DANDROID_PLATFORM=android-26 -DCMAKE_C_FLAGS="-march=$(if $(filter aarch64,$(ARCH)),armv8.7a,x86-64)" -DCMAKE_CXX_FLAGS="-march=$(if $(filter aarch64,$(ARCH)),armv8.7a,x86-64)" -DCMAKE_POSITION_INDEPENDENT_CODE=ON
	LLAMA_OPTIONS += $(ANDROID_OPTIONS) -DGGML_OPENMP=OFF -DGGML_LLAMAFILE=OFF
	WHISPER_OPTIONS += $(ANDROID_OPTIONS) -DGGML_OPENMP=OFF -DGGML_LLAMAFILE=OFF
	MINIAUDIO_OPTIONS += $(ANDROID_OPTIONS)
	STRIP = $(BIN)/llvm-strip --strip-unneeded $@
else ifeq ($(PLATFORM),ios)
	TARGET := $(DIST_DIR)/ai.dylib
	SDK := -isysroot $(shell xcrun --sdk iphoneos --show-sdk-path) -miphoneos-version-min=14.0
	LLAMA_LIBS += $(BUILD_LLAMA)/ggml/src/ggml-metal/libggml-metal.a $(BUILD_LLAMA)/ggml/src/ggml-blas/libggml-blas.a
	WHISPER_LDFLAGS += -lwhisper.coreml
	LDFLAGS += -L./$(BUILD_LLAMA)/ggml/src/ggml-metal -lggml-metal -L./$(BUILD_LLAMA)/ggml/src/ggml-blas -lggml-blas -framework Accelerate -framework Metal -framework Foundation -framework CoreML -framework AVFoundation -framework AudioToolbox -framework CoreAudio -framework Security -ldl -dynamiclib $(SDK)
	CFLAGS += -arch arm64 -x objective-c $(SDK)
	LLAMA_OPTIONS += -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0
	WHISPER_OPTIONS += -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0 -DWHISPER_COREML=ON
	STRIP = strip -x -S $@
else ifeq ($(PLATFORM),isim)
	TARGET := $(DIST_DIR)/ai.dylib
	SDK := -isysroot $(shell xcrun --sdk iphonesimulator --show-sdk-path) -miphonesimulator-version-min=14.0
	LLAMA_LIBS += $(BUILD_LLAMA)/ggml/src/ggml-metal/libggml-metal.a $(BUILD_LLAMA)/ggml/src/ggml-blas/libggml-blas.a
	WHISPER_LDFLAGS += -lwhisper.coreml
	LDFLAGS += -arch x86_64 -arch arm64 -L./$(BUILD_LLAMA)/ggml/src/ggml-metal -lggml-metal -L./$(BUILD_LLAMA)/ggml/src/ggml-blas -lggml-blas -framework Accelerate -framework Metal -framework Foundation -framework CoreML -framework AVFoundation -framework AudioToolbox -framework CoreAudio -framework Security -ldl -dynamiclib $(SDK)
	CFLAGS += -arch x86_64 -arch arm64 -x objective-c $(SDK)
	LLAMA_OPTIONS += -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_SYSROOT=iphonesimulator -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0 -DCMAKE_OSX_ARCHITECTURES="x86_64;arm64"
	WHISPER_OPTIONS += -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_SYSROOT=iphonesimulator -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0 -DCMAKE_OSX_ARCHITECTURES="x86_64;arm64" -DWHISPER_COREML=ON
else # linux
	TARGET := $(DIST_DIR)/ai.so
	LDFLAGS += -shared
	MINIAUDIO_LDFLAGS += -lpthread -lm
	LLAMA_OPTIONS += -DGGML_OPENMP=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON
	WHISPER_OPTIONS += -DGGML_OPENMP=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON
	STRIP = strip --strip-unneeded $@
endif

# Backend specific settings
ifneq (,$(findstring VULKAN,$(LLAMA)))
	LLAMA_LIBS += $(BUILD_LLAMA)/ggml/src/ggml-vulkan/libggml-vulkan.a
	LLAMA_LDFLAGS += -L./$(BUILD_LLAMA)/ggml/src/ggml-vulkan $(L)ggml-vulkan$(A)
	# Vulkan variations
	ifeq ($(PLATFORM),windows)
		VULKAN_VAR = -1
	endif
	# Add Vulkan SDK library path if available
	ifdef VULKAN_SDK
		LLAMA_LDFLAGS += -L$(VULKAN_SDK)/lib -lvulkan$(VULKAN_VAR)
	else # system Vulkan library locations
		LLAMA_LDFLAGS += -lvulkan$(VULKAN_VAR) -ldl
	endif
endif
ifneq (,$(findstring OPENCL,$(LLAMA)))
	LLAMA_LIBS += $(BUILD_LLAMA)/ggml/src/ggml-opencl/libggml-opencl.a
	LLAMA_LDFLAGS += -L./$(BUILD_LLAMA)/ggml/src/ggml-opencl $(L)ggml-opencl$(A) -lOpenCL
	ifneq ($(PLATFORM),windows)
		LLAMA_LDFLAGS += -ldl
	endif
endif
ifneq (,$(findstring BLAS,$(LLAMA)))
	LLAMA_LIBS += $(BUILD_LLAMA)/ggml/src/ggml-blas/libggml-blas.a
	LLAMA_LDFLAGS += -L./$(BUILD_LLAMA)/ggml/src/ggml-blas $(L)ggml-blas$(A)
	# Link against specific BLAS implementations
	ifneq (,$(findstring OpenBLAS,$(LLAMA)))
		LLAMA_LDFLAGS += -lopenblas
	else ifneq (,$(findstring Apple,$(LLAMA)))
		LDFLAGS += -framework Accelerate
	else # Generic BLAS
		LLAMA_LDFLAGS += -lblas
	endif
endif
ifneq (,$(findstring COREML,$(WHISPER))) # CoreML - only macos
	WHISPER_LIBS += $(BUILD_WHISPER)/src/libwhisper.coreml.a
	WHISPER_LDFLAGS += -lwhisper.coreml
	WHISPER_OPTIONS += -DWHISPER_COREML=ON
	LDFLAGS += -framework CoreML
endif
ifneq (,$(findstring CUDA,$(LLAMA)))
	ifneq ($(PLATFORM),windows)
		LLAMA_LDFLAGS += -L./$(BUILD_LLAMA)/ggml/src/ggml-cuda $(L)ggml-cuda$(A) -lcuda -lcublas -lcublasLt -lcudart -ldl
		LLAMA_LIBS += $(BUILD_LLAMA)/ggml/src/ggml-cuda/libggml-cuda.a
	else
		ifeq ($(USE_MSVC),1)
			# MSVC CUDA build - use .lib files and MSVC linker syntax
			LLAMA_LIBS += $(BUILD_LLAMA)/ggml/src/ggml-cuda/Release/ggml-cuda.lib
			MSVC_LIBS += "$(CUDA_PATH)/lib/x64/cuda.lib" "$(CUDA_PATH)/lib/x64/cudart.lib"
		else
			# MinGW CUDA build - use original approach
			LLAMA_LDFLAGS = -L./$(BUILD_LLAMA)/common/Release -L./$(BUILD_LLAMA)/ggml/src/Release -L./$(BUILD_LLAMA)/src/Release -L./$(BUILD_LLAMA)/ggml/src/ggml-cuda/Release -L"$(CUDA_PATH)/lib/x64" $(L)common.lib $(L)llama.lib $(L)ggml.lib $(L)ggml-base.lib $(L)ggml-cuda.lib -lcuda -lcudart
			WHISPER_LDFLAGS = -L./$(BUILD_WHISPER)/src -L./$(BUILD_WHISPER)/src/Release -lwhisper
			MINIAUDIO_LDFLAGS = -L./$(BUILD_MINIAUDIO) -L./$(BUILD_MINIAUDIO)/Release -lminiaudio
			$(error Windows MinGW CUDA build is not supported yet)
		endif
	endif
endif
ifneq (,$(findstring HIP,$(LLAMA)))
	ifneq ($(PLATFORM),windows)
		LLAMA_LDFLAGS += -L./$(BUILD_LLAMA)/ggml/src/ggml-hip $(L)ggml-hip$(A) -lhip -lrocblas -ldl
		LLAMA_LIBS += $(BUILD_LLAMA)/ggml/src/ggml-hip/libggml-hip.a
	else
		# Windows HIP build - use MinGW-style linking with .lib files
		LLAMA_LDFLAGS += -L./$(BUILD_LLAMA)/ggml/src/ggml-hip/Release -L"$(HIP_PATH)/lib" -lggml-hip -lamdhip64 -lrocblas
		LLAMA_LIBS += $(BUILD_LLAMA)/ggml/src/ggml-hip/Release/ggml-hip.lib
	endif
endif
ifneq (,$(findstring SYCL,$(LLAMA)))
	ifneq ($(PLATFORM),windows)
		LLAMA_LDFLAGS += -L./$(BUILD_LLAMA)/ggml/src/ggml-sycl $(L)ggml-sycl$(A) -lsycl -ldl
		LLAMA_LIBS += $(BUILD_LLAMA)/ggml/src/ggml-sycl/libggml-sycl.a
	else
		# Windows SYCL build - use MSVC linking with .lib files
		ifeq ($(USE_MSVC),1)
			LLAMA_LIBS += $(BUILD_LLAMA)/ggml/src/ggml-sycl/Release/ggml-sycl.lib
			MSVC_LIBS += sycl.lib
		else
			LLAMA_LDFLAGS += -L./$(BUILD_LLAMA)/ggml/src/ggml-sycl/Release -lggml-sycl -lsycl
			LLAMA_LIBS += $(BUILD_LLAMA)/ggml/src/ggml-sycl/Release/ggml-sycl.lib
		endif
	endif
endif

# Windows .def file generation
$(DEF_FILE):
ifeq ($(PLATFORM),windows)
	@echo "LIBRARY ai.dll" > $@
	@echo "EXPORTS" >> $@
	@echo "	sqlite3_ai_init" >> $@
endif

# Make sure the build and dist directories exist
$(shell mkdir -p $(BUILD_DIR) $(DIST_DIR))

# Default target
extension: $(TARGET)
all: $(TARGET) 

# Loadable library
$(TARGET): $(OBJ_FILES) $(DEF_FILE) $(LLAMA_LIBS) $(WHISPER_LIBS) $(MINIAUDIO_LIBS)
ifeq ($(USE_MSVC),1)
	link.exe /nologo $(LDFLAGS) /OUT:$@ $(OBJ_FILES) $(LLAMA_LIBS) $(WHISPER_LIBS) $(MINIAUDIO_LIBS) $(MSVC_LIBS)
else
	$(CXX) $(OBJ_FILES) $(DEF_FILE) -o $@ $(LDFLAGS)
ifeq ($(PLATFORM),windows)
	# Generate import library for Windows
	dlltool -D $@ -d $(DEF_FILE) -l $(DIST_DIR)/ai.lib
endif
endif
	# Strip debug symbols
	$(STRIP)

# Object files
$(BUILD_DIR)/%$(OBJ_EXT): %.c
ifeq ($(USE_MSVC),1)
	$(CC) $(CFLAGS) /O2 /c $< /Fo:$@
else
	$(CC) $(CFLAGS) -O3 -fPIC -c $< -o $@
endif

test: $(TARGET)
	$(SQLITE3) ":memory:" -cmd ".bail on" ".load ./dist/ai" "SELECT ai_version();"

# Build submodules
ifeq ($(PLATFORM),windows)
    ifneq (,$(findstring Ninja,$(LLAMA)))
        ARGS = -j $(CPUS)
    else
        ARGS = --parallel $(CPUS)
    endif
else
    ARGS = -- -j$(CPUS)
endif
build/llama.cpp.stamp:
	cmake -B $(BUILD_LLAMA) $(LLAMA_OPTIONS) $(LLAMA_DIR)
	cmake --build $(BUILD_LLAMA) --config Release $(LLAMA_ARGS) $(ARGS)
	touch $@

build/whisper.cpp.stamp:
	cmake -B $(BUILD_WHISPER) $(WHISPER_OPTIONS) $(WHISPER_DIR)
	cmake --build $(BUILD_WHISPER) --config Release $(WHISPER_ARGS) $(ARGS)
	touch $@

build/miniaudio.stamp:
	cmake -B $(BUILD_MINIAUDIO) $(MINIAUDIO_OPTIONS) $(MINIAUDIO_DIR)
	cmake --build $(BUILD_MINIAUDIO) --config Release $(MINIAUDIO_ARGS) $(ARGS)
	touch $@

$(LLAMA_LIBS): build/llama.cpp.stamp
$(WHISPER_LIBS): build/whisper.cpp.stamp
$(MINIAUDIO_LIBS): build/miniaudio.stamp

# Tools
version:
	@echo $(shell sed -n 's/^#define SQLITE_AI_VERSION[[:space:]]*"\([^"]*\)".*/\1/p' src/sqlite-ai.h)

# Clean up generated files
clean:
	rm -rf $(BUILD_DIR)/* $(DIST_DIR)/* *.gcda *.gcno *.gcov *.sqlite

# Help message
help:
	@echo "SQLite AI Extension Makefile"
	@echo "Usage:"
	@echo "  make [PLATFORM=platform] [ARCH=arch] [ANDROID_NDK=\$$ANDROID_HOME/ndk/26.1.10909125] [target] [LLAMA=options] [WHISPER=options] [MINIAUDIO=options]"
	@echo ""
	@echo "Platforms:"
	@echo "  linux (default on Linux)"
	@echo "  macos (default on macOS)"
	@echo "  windows (default on Windows)"
	@echo "  android (needs ARCH to be set to x86_64 or arm64-v8a and ANDROID_NDK to be set)"
	@echo "  ios (only on macOS)"
	@echo "  isim (only on macOS)"
	@echo ""
	@echo "Targets:"
	@echo "  all					- Build the extension (default)"
	@echo "  clean					- Remove built files"
	@echo "  test					- Test the extension"
	@echo "  help					- Display this help message"

.PHONY: all clean test extension help