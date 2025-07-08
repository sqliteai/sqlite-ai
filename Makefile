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

# Compiler and flags
CC = gcc
CXX = g++
CFLAGS = -Wall -Wextra -Wno-unused-parameter -I$(SRC_DIR) -I$(LLAMA_DIR)/ggml/include -I$(LLAMA_DIR)/include
LDFLAGS = -L./$(BUILD_LLAMA)/common -L./$(BUILD_LLAMA)/ggml/src -L./$(BUILD_LLAMA)/src -L./$(BUILD_WHISPER)/src -lcommon -lggml -lggml-base -lggml-cpu -lllama -lwhisper
LLAMA_OPTIONS = -DLLAMA_CURL=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF
WHISPER_OPTIONS = -DWHISPER_BUILD_EXAMPLES=OFF -DWHISPER_BUILD_TESTS=OFF -DWHISPER_BUILD_SERVER=OFF

# Directories
SRC_DIR = src
DIST_DIR = dist
VPATH = $(SRC_DIR)
BUILD_DIR = build
LLAMA_DIR = modules/llama.cpp
WHISPER_DIR = modules/whisper.cpp
BUILD_LLAMA = $(BUILD_DIR)/llama.cpp
BUILD_WHISPER = $(BUILD_DIR)/whisper.cpp

# Files
SRC_FILES = $(wildcard $(SRC_DIR)/*.c)
OBJ_FILES = $(patsubst %.c, $(BUILD_DIR)/%.o, $(notdir $(SRC_FILES)))
LLAMA_LIBS = $(BUILD_LLAMA)/common/libcommon.a $(BUILD_LLAMA)/ggml/src/libggml.a $(BUILD_LLAMA)/ggml/src/libggml-base.a $(BUILD_LLAMA)/ggml/src/libggml-cpu.a $(BUILD_LLAMA)/src/libllama.a
WHISPER_LIBS = $(BUILD_WHISPER)/src/libwhisper.a

# Platform-specific settings
ifeq ($(PLATFORM),windows)
    TARGET := $(DIST_DIR)/ai.dll
    LDFLAGS += -shared
    # Create .def file for Windows
    DEF_FILE := $(BUILD_DIR)/ai.def
else ifeq ($(PLATFORM),macos)
    TARGET := $(DIST_DIR)/ai.dylib
    LLAMA_LIBS += $(BUILD_LLAMA)/ggml/src/ggml-metal/libggml-metal.a $(BUILD_LLAMA)/ggml/src/ggml-blas/libggml-blas.a
    LDFLAGS += -arch x86_64 -arch arm64 -L./$(BUILD_LLAMA)/ggml/src/ggml-metal -lggml-metal -L./$(BUILD_LLAMA)/ggml/src/ggml-blas -lggml-blas -framework Metal -framework Foundation -framework CoreFoundation -framework QuartzCore -framework Accelerate -dynamiclib -undefined dynamic_lookup
    CFLAGS += -arch x86_64 -arch arm64
    LLAMA_OPTIONS += -DCMAKE_OSX_ARCHITECTURES="x86_64;arm64"
    WHISPER_OPTIONS += -DCMAKE_OSX_ARCHITECTURES="x86_64;arm64"
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
    LDFLAGS += -shared
    LLAMA_OPTIONS += -DCMAKE_TOOLCHAIN_FILE=$(ANDROID_NDK)/build/cmake/android.toolchain.cmake -DANDROID_ABI=$(if $(filter aarch64,$(ARCH)),arm64-v8a,$(ARCH)) -DANDROID_PLATFORM=android-26 -DCMAKE_C_FLAGS="-march=$(if $(filter aarch64,$(ARCH)),armv8.7a,x86-64)" -DCMAKE_CXX_FLAGS="-march=$(if $(filter aarch64,$(ARCH)),armv8.7a,x86-64)" -DGGML_OPENMP=OFF -DGGML_LLAMAFILE=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    WHISPER_OPTIONS += -DCMAKE_TOOLCHAIN_FILE=$(ANDROID_NDK)/build/cmake/android.toolchain.cmake -DANDROID_ABI=$(if $(filter aarch64,$(ARCH)),arm64-v8a,$(ARCH)) -DANDROID_PLATFORM=android-26 -DCMAKE_C_FLAGS="-march=$(if $(filter aarch64,$(ARCH)),armv8.7a,x86-64)" -DCMAKE_CXX_FLAGS="-march=$(if $(filter aarch64,$(ARCH)),armv8.7a,x86-64)" -DGGML_OPENMP=OFF -DGGML_LLAMAFILE=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON

else ifeq ($(PLATFORM),ios)
    TARGET := $(DIST_DIR)/ai.dylib
    SDK := -isysroot $(shell xcrun --sdk iphoneos --show-sdk-path) -miphoneos-version-min=14.0
    LLAMA_LIBS += $(BUILD_LLAMA)/ggml/src/ggml-metal/libggml-metal.a $(BUILD_LLAMA)/ggml/src/ggml-blas/libggml-blas.a
    LDFLAGS += -L./$(BUILD_LLAMA)/ggml/src/ggml-metal -lggml-metal -L./$(BUILD_LLAMA)/ggml/src/ggml-blas -lggml-blas -framework Accelerate -framework Metal -framework Foundation -dynamiclib $(SDK)
    CFLAGS += -arch arm64 $(SDK)
    LLAMA_OPTIONS += -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0
    WHISPER_OPTIONS += -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0
else ifeq ($(PLATFORM),isim)
    TARGET := $(DIST_DIR)/ai.dylib
    SDK := -isysroot $(shell xcrun --sdk iphonesimulator --show-sdk-path) -miphonesimulator-version-min=14.0
    LLAMA_LIBS += $(BUILD_LLAMA)/ggml/src/ggml-metal/libggml-metal.a $(BUILD_LLAMA)/ggml/src/ggml-blas/libggml-blas.a
    LDFLAGS += -arch x86_64 -arch arm64 -L./$(BUILD_LLAMA)/ggml/src/ggml-metal -lggml-metal -L./$(BUILD_LLAMA)/ggml/src/ggml-blas -lggml-blas -framework Accelerate -framework Metal -framework Foundation -dynamiclib $(SDK)
    CFLAGS += -arch x86_64 -arch arm64 $(SDK)
    LLAMA_OPTIONS += -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_SYSROOT=iphonesimulator -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0 -DCMAKE_OSX_ARCHITECTURES="x86_64;arm64"
    WHISPER_OPTIONS += -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_SYSROOT=iphonesimulator -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0 -DCMAKE_OSX_ARCHITECTURES="x86_64;arm64"
else # linux
    TARGET := $(DIST_DIR)/ai.so
    LDFLAGS += -shared
    LLAMA_OPTIONS += -DGGML_OPENMP=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    WHISPER_OPTIONS += -DGGML_OPENMP=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON
endif

# Windows .def file generation
$(DEF_FILE):
ifeq ($(PLATFORM),windows)
	@echo "LIBRARY ai.dll" > $@
	@echo "EXPORTS" >> $@
	@echo "    sqlite3_ai_init" >> $@
endif

# Make sure the build and dist directories exist
$(shell mkdir -p $(BUILD_DIR) $(DIST_DIR))

# Default target
extension: $(TARGET)
all: $(TARGET) 

# Loadable library
$(TARGET): $(OBJ_FILES) $(DEF_FILE) $(LLAMA_LIBS) $(WHISPER_LIBS)
	$(CXX) $(OBJ_FILES) $(DEF_FILE) -o $@ $(LDFLAGS)
ifeq ($(PLATFORM),windows)
    # Generate import library for Windows
	dlltool -D $@ -d $(DEF_FILE) -l $(DIST_DIR)/ai.lib
endif

# Object files
$(BUILD_DIR)/%.o: %.c
	$(CC) $(CFLAGS) -O3 -fPIC -c $< -o $@

test: $(TARGET)
	$(SQLITE3) ":memory:" -cmd ".bail on" ".load ./dist/ai" "SELECT ai_version();"

# Build submodules
build/llama.cpp.stamp:
	cmake -B $(BUILD_LLAMA) -DBUILD_SHARED_LIBS=OFF $(LLAMA_OPTIONS) $(LLAMA_DIR)
	cmake --build $(BUILD_LLAMA) --config Release -- -j$(CPUS)
	touch $@

build/whisper.cpp.stamp:
	cmake -B $(BUILD_WHISPER) -DBUILD_SHARED_LIBS=OFF $(WHISPER_OPTIONS) $(WHISPER_DIR)
	cmake --build $(BUILD_WHISPER) --config Release -- -j$(CPUS)
	touch $@

$(LLAMA_LIBS): build/llama.cpp.stamp
$(WHISPER_LIBS): build/whisper.cpp.stamp

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
	@echo "  make [PLATFORM=platform] [ARCH=arch] [ANDROID_NDK=\$$ANDROID_HOME/ndk/26.1.10909125] [target]"
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
