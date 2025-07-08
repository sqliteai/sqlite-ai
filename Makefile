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
LDFLAGS = -L./$(BUILD_DIR)/lib/common -L./$(BUILD_DIR)/lib/ggml/src -L./$(BUILD_DIR)/lib/src -lcommon -lggml -lggml-base -lggml-cpu -lllama
LLAMA_OPTIONS = -DLLAMA_CURL=OFF

# Directories
SRC_DIR = src
DIST_DIR = dist
VPATH = $(SRC_DIR)
BUILD_DIR = build
LLAMA_DIR = modules/llama.cpp

# Files
SRC_FILES = $(wildcard $(SRC_DIR)/*.c)
OBJ_FILES = $(patsubst %.c, $(BUILD_DIR)/%.o, $(notdir $(SRC_FILES)))
LIBS = $(BUILD_DIR)/lib/common/libcommon.a \
	   $(BUILD_DIR)/lib/ggml/src/libggml.a \
	   $(BUILD_DIR)/lib/ggml/src/libggml-base.a \
	   $(BUILD_DIR)/lib/ggml/src/libggml-cpu.a \
	   $(BUILD_DIR)/lib/src/libllama.a

# Platform-specific settings
ifeq ($(PLATFORM),windows)
    TARGET := $(DIST_DIR)/ai.dll
    LDFLAGS += -shared
    # Create .def file for Windows
    DEF_FILE := $(BUILD_DIR)/ai.def
else ifeq ($(PLATFORM),macos)
    TARGET := $(DIST_DIR)/ai.dylib
    LIBS += $(BUILD_DIR)/lib/ggml/src/ggml-metal/libggml-metal.a $(BUILD_DIR)/lib/ggml/src/ggml-blas/libggml-blas.a
    LDFLAGS += -arch x86_64 -arch arm64 -L./$(BUILD_DIR)/lib/ggml/src/ggml-metal -lggml-metal -L./$(BUILD_DIR)/lib/ggml/src/ggml-blas -lggml-blas -framework Metal -framework Foundation -framework CoreFoundation -framework QuartzCore -framework Accelerate -dynamiclib -undefined dynamic_lookup
    CFLAGS += -arch x86_64 -arch arm64
    LLAMA_OPTIONS += -DCMAKE_OSX_ARCHITECTURES="x86_64;arm64"
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
    TARGET := $(DIST_DIR)/ai.so
    LDFLAGS += -shared
else ifeq ($(PLATFORM),ios)
    TARGET := $(DIST_DIR)/ai.dylib
    SDK := -isysroot $(shell xcrun --sdk iphoneos --show-sdk-path) -miphoneos-version-min=11.0
    LDFLAGS += -dynamiclib $(SDK)
    CFLAGS += -arch arm64 $(SDK)
    LLAMA_OPTIONS += -DCMAKE_SYSTEM_NAME=iOS
else ifeq ($(PLATFORM),isim)
    TARGET := $(DIST_DIR)/ai.dylib
    SDK := -isysroot $(shell xcrun --sdk iphonesimulator --show-sdk-path) -miphonesimulator-version-min=11.0
    LDFLAGS += -arch x86_64 -arch arm64 -dynamiclib $(SDK)
    CFLAGS += -arch x86_64 -arch arm64 $(SDK)
    LLAMA_OPTIONS += -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_ARCHITECTURES="x86_64;arm64"
else # linux
    TARGET := $(DIST_DIR)/ai.so
    LDFLAGS += -shared
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
$(TARGET): $(OBJ_FILES) $(DEF_FILE) $(LIBS)
	$(CXX) $(OBJ_FILES) $(DEF_FILE) -o $@ $(LDFLAGS)
ifeq ($(PLATFORM),windows)
    # Generate import library for Windows
	dlltool -D $@ -d $(DEF_FILE) -l $(DIST_DIR)/ai.lib
endif

# Object files
$(BUILD_DIR)/%.o: %.c
	$(CC) $(CFLAGS) -O3 -fPIC -c $< -o $@

test: $(TARGET)
	$(SQLITE3) ":memory:" -cmd ".bail on" ".load ./$<" "SELECT ai_version();"

# Build all libraries at once using one CMake call
build/libs.stamp:
	cmake -B $(BUILD_DIR)/lib -DBUILD_SHARED_LIBS=OFF $(LLAMA_OPTIONS) $(LLAMA_DIR)
	cmake --build $(BUILD_DIR)/lib --config Release -- -j$(CPUS)
	touch $@

$(LIBS): build/libs.stamp

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
