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
CTEST_BIN = $(BUILD_DIR)/tests/sqlite_ai_tests
GGUF_MODEL_DIR ?= tests/models/unsloth/gemma-3-270m-it-GGUF
GGUF_MODEL_NAME ?= gemma-3-270m-it-UD-IQ2_M.gguf
GGUF_MODEL_URL ?= https://huggingface.co/unsloth/gemma-3-270m-it-GGUF/resolve/main/gemma-3-270m-it-UD-IQ2_M.gguf
GGUF_MODEL_PATH := $(GGUF_MODEL_DIR)/$(GGUF_MODEL_NAME)
SKIP_UNITTEST ?= 0
LLAMA_DIR = modules/llama.cpp
WHISPER_DIR = modules/whisper.cpp
MINIAUDIO_DIR = modules/miniaudio
BUILD_GGML = $(BUILD_DIR)/ggml
BUILD_LLAMA = $(BUILD_DIR)/llama.cpp
BUILD_WHISPER = $(BUILD_DIR)/whisper.cpp
BUILD_MINIAUDIO = $(BUILD_DIR)/miniaudio

# Compiler and flags
CC = gcc
CXX = g++
CFLAGS = -Wall -Wextra -Wno-unused-parameter -I$(SRC_DIR) -I$(BUILD_GGML)/include -I$(WHISPER_DIR)/include -I$(MINIAUDIO_DIR)
LLAMA_OPTIONS = $(LLAMA) -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF -DGGML_RPC=OFF
WHISPER_OPTIONS = $(LLAMA) $(WHISPER) -DBUILD_SHARED_LIBS=OFF -DWHISPER_BUILD_EXAMPLES=OFF -DWHISPER_BUILD_TESTS=OFF -DWHISPER_BUILD_SERVER=OFF -DWHISPER_RPC=OFF -DWHISPER_USE_SYSTEM_GGML=ON
MINIAUDIO_OPTIONS = $(MINIAUDIO) -DBUILD_SHARED_LIBS=OFF -DMINIAUDIO_BUILD_EXAMPLES=OFF -DMINIAUDIO_BUILD_TESTS=OFF
# MinGW produces .a files without lib prefix, use -l:filename.a syntax
ifeq ($(PLATFORM),windows)
	L = -l:
	A = .a
else
	L = -l
endif
LLAMA_LDFLAGS = -L./$(BUILD_LLAMA)/common -L./$(BUILD_GGML)/lib -L./$(BUILD_LLAMA)/src -lcommon -lllama $(L)ggml$(A) $(L)ggml-base$(A) $(L)ggml-cpu$(A)
WHISPER_LDFLAGS = -L./$(BUILD_WHISPER)/src -lwhisper
MINIAUDIO_LDFLAGS = -L./$(BUILD_MINIAUDIO) -lminiaudio -lminiaudio_channel_combiner_node -lminiaudio_channel_separator_node -lminiaudio_ltrim_node -lminiaudio_reverb_node -lminiaudio_vocoder_node
LDFLAGS = $(LLAMA_LDFLAGS) $(WHISPER_LDFLAGS) $(MINIAUDIO_LDFLAGS)
SQLITE_TEST_LIBS =
ifneq ($(PLATFORM),windows)
	SQLITE_TEST_LIBS += -lpthread -lm
	ifneq ($(PLATFORM),macos)
		SQLITE_TEST_LIBS += -ldl
	endif
endif
SQLITE_TEST_SRC = tests/c/sqlite3.c

# Files
SRC_FILES = $(wildcard $(SRC_DIR)/*.c)
OBJ_FILES = $(patsubst %.c, $(BUILD_DIR)/%.o, $(notdir $(SRC_FILES)))
LLAMA_LIBS = $(BUILD_LLAMA)/common/libcommon.a $(BUILD_GGML)/libggml.a $(BUILD_GGML)/libggml-base.a $(BUILD_GGML)/libggml-cpu.a $(BUILD_LLAMA)/src/libllama.a
WHISPER_LIBS = $(BUILD_WHISPER)/src/libwhisper.a
MINIAUDIO_LIBS = $(BUILD_MINIAUDIO)/libminiaudio.a

# Platform-specific settings
ifeq ($(PLATFORM),windows)
	TARGET := $(DIST_DIR)/ai.dll
	LDFLAGS += -lbcrypt -static-libgcc -Wl,--push-state,-Bstatic,-lgomp,-lstdc++,-lwinpthread,--pop-state -shared
	DEF_FILE := $(BUILD_DIR)/ai.def
	STRIP = strip --strip-unneeded $@
else ifeq ($(PLATFORM),macos)
	TARGET := $(DIST_DIR)/ai.dylib
	LLAMA_LIBS += $(BUILD_GGML)/lib/libggml-metal.a
	ifndef ARCH
		LDFLAGS += -arch x86_64 -arch arm64
		CFLAGS += -arch x86_64 -arch arm64
		LLAMA_OPTIONS += -DGGML_OPENMP=OFF -DCMAKE_OSX_ARCHITECTURES="x86_64;arm64" -DCMAKE_OSX_DEPLOYMENT_TARGET=11.0
		WHISPER_OPTIONS += -DGGML_OPENMP=OFF -DCMAKE_OSX_ARCHITECTURES="x86_64;arm64" -DCMAKE_OSX_DEPLOYMENT_TARGET=11.0
		MINIAUDIO_OPTIONS += -DCMAKE_OSX_ARCHITECTURES="x86_64;arm64" -DCMAKE_OSX_DEPLOYMENT_TARGET=11.0
	else
		LDFLAGS += -arch $(ARCH)
		CFLAGS += -arch $(ARCH)
		LLAMA_OPTIONS += -DGGML_OPENMP=OFF -DCMAKE_OSX_ARCHITECTURES="$(ARCH)" -DCMAKE_OSX_DEPLOYMENT_TARGET=11.0
		WHISPER_OPTIONS += -DGGML_OPENMP=OFF -DCMAKE_OSX_ARCHITECTURES="$(ARCH)" -DCMAKE_OSX_DEPLOYMENT_TARGET=11.0
		MINIAUDIO_OPTIONS += -DCMAKE_OSX_ARCHITECTURES="$(ARCH)" -DCMAKE_OSX_DEPLOYMENT_TARGET=11.0
	endif
	LDFLAGS += -L./$(BUILD_GGML)/lib -lggml-metal -L./$(BUILD_GGML)/lib -framework Metal -framework Foundation -framework CoreFoundation -framework QuartzCore -dynamiclib -undefined dynamic_lookup -headerpad_max_install_names
	STRIP = strip -x -S $@
else ifeq ($(PLATFORM),android)
	ifndef ARCH # Set ARCH to find Android NDK's Clang compiler, the user should set the ARCH
		$(error "Android ARCH must be set to ARCH=x86_64 or ARCH=arm64-v8a")
	endif
	ifndef ANDROID_NDK # Set ANDROID_NDK path to find android build tools; e.g. on MacOS: export ANDROID_NDK=/Users/username/Library/Android/sdk/ndk/25.2.9519653
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
	ANDROID_OPTIONS = -DCMAKE_TOOLCHAIN_FILE=$(ANDROID_NDK)/build/cmake/android.toolchain.cmake -DANDROID_ABI=$(if $(filter aarch64,$(ARCH)),arm64-v8a,$(ARCH)) -DANDROID_PLATFORM=android-26 -DCMAKE_C_FLAGS="-march=$(if $(filter aarch64,$(ARCH)),armv8.7a,x86-64)" -DCMAKE_CXX_FLAGS="-march=$(if $(filter aarch64,$(ARCH)),armv8.7a,x86-64)" -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DGGML_OPENMP=OFF -DGGML_LLAMAFILE=OFF
	ifneq (,$(filter $(ARCH),aarch64))
		ANDROID_OPTIONS += -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod
	endif
	LLAMA_OPTIONS += $(ANDROID_OPTIONS)
	WHISPER_OPTIONS += $(ANDROID_OPTIONS)
	MINIAUDIO_OPTIONS += $(ANDROID_OPTIONS)
	STRIP = $(BIN)/llvm-strip --strip-unneeded $@
else ifeq ($(PLATFORM),ios)
	TARGET := $(DIST_DIR)/ai.dylib
	SDK := -isysroot $(shell xcrun --sdk iphoneos --show-sdk-path) -miphoneos-version-min=14.0
	LLAMA_LIBS += $(BUILD_GGML)/lib/libggml-metal.a
	LDFLAGS += -L./$(BUILD_GGML)/lib -lggml-metal -L./$(BUILD_GGML)/lib -framework Accelerate -framework Metal -framework Foundation -framework AVFoundation -framework AudioToolbox -framework CoreAudio -framework CoreFoundation -framework Security -ldl -lpthread -lm -dynamiclib $(SDK) -headerpad_max_install_names
	CFLAGS += -arch arm64 -x objective-c $(SDK)
	LLAMA_OPTIONS += -DGGML_OPENMP=OFF -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0
	WHISPER_OPTIONS += -DGGML_OPENMP=OFF -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0
	MINIAUDIO_OPTIONS += -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0 -DMINIAUDIO_NO_RUNTIME_LINKING=ON
	STRIP = strip -x -S $@
else ifeq ($(PLATFORM),ios-sim)
	TARGET := $(DIST_DIR)/ai.dylib
	SDK := -isysroot $(shell xcrun --sdk iphonesimulator --show-sdk-path) -miphonesimulator-version-min=14.0
	LLAMA_LIBS += $(BUILD_GGML)/lib/libggml-metal.a
	LDFLAGS += -arch x86_64 -arch arm64 -L./$(BUILD_GGML)/lib -lggml-metal -L./$(BUILD_GGML)/lib -framework Accelerate -framework Metal -framework Foundation -framework AVFoundation -framework AudioToolbox -framework CoreAudio -framework CoreFoundation -framework Security -ldl -lpthread -lm -dynamiclib $(SDK) -headerpad_max_install_names
	CFLAGS += -arch x86_64 -arch arm64 -x objective-c $(SDK)
	LLAMA_OPTIONS += -DGGML_OPENMP=OFF -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_SYSROOT=iphonesimulator -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0 -DCMAKE_OSX_ARCHITECTURES="x86_64;arm64"
	WHISPER_OPTIONS += -DGGML_OPENMP=OFF -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_SYSROOT=iphonesimulator -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0 -DCMAKE_OSX_ARCHITECTURES="x86_64;arm64"
	MINIAUDIO_OPTIONS += -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_SYSROOT=iphonesimulator -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0 -DCMAKE_OSX_ARCHITECTURES="x86_64;arm64" -DMINIAUDIO_NO_RUNTIME_LINKING=ON
	STRIP = strip -x -S $@
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
	LLAMA_LIBS += $(BUILD_GGML)/lib/libggml-vulkan.a
	LLAMA_LDFLAGS += -L./$(BUILD_GGML)/lib $(L)ggml-vulkan$(A)
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
	LLAMA_LIBS += $(BUILD_GGML)/lib/libggml-opencl.a
	LLAMA_LDFLAGS += -L./$(BUILD_GGML)/lib $(L)ggml-opencl$(A) -lOpenCL
	ifneq ($(PLATFORM),windows)
		LLAMA_LDFLAGS += -ldl
	endif
endif
ifneq (,$(findstring BLAS,$(LLAMA)))
	LLAMA_LIBS += $(BUILD_GGML)/lib/libggml-blas.a
	LLAMA_LDFLAGS += -L./$(BUILD_GGML)/lib $(L)ggml-blas$(A)
	# Link against specific BLAS implementations
	ifneq (,$(findstring OpenBLAS,$(LLAMA)))
		LLAMA_LDFLAGS += -lopenblas
	else ifneq (,$(findstring Apple,$(LLAMA)))
		LDFLAGS += -framework Accelerate
	else # Generic BLAS
		LLAMA_LDFLAGS += -lblas
	endif
endif
ifneq (,$(findstring COREML,$(WHISPER))) # CoreML - only Apple platforms
	WHISPER_LIBS += $(BUILD_WHISPER)/src/libwhisper.coreml.a
	WHISPER_LDFLAGS += -lwhisper.coreml
	WHISPER_OPTIONS += -DWHISPER_COREML=ON
	LDFLAGS += -framework CoreML
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
$(TARGET): $(OBJ_FILES) $(DEF_FILE) $(LLAMA_LIBS) $(WHISPER_LIBS) $(MINIAUDIO_LIBS)
	$(CXX) $(OBJ_FILES) $(DEF_FILE) -o $@ $(LDFLAGS)
ifeq ($(PLATFORM),windows)
	# Generate import library for Windows
	dlltool -D $@ -d $(DEF_FILE) -l $(DIST_DIR)/ai.lib
endif
	# Strip debug symbols
	$(STRIP)

# Object files
$(BUILD_DIR)/%.o: %.c $(BUILD_DIR)/llama.cpp.stamp
	$(CC) $(CFLAGS) -O3 -fPIC -c $< -o $@

$(CTEST_BIN): tests/c/unittest.c $(SQLITE_TEST_SRC)
	@mkdir -p $(dir $@)
	$(CC) -std=c11 -Wall -Wextra -DSQLITE_ENABLE_LOAD_EXTENSION -I$(SRC_DIR) tests/c/unittest.c $(SQLITE_TEST_SRC) -o $@ $(SQLITE_TEST_LIBS)

$(GGUF_MODEL_PATH):
	@mkdir -p $(GGUF_MODEL_DIR)
	curl -L --fail --retry 3 -o $@ $(GGUF_MODEL_URL)

TEST_DEPS := $(TARGET)
ifeq ($(SKIP_UNITTEST),0)
TEST_DEPS += $(CTEST_BIN) $(GGUF_MODEL_PATH)
endif

test: $(TEST_DEPS)
		@echo "Running sqlite3 CLI smoke test (ensures .load works)..."
		$(SQLITE3) ":memory:" -cmd ".bail on" ".load ./dist/ai" "SELECT ai_version();"
ifeq ($(SKIP_UNITTEST),0)
		$(CTEST_BIN) --extension "$(TARGET)" --model "$(GGUF_MODEL_PATH)"
else
		@echo "Skipping C unit tests (SKIP_UNITTEST=$(SKIP_UNITTEST))."
endif

# Build submodules
ifeq ($(PLATFORM),windows)
    ARGS = --parallel $(CPUS)
else
    ARGS = -- -j$(CPUS)
endif
$(BUILD_DIR)/llama.cpp.stamp:
	cmake -B $(BUILD_LLAMA) $(LLAMA_OPTIONS) $(LLAMA_DIR)
	cmake --build $(BUILD_LLAMA) --config Release $(LLAMA_ARGS) $(ARGS)
	cmake --install $(BUILD_LLAMA) --prefix $(BUILD_GGML)
	touch $@

$(BUILD_DIR)/whisper.cpp.stamp: $(BUILD_DIR)/llama.cpp.stamp
	cmake -Dggml_DIR=$(shell pwd)/$(BUILD_GGML)/lib/cmake/ggml -B $(BUILD_WHISPER) $(WHISPER_OPTIONS) $(WHISPER_DIR)
	cmake --build $(BUILD_WHISPER) --config Release $(WHISPER_ARGS) $(ARGS)
	touch $@

$(BUILD_DIR)/miniaudio.stamp:
	cmake -B $(BUILD_MINIAUDIO) $(MINIAUDIO_OPTIONS) $(MINIAUDIO_DIR)
ifeq ($(PLATFORM),ios)
	# Patch the build files to add Objective-C flag for iOS
	sed -i.bak 's/\(C_FLAGS = \)/\1-x objective-c /' $(BUILD_MINIAUDIO)/CMakeFiles/miniaudio.dir/flags.make
	sed -i.bak 's/\(C_FLAGS = \)/\1-x objective-c /' $(BUILD_MINIAUDIO)/CMakeFiles/miniaudio_*.dir/flags.make || true
endif
ifeq ($(PLATFORM),ios-sim)
	# Patch the build files to add Objective-C flag for iOS simulator
	sed -i.bak 's/\(C_FLAGS = \)/\1-x objective-c /' $(BUILD_MINIAUDIO)/CMakeFiles/miniaudio.dir/flags.make
	sed -i.bak 's/\(C_FLAGS = \)/\1-x objective-c /' $(BUILD_MINIAUDIO)/CMakeFiles/miniaudio_*.dir/flags.make || true
endif
	cmake --build $(BUILD_MINIAUDIO) --config Release $(MINIAUDIO_ARGS) $(ARGS)
	touch $@

$(LLAMA_LIBS): $(BUILD_DIR)/llama.cpp.stamp
$(WHISPER_LIBS): $(BUILD_DIR)/whisper.cpp.stamp
$(MINIAUDIO_LIBS): $(BUILD_DIR)/miniaudio.stamp

# Tools
version:
	@echo $(shell sed -n 's/^#define SQLITE_AI_VERSION[[:space:]]*"\([^"]*\)".*/\1/p' src/sqlite-ai.h)

# Clean up generated files
clean:
	rm -rf $(BUILD_DIR)/* $(DIST_DIR)/* *.gcda *.gcno *.gcov *.sqlite

.NOTPARALLEL: %.dylib
%.dylib:
	rm -rf $(BUILD_DIR) && $(MAKE) PLATFORM=$*
	mv $(DIST_DIR)/ai.dylib $(DIST_DIR)/$@

define PLIST
<?xml version=\"1.0\" encoding=\"UTF-8\"?>\
<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\
<plist version=\"1.0\">\
<dict>\
<key>CFBundleDevelopmentRegion</key>\
<string>en</string>\
<key>CFBundleExecutable</key>\
<string>ai</string>\
<key>CFBundleIdentifier</key>\
<string>ai.sqlite.ai</string>\
<key>CFBundleInfoDictionaryVersion</key>\
<string>6.0</string>\
<key>CFBundlePackageType</key>\
<string>FMWK</string>\
<key>CFBundleSignature</key>\
<string>????</string>\
<key>CFBundleVersion</key>\
<string>$(shell make version)</string>\
<key>CFBundleShortVersionString</key>\
<string>$(shell make version)</string>\
<key>MinimumOSVersion</key>\
<string>11.0</string>\
</dict>\
</plist>
endef

define MODULEMAP
framework module ai {\
  umbrella header \"sqlite-ai.h\"\
  export *\
}
endef

LIB_NAMES = ios.dylib ios-sim.dylib macos.dylib
FMWK_NAMES = ios-arm64 ios-arm64_x86_64-simulator macos-arm64_x86_64
$(DIST_DIR)/%.xcframework: $(LIB_NAMES)
	@$(foreach i,1 2 3,\
		lib=$(word $(i),$(LIB_NAMES)); \
		fmwk=$(word $(i),$(FMWK_NAMES)); \
		mkdir -p $(DIST_DIR)/$$fmwk/ai.framework/Headers; \
		mkdir -p $(DIST_DIR)/$$fmwk/ai.framework/Modules; \
		cp src/sqlite-ai.h $(DIST_DIR)/$$fmwk/ai.framework/Headers; \
		printf "$(PLIST)" > $(DIST_DIR)/$$fmwk/ai.framework/Info.plist; \
		printf "$(MODULEMAP)" > $(DIST_DIR)/$$fmwk/ai.framework/Modules/module.modulemap; \
		mv $(DIST_DIR)/$$lib $(DIST_DIR)/$$fmwk/ai.framework/ai; \
		install_name_tool -id "@rpath/ai.framework/ai" $(DIST_DIR)/$$fmwk/ai.framework/ai; \
	)
	xcodebuild -create-xcframework $(foreach fmwk,$(FMWK_NAMES),-framework $(DIST_DIR)/$(fmwk)/ai.framework) -output $@
	rm -rf $(foreach fmwk,$(FMWK_NAMES),$(DIST_DIR)/$(fmwk))

xcframework: $(DIST_DIR)/ai.xcframework

AAR_ARM = packages/android/src/main/jniLibs/arm64-v8a/
AAR_X86 = packages/android/src/main/jniLibs/x86_64/
aar:
	mkdir -p $(AAR_ARM) $(AAR_X86)
	$(MAKE) clean && $(MAKE) PLATFORM=android ARCH=arm64-v8a
	mv $(DIST_DIR)/ai.so $(AAR_ARM)
	$(MAKE) clean && $(MAKE) PLATFORM=android ARCH=x86_64
	mv $(DIST_DIR)/ai.so $(AAR_X86)
	cd packages/android && ./gradlew clean assembleRelease
	cp packages/android/build/outputs/aar/android-release.aar $(DIST_DIR)/ai.aar

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
	@echo "  ios-sim (only on macOS)"
	@echo ""
	@echo "Targets:"
	@echo "  all			- Build the extension (default)"
	@echo "  clean			- Remove built files"
	@echo "  test			- Test the extension"
	@echo "  help			- Display this help message"
	@echo "  xcframework	- Build the Apple XCFramework"
	@echo "  aar				- Build the Android AAR package"

.PHONY: all clean test extension help version xcframework aar
