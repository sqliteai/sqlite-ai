import PackagePlugin
import Foundation

@main
struct ai: BuildToolPlugin {
    /// Entry point for creating build commands for targets in Swift packages.
    func createBuildCommands(context: PluginContext, target: Target) async throws -> [Command] {
        let packageDirectory = context.package.directoryURL
        let outputDirectory = context.pluginWorkDirectoryURL
        return createaiBuildCommands(packageDirectory: packageDirectory, outputDirectory: outputDirectory)
    }
}

#if canImport(XcodeProjectPlugin)
import XcodeProjectPlugin

extension ai: XcodeBuildToolPlugin {
    // Entry point for creating build commands for targets in Xcode projects.
    func createBuildCommands(context: XcodePluginContext, target: XcodeTarget) throws -> [Command] {
        let outputDirectory = context.pluginWorkDirectoryURL
        return createaiBuildCommands(packageDirectory: nil, outputDirectory: outputDirectory)
    }
}

#endif

/// Shared function to create ai build commands
func createaiBuildCommands(packageDirectory: URL?, outputDirectory: URL) -> [Command] {

    // For Xcode projects, use current directory; for Swift packages, use provided packageDirectory
    let workingDirectory = packageDirectory?.path ?? "$(pwd)"
    let packageDirInfo = packageDirectory != nil ? "Package directory: \(packageDirectory!.path)" : "Working directory: $(pwd)"

    return [
        .prebuildCommand(
            displayName: "Building ai XCFramework",
            executable: URL(fileURLWithPath: "/bin/bash"),
            arguments: [
                "-c",
                """
                set -e
                echo "Starting ai XCFramework prebuild..."
                echo "\(packageDirInfo)"
                
                # Clean and create output directory
                rm -rf "\(outputDirectory.path)"
                mkdir -p "\(outputDirectory.path)"
                
                # Copy source to writable location first due to sandbox restrictions
                cp -R "\(workingDirectory)" "\(outputDirectory.path)/src" && \
                cd "\(outputDirectory.path)/src" && \
                echo "Building XCFramework..." && \
                PATH="/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/opt/local/bin:/usr/sbin:/sbin:$PATH" make xcframework DIST_DIR="\(outputDirectory.path)" LLAMA="-DGGML_NATIVE=OFF -DGGML_METAL=ON -DGGML_ACCELERATE=ON -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=Apple" WHISPER="-DWHISPER_COREML=ON -DWHISPER_COREML_ALLOW_FALLBACK=ON" && \
                rm -rf "\(outputDirectory.path)/src" && \
                echo "XCFramework build completed successfully!"
                """
            ],
            outputFilesDirectory: outputDirectory
        )
    ]
}