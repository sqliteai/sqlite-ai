// swift-tools-version: 6.1
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "ai",
    platforms: [.macOS(.v11), .iOS(.v11)],
    products: [
        .library(
            name: "ai",
            targets: ["ai"])
    ],
    targets: [
        .binaryTarget(
            name: "aiBinary",
            url: "https://github.com/sqliteai/sqlite-ai/releases/download/1.0.4/ai-apple-xcframework-1.0.4.zip",
            checksum: "c8d2928f9bbeed9a5530cf2a0a8bc352d69b2febaf48697de690e9a305f782e5"
        ),
        .target(
            name: "ai",
            dependencies: ["aiBinary"],
            path: "packages/swift"
        ),
    ]
)
