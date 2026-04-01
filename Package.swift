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
            url: "https://github.com/sqliteai/sqlite-ai/releases/download/1.0.3/ai-apple-xcframework-1.0.3.zip",
            checksum: "f33dfa4ab3538a6ac201d565410ce4bf4c2fc59aa31376102c1e9e9261f5e811"
        ),
        .target(
            name: "ai",
            dependencies: ["aiBinary"],
            path: "packages/swift"
        ),
    ]
)
