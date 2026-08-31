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
            url: "https://github.com/sqliteai/sqlite-ai/releases/download/1.0.8/ai-apple-xcframework-1.0.8.zip",
            checksum: "4a29926a603d57aebb4cde6a8026a103d54e1dfea4d659b3983fca40b1e70bf5"
        ),
        .target(
            name: "ai",
            dependencies: ["aiBinary"],
            path: "packages/swift"
        ),
    ]
)
