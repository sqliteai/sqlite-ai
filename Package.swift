// swift-tools-version: 6.1
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "ai",
    platforms: [.macOS(.v11), .iOS(.v11)],
    products: [
        // Products can be used to vend plugins, making them visible to other packages.
        .plugin(
            name: "aiPlugin",
            targets: ["aiPlugin"]),
        .library(
            name: "ai",
            targets: ["ai"])
    ],
    targets: [
        // Build tool plugin that invokes the Makefile
        .plugin(
            name: "aiPlugin",
            capability: .buildTool(),
            path: "packages/swift/plugin"
        ),
        // ai library target
        .target(
            name: "ai",
            dependencies: [],
            path: "packages/swift/extension",
            plugins: ["aiPlugin"]
        ),
    ]
)