// ai.swift
// This file serves as a placeholder for the ai target.
// The actual SQLite extension is built using the Makefile through the build plugin.

import Foundation

/// Placeholder structure for ai
public struct ai {
    /// Returns the path to the built ai dylib inside the XCFramework
    public static var path: String {
        #if os(macOS)
        return "ai.xcframework/macos-arm64_x86_64/ai.framework/ai"
        #elseif targetEnvironment(simulator)
        return "ai.xcframework/ios-arm64_x86_64-simulator/ai.framework/ai"
        #else
        return "ai.xcframework/ios-arm64/ai.framework/ai"
        #endif
    }
}