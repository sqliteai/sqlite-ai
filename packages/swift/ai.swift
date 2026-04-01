// ai.swift
// Provides the path to the ai SQLite extension for use with sqlite3_load_extension.

import Foundation

public struct ai {
    /// Returns the absolute path to the ai dylib for use with sqlite3_load_extension.
    public static var path: String {
        #if os(macOS)
        return Bundle.main.bundlePath + "/Contents/Frameworks/ai.framework/ai"
        #else
        return Bundle.main.bundlePath + "/Frameworks/ai.framework/ai"
        #endif
    }
}
