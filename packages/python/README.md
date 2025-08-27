## SQLite AI Python package

This package provides the sqlite-ai extension prebuilt binaries for multiple platforms and architectures.

### SQLite-AI

SQLite-AI is an extension for SQLite that brings artificial intelligence capabilities directly into the database. It enables developers to run, fine-tune, and serve AI models from within SQLite using simple SQL queries — ideal for on-device and edge applications where low-latency and offline inference are critical. The extension is actively developed by [SQLite AI](https://sqlite.ai), some API and features are still evolving.

More details on the official repository [sqliteai/sqlite-ai](https://github.com/sqliteai/sqlite-ai).

### Documentation

For detailed information on all available functions, their parameters, and examples, refer to the [comprehensive API Reference](https://github.com/sqliteai/sqlite-ai/blob/main/API.md).

### Supported Platforms and Architectures

| Platform      | Arch         | Subpackage name         | Binary name |
| ------------- | ------------ | ----------------------- | ----------- |
| Linux (CPU)   | x86_64/arm64 | sqliteai.binaries.cpu   | ai.so       |
| Linux (GPU)   | x86_64/arm64 | sqliteai.binaries.gpu   | ai.so       |
| Windows (CPU) | x86_64       | sqliteai.binaries.cpu   | ai.dll      |
| Windows (GPU) | x86_64       | sqliteai.binaries.gpu   | ai.dll      |
| macOS (CPU)   | x86_64/arm64 | sqliteai.binaries.cpu   | ai.dylib    |

## Usage

> **Note:** Some SQLite installations on certain operating systems may have extension loading disabled by default.   
If you encounter issues loading the extension, refer to the [sqlite-extensions-guide](https://github.com/sqliteai/sqlite-extensions-guide/) for platform-specific instructions on enabling and using SQLite extensions.

```python
import importlib.resources
import sqlite3

# Connect to your SQLite database
conn = sqlite3.connect("example.db")

# Load the sqlite-ai extension
# pip will install the correct binary package for your platform and architecture
# Choose between CPU or GPU variant
ext_path = importlib.resources.files("sqliteai.binaries.cpu") / "ai"

conn.enable_load_extension(True)
conn.load_extension(str(ext_path))
conn.enable_load_extension(False)


# Now you can use sqlite-ai features in your SQL queries
print(conn.execute("SELECT ai_version();").fetchone())
```