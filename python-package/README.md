## SQLite AI Python package

This package provides the sqlite-ai extension prebuilt binaries for multiple platforms and architectures.

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
# Pip will download the right package for your platform/architecture
# Choose between CPU or GPU backend
ext_path = importlib.resources.files("sqliteai.binaries.cpu") / "ai"

conn.enable_load_extension(True)
conn.load_extension(str(ext_path))
conn.enable_load_extension(False)


# Now you can use sqlite-ai features in your SQL queries
print(conn.execute("SELECT ai_version();").fetchone())
```