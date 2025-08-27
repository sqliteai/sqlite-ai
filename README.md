# SQLite-AI

**SQLite-AI** is an extension for SQLite that brings artificial intelligence capabilities directly into the database. It enables developers to run, fine-tune, and serve AI models from within SQLite using simple SQL queries — ideal for on-device and edge applications where low-latency and offline inference are critical. The extension is actively developed by [SQLite AI](https://sqlite.ai), some API and features are still evolving.

## Features

* **Embedded AI Inference**: Run transformer models directly from SQL queries.
* **Streaming I/O**: Token-by-token streaming via SQL aggregate functions.
* **Fine-tuning & Embedding**: On-device model customization and vector embedding.
* **Full On-Device Support**: Works on iOS, Android, Linux, macOS, and Windows.
* **Offline-First**: No server dependencies or internet connection required.
* **Composable SQL Interface**: AI + relational logic in a single unified layer.
* **Supports any GGUF model**: available on Huggingface; Qwen, Gemma, Llama, DeepSeek and more

SQLite-AI supports **text embedding generation** for search and classification, a **chat-like interface with history and token streaming**, and **automatic context save and restore** across sessions — making it ideal for building conversational agents and memory-aware assistants. Support for **multimodal** (sound and image understanding) is coming soon, bringing even richer on-device intelligence.

## Documentation

For detailed information on all available functions, their parameters, and examples, refer to the [comprehensive API Reference](./API.md).

## Installation

### Pre-built Binaries

Download the appropriate pre-built binary for your platform from the official [Releases](https://github.com/sqliteai/sqlite-ai/releases) page:

- Linux: x86 and ARM
- macOS: x86 and ARM
- Windows: x86
- Android
- iOS

### Loading the Extension

```sql
-- In SQLite CLI
.load ./ai

-- In SQL
SELECT load_extension('./ai');
```

### Python Package

Python developers can quickly get started using the ready-to-use `sqlite-ai` package available on PyPI:

```bash
pip install sqlite-ai
```

For usage details and examples, see the [Python package documentation](./packages/python/README.md).


## Getting Started

Here's a quick example to get started with SQLite Sync:

```bash
# Start SQLite CLI
sqlite3 myapp.db
```

```sql
-- Load the extension
.load ./ai

-- Load a model
SELECT llm_model_load('models/llama-2-7b.gguf', 'context_size=4096,n_gpu_layers=99');

-- Run inference
SELECT llm_text_generate('What is the most beautiful city in Italy?');
```

## 📦 Integrations

Use SQLite-AI alongside:

* **[SQLite-Vector](https://github.com/sqliteai/sqlite-vector)** – vector search from SQL
* **[SQLite-Sync](https://github.com/sqliteai/sqlite-sync)** – sync on-device databases with the cloud
* **[SQLite-JS](https://github.com/sqliteai/sqlite-js)** – define SQLite functions in JavaScript

---

## License

This project is licensed under the [Elastic License 2.0](./LICENSE.md). You can use, copy, modify, and distribute it under the terms of the license for non-production use. For production or managed service use, please [contact SQLite Cloud, Inc](mailto:info@sqlitecloud.io) for a commercial license.
