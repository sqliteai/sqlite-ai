# SQLite-AI

**SQLite-AI** is an extension for SQLite that brings artificial intelligence capabilities directly into the database. It enables developers to run, fine-tune, and serve AI models from within SQLite using simple SQL queries — ideal for on-device and edge applications where low-latency and offline inference are critical. The extension is actively developed by [SQLite AI](https://sqlite.ai), some API and features are still evolving.

## Features

* **Embedded AI Inference**: Run transformer models directly from SQL queries.
* **Streaming I/O**: Token-by-token streaming via SQL aggregate functions.
* **Fine-tuning & Embedding**: On-device model customization and vector embedding.
* **Full On-Device Support**: Works on iOS, Android, Linux, macOS, and Windows.
* **Offline-First**: No server dependencies or internet connection required.
* **Composable SQL Interface**: AI + relational logic in a single unified layer.
* **Audio Transcription**: Speech-to-text via Whisper models (WAV, MP3, FLAC).
* **Vision / Multimodal**: Analyze images via multimodal models (JPG, PNG, BMP, GIF).
* **Supports any GGUF model**: available on Huggingface; Qwen, Gemma, Llama, DeepSeek and more

SQLite-AI supports **text embedding generation** for search and classification, a **chat-like interface with history and token streaming**, **automatic context save and restore** across sessions, **audio transcription** via Whisper models, and **vision/multimodal** image understanding — making it ideal for building conversational agents, memory-aware assistants, and voice-enabled applications.

## Getting Started

```bash
# Start SQLite CLI
sqlite3 myapp.db
```

```sql
-- Load the extension
.load ./ai
```

### Text Generation

```sql
-- Load a text generation model
SELECT llm_model_load('./models/Qwen2.5-3B-Q4_K_M.gguf', 'gpu_layers=99');
SELECT llm_context_create_textgen();

-- Generate text
SELECT llm_text_generate('What is the most beautiful city in Italy?');
```

### Embedding Generation

```sql
-- Load an embedding model
SELECT llm_model_load('./models/nomic-embed-text-v1.5-Q8_0.gguf', 'gpu_layers=99');
SELECT llm_context_create_embedding('embedding_type=FLOAT32');

-- Generate an embedding vector
SELECT llm_embed_generate('Hello world');

-- Generate an embedding as JSON
SELECT llm_embed_generate('Hello world', 'json_output=1');
```

### Chat

```sql
-- Load a chat model
SELECT llm_model_load('./models/Llama-3.2-3B-Instruct-Q4_K_M.gguf', 'gpu_layers=99');
SELECT llm_context_create_chat();

-- Send a message and get a complete response
SELECT llm_chat_respond('Tell me a joke.');

-- Or stream the reply token by token
SELECT reply FROM llm_chat('Tell me another joke.');
```

### Audio Transcription

```sql
-- Load a Whisper model
SELECT audio_model_load('./models/ggml-tiny.bin');

-- Transcribe from a file path
SELECT audio_model_transcribe('./audio/speech.wav');

-- Transcribe with options
SELECT audio_model_transcribe('./audio/speech.mp3', 'language=it,translate=1');

-- Transcribe from a BLOB column
SELECT audio_model_transcribe(audio_data) FROM recordings WHERE id = 1;
```

### Vision / Multimodal

```sql
-- Load a multimodal model and its vision projector
SELECT llm_model_load('./models/Gemma-3-4B-IT-Q4_K_M.gguf', 'gpu_layers=99');
SELECT llm_context_create_textgen();
SELECT llm_vision_load('./models/mmproj-Gemma-3-4B-IT-f16.gguf');

-- Describe an image
SELECT llm_text_generate('Describe this image', './photos/cat.jpg');

-- Use vision in a chat conversation
SELECT llm_context_create_chat();
SELECT llm_chat_respond('What do you see in this photo?', './photos/landscape.jpg');

-- Analyze multiple images
SELECT llm_text_generate('Compare these two images', './img1.jpg', './img2.jpg');
```

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

### Swift Package

You can [add this repository as a package dependency to your Swift project](https://developer.apple.com/documentation/xcode/adding-package-dependencies-to-your-app#Add-a-package-dependency). After adding the package, you'll need to set up SQLite with extension loading by following steps 4 and 5 of [this guide](https://github.com/sqliteai/sqlite-extensions-guide/blob/main/platforms/ios.md#4-set-up-sqlite-with-extension-loading).

Here's an example of how to use the package:
```swift
import ai

...

var db: OpaquePointer?
sqlite3_open(":memory:", &db)
sqlite3_enable_load_extension(db, 1)
var errMsg: UnsafeMutablePointer<Int8>? = nil
sqlite3_load_extension(db, ai.path, nil, &errMsg)
var stmt: OpaquePointer?
sqlite3_prepare_v2(db, "SELECT ai_version()", -1, &stmt, nil)
defer { sqlite3_finalize(stmt) }
sqlite3_step(stmt)
log("ai_version(): \(String(cString: sqlite3_column_text(stmt, 0)))")
sqlite3_close(db)
```

### Android Package

Add the [following](https://central.sonatype.com/artifact/ai.sqlite/ai) to your Gradle dependencies:

```gradle
implementation 'ai.sqlite:ai:0.7.55'
```

Here's an example of how to use the package:
```java
SQLiteCustomExtension aiExtension = new SQLiteCustomExtension(getApplicationInfo().nativeLibraryDir + "/ai", null);
SQLiteDatabaseConfiguration config = new SQLiteDatabaseConfiguration(
    getCacheDir().getPath() + "/ai_test.db",
    SQLiteDatabase.CREATE_IF_NECESSARY | SQLiteDatabase.OPEN_READWRITE,
    Collections.emptyList(),
    Collections.emptyList(),
    Collections.singletonList(aiExtension)
);
SQLiteDatabase db = SQLiteDatabase.openDatabase(config, null, null);
```

**Note:** Additional settings and configuration are required for a complete setup. For full implementation details, see the [complete Android example](https://github.com/sqliteai/sqlite-extensions-guide/blob/main/examples/android/README.md).

### Python Package

Python developers can quickly get started using the ready-to-use `sqlite-ai` package available on PyPI:

```bash
pip install sqlite-ai
```

For usage details and examples, see the [Python package documentation](./packages/python/README.md).

### Flutter Package

Add the [sqlite_ai](https://pub.dev/packages/sqlite_ai) package to your project:

```bash
flutter pub add sqlite_ai  # Flutter projects
dart pub add sqlite_ai     # Dart projects
```

Usage with `sqlite3` package:
```dart
import 'package:sqlite3/sqlite3.dart';
import 'package:sqlite_ai/sqlite_ai.dart';

sqlite3.loadSqliteAiExtension();
final db = sqlite3.openInMemory();
print(db.select('SELECT ai_version()'));
```

For a complete example, see the [Flutter example](https://github.com/sqliteai/sqlite-extensions-guide/blob/main/examples/flutter/README.md).

---

## License

This project is licensed under the [Elastic License 2.0](./LICENSE.md). You can use, copy, modify, and distribute it under the terms of the license for non-production use. For production or managed service use, please [contact SQLite Cloud, Inc](mailto:info@sqlitecloud.io) for a commercial license.

---
<!-- SQLITE-AI-ECOSYSTEM:START -->
<!-- SQLITE-AI-ECOSYSTEM:END -->
