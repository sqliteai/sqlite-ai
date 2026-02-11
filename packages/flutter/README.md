# sqlite_ai

SQLite-AI is an extension for SQLite that brings artificial intelligence capabilities directly into the database. It enables developers to run, fine-tune, and serve AI models from within SQLite using simple SQL queries — ideal for on-device and edge applications where low-latency and offline inference are critical.

## Installation

```
dart pub add sqlite_ai
```

Requires Dart 3.10+ / Flutter 3.38+.

## Usage

### With `sqlite3`

```dart
import 'package:sqlite3/sqlite3.dart';
import 'package:sqlite_ai/sqlite_ai.dart';

void main() {
  // Load once at startup.
  sqlite3.loadSqliteAiExtension();

  final db = sqlite3.openInMemory();

  // Check version.
  final result = db.select('SELECT ai_version() AS version');
  print(result.first['version']);

  // Load a GGUF model.
  db.execute("SELECT llm_model_load('./models/llama.gguf', 'gpu_layers=99')");

  // Run inference.
  final response = db.select("SELECT llm_chat_respond('What is 2+2?') AS answer");
  print(response.first['answer']);

  db.dispose();
}
```

### With `drift`

```dart
import 'package:sqlite3/sqlite3.dart';
import 'package:sqlite_ai/sqlite_ai.dart';
import 'package:drift/native.dart';

Sqlite3 loadExtensions() {
  sqlite3.loadSqliteAiExtension();
  return sqlite3;
}

// Use when creating the database:
NativeDatabase.createInBackground(
  File(path),
  sqlite3: loadExtensions,
);
```

## Supported platforms

| Platform | Architectures |
|----------|---------------|
| Android  | arm64, x64 |
| iOS      | arm64 (device + simulator) |
| macOS    | arm64, x64 |
| Linux    | arm64, x64 |
| Windows  | x64 |

## API

See the full [sqlite-ai API documentation](https://github.com/sqliteai/sqlite-ai/blob/main/API.md).

## License

See [LICENSE](LICENSE).
