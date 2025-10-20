# @sqliteai/sqlite-ai

[![npm version](https://badge.fury.io/js/@sqliteai%2Fsqlite-ai.svg)](https://www.npmjs.com/package/@sqliteai/sqlite-ai)
[![License](https://img.shields.io/badge/license-Elastic%202.0-blue.svg)](LICENSE.md)

> SQLite AI extension packaged for Node.js

**SQLite-AI** is an extension for SQLite that brings artificial intelligence capabilities directly into the database. It enables developers to run, fine-tune, and serve AI models from within SQLite using simple SQL queries — ideal for on-device and edge applications where low-latency and offline inference are critical. The extension is actively developed by [SQLite AI](https://sqlite.ai), some API and features are still evolving.

## Features

- ✅ **LLaMA Integration** - Run LLaMA models directly in SQLite
- ✅ **Whisper Speech Recognition** - Transcribe audio with Whisper
- ✅ **Embedding Generation** - Generate vector embeddings for semantic search
- ✅ **Cross-platform** - Works on macOS, Linux (glibc/musl), and Windows
- ✅ **Zero configuration** - Automatically detects and loads the correct binary for your platform
- ✅ **TypeScript native** - Full type definitions included
- ✅ **Modern ESM + CJS** - Works with both ES modules and CommonJS
- ✅ **Offline-ready** - No external services required

## Installation

```bash
npm install @sqliteai/sqlite-ai
```

The package automatically downloads the correct native extension for your platform during installation.

### Supported Platforms

| Platform | Architecture | Package |
|----------|-------------|---------|
| macOS | ARM64 (Apple Silicon) | `@sqliteai/sqlite-ai-darwin-arm64` |
| macOS | x86_64 (Intel) | `@sqliteai/sqlite-ai-darwin-x86_64` |
| Linux | ARM64 (glibc) | `@sqliteai/sqlite-ai-linux-arm64` |
| Linux | ARM64 (musl/Alpine) | `@sqliteai/sqlite-ai-linux-arm64-musl` |
| Linux | x86_64 (glibc) | `@sqliteai/sqlite-ai-linux-x86_64` |
| Linux | x86_64 (musl/Alpine) | `@sqliteai/sqlite-ai-linux-x86_64-musl` |
| Windows | x86_64 | `@sqliteai/sqlite-ai-win32-x86_64` |

## sqlite-ai API

For detailed information on how to use the AI extension features, see the [main documentation](https://github.com/sqliteai/sqlite-ai/blob/main/API.md).

## Usage

```typescript
import { getExtensionPath } from '@sqliteai/sqlite-ai';
import Database from 'better-sqlite3';

const db = new Database(':memory:');
db.loadExtension(getExtensionPath());

// Ready to use
const version = db.prepare('SELECT ai_version()').pluck().get();
console.log('AI extension version:', version);
```

## Examples

For complete, runnable examples, see the [sqlite-extensions-guide](https://github.com/sqliteai/sqlite-extensions-guide/tree/main/examples/node).

These examples are generic and work with all SQLite extensions: `sqlite-vector`, `sqlite-sync`, `sqlite-js`, and `sqlite-ai`.

## API Reference

### `getExtensionPath(): string`

Returns the absolute path to the SQLite AI extension binary for the current platform.

**Returns:** `string` - Absolute path to the extension file (`.so`, `.dylib`, or `.dll`)

**Throws:** `ExtensionNotFoundError` - If the extension binary cannot be found for the current platform

**Example:**
```typescript
import { getExtensionPath } from '@sqliteai/sqlite-ai';

const path = getExtensionPath();
// => '/path/to/node_modules/@sqliteai/sqlite-ai-darwin-arm64/ai.dylib'
```

---

### `getExtensionInfo(): ExtensionInfo`

Returns detailed information about the extension for the current platform.

**Returns:** `ExtensionInfo` object with the following properties:
- `platform: Platform` - Current platform identifier (e.g., `'darwin-arm64'`)
- `packageName: string` - Name of the platform-specific npm package
- `binaryName: string` - Filename of the binary (e.g., `'ai.dylib'`)
- `path: string` - Full path to the extension binary

**Throws:** `ExtensionNotFoundError` - If the extension binary cannot be found

**Example:**
```typescript
import { getExtensionInfo } from '@sqliteai/sqlite-ai';

const info = getExtensionInfo();
console.log(`Running on ${info.platform}`);
console.log(`Extension path: ${info.path}`);
```

---

### `getCurrentPlatform(): Platform`

Returns the current platform identifier.

**Returns:** `Platform` - One of:
- `'darwin-arm64'` - macOS ARM64
- `'darwin-x86_64'` - macOS x86_64
- `'linux-arm64'` - Linux ARM64 (glibc)
- `'linux-arm64-musl'` - Linux ARM64 (musl)
- `'linux-x86_64'` - Linux x86_64 (glibc)
- `'linux-x86_64-musl'` - Linux x86_64 (musl)
- `'win32-x86_64'` - Windows x86_64

**Throws:** `Error` - If the platform is unsupported

---

### `isMusl(): boolean`

Detects if the system uses musl libc (Alpine Linux, etc.).

**Returns:** `boolean` - `true` if musl is detected, `false` otherwise

---

### `class ExtensionNotFoundError extends Error`

Error thrown when the SQLite AI extension cannot be found for the current platform.

## Related Projects

- **[@sqliteai/sqlite-vector](https://www.npmjs.com/package/@sqliteai/sqlite-vector)** - Vector search and similarity matching
- **[@sqliteai/sqlite-sync](https://www.npmjs.com/package/@sqliteai/sqlite-sync)** - Sync on-device databases with the cloud
- **[@sqliteai/sqlite-js](https://www.npmjs.com/package/@sqliteai/sqlite-js)** - Define SQLite functions in JavaScript

## License

This project is licensed under the [Elastic License 2.0](LICENSE.md).

For production or managed service use, please [contact SQLite Cloud, Inc](mailto:info@sqlitecloud.io) for a commercial license.

## Contributing

Contributions are welcome! Please see the [main repository](https://github.com/sqliteai/sqlite-ai) to open an issue.

## Support

- 📖 [Documentation](https://github.com/sqliteai/sqlite-ai/blob/main/API.md)
- 🐛 [Report Issues](https://github.com/sqliteai/sqlite-ai/issues)
