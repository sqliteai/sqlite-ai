# Changelog

## 1.0.9

### Security
- **Process-wide logger no longer dangles.** `llama_log_set()` held a pointer to the most recently opened connection's context; closing that connection left it registered and the next llama/ggml log line crashed the process. The logger now routes through a mutex-protected registry of live connections.
- **`llm_chat()` survives releases mid-stream.** `SELECT reply, llm_context_free() FROM llm_chat(...)` (and the same with `llm_sampler_free()`, `llm_model_free()`, `llm_chat_free()`) crashed or aborted. The cursor now validates model, context, sampler and chat state on every row and returns an SQL error.
- **Audio resampling is bounded.** The output size scales with `16000 / sample_rate` taken from the file header, so a few KB at 1 Hz could allocate GBs. Sample rates outside 1000-1000000 Hz, zero channels or samples, and outputs that do not fit an `int` are rejected, and the error path no longer leaks.
- **Option validation.** `type_k` / `type_v` accept only the ggml types llama.cpp supports for the KV cache (an out-of-range value aborted the process); whisper `offset_ms` / `duration_ms` are clamped to 0-24h (a negative offset hung the transcription).
- **No `int` truncation in allocations.** Token buffers and `sqlite_strdup` use `sqlite3_malloc64`.

### Fixed
- `llm_model_cls_label()` read its argument with no argument registered; it is now available with zero arguments (first label) and with an index.
- `llm_sampler_init_infill()` without a loaded model.
- Vision generation loops stop when the context is full instead of failing in `llama_decode`.
- Chat template / buffer length handling in `llm_chat_*`, and `llm_chat_save` is atomic (SAVEPOINT).
- Embedding contexts set `n_ubatch = n_batch`.
- Freeing the model keeps the connection's `log_info` setting.
- `buffer_resize` / `buffer_append` reject sizes above `INT_MAX`.

### Tests
- `llm_embedding_then_chat` uses an explicit sampler chain with repetition penalties; the previous seeded chain fell into a repetition loop with the small test model.
