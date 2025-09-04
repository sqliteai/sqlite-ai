# SQLite-AI: API Reference

This document provides reference-level documentation for all public SQLite-AI functions, virtual tables, and metadata properties exposed to SQL.
These functions enable loading and interacting with LLMs, configuring samplers, generating embeddings and text, and managing chat sessions.

---

## `ai_version()`

**Returns:** `TEXT`

**Description:**
Returns the current version of the SQLite-AI extension.

**Example:**

```sql
SELECT ai_version();
-- e.g., '0.5.1'
```

---

## `ai_log_info(extended_enable BOOLEAN)`

**Returns:** `NULL`

**Description:**
Enables or disables extended logging information. Use `1` to enable, `0` to disable.

**Example:**

```sql
SELECT ai_log_info(1);
```

---

## `llm_model_load(path TEXT, options TEXT)`

**Returns:** `NULL`

**Description:**
Loads a GGUF model from the specified file path with optional comma separated key=value configuration.
If no options are provided the following default value is used: `gpu_layers=99`

The following keys are available:
```
gpu_layers=N       (N is the number of layers to store in VRAM)
main_gpu=K         (K is the GPU that is used for the entire model when split_mode is 0)
split_mode=N       (how to split the model across multiple GPUs, 0 means none, 1 means layer, 2 means rows)
vocab_only=1/0     (only load the vocabulary, no weights)
use_mmap=1/0       (use mmap if possible)
use_mlock=1/0      (force system to keep model in RAM)
check_tensors=1/0  (validate model tensor data)
log_info=1/0       (enable/disable the logging of info)
```

**Example:**

```sql
SELECT llm_model_load('./models/llama.gguf', 'gpu_layers=99');
```

---

## `llm_model_free()`

**Returns:** `NULL`

**Description:**
Unloads the current model and frees associated memory.

**Example:**

```sql
SELECT llm_model_free();
```

---

## `llm_context_create(options TEXT)`

**Returns:** `NULL`

**Description:**
Creates a new inference context with comma separated key=value configuration.

Context must explicitly created before performing any AI operation!

The following keys are available:
```
```

**Example:**

```sql
SELECT llm_context_create('n_ctx=2048');
```

---

## `llm_context_create_embedding()`

**Returns:** `NULL`

**Description:**
Creates a new inference context specifically set for embedding generation.

It is equivalent to `SELECT llm_context_create('generate_embedding=1,normalize_embedding=1,pooling_type=mean');`

Context must explicitly created before performing any AI operation!

**Example:**

```sql
SELECT llm_context_create_embedding();
```

---

## `llm_context_create_chat()`

**Returns:** `NULL`

**Description:**
Creates a new inference context specifically set for chat conversation.

It is equivalent to `SELECT llm_context_create('context_size=4096');`

Context must explicitly created before performing any AI operation!

**Example:**

```sql
SELECT llm_context_create_chat();
```

---

## `llm_context_create_textgen()`

**Returns:** `NULL`

**Description:**
Creates a new inference context specifically set for text generation.

It is equivalent to `SELECT llm_context_create('context_size=4096');`

Context must explicitly created before performing any AI operation!

**Example:**

```sql
SELECT llm_context_create_textgen();
```

---

## `llm_context_free()`

**Returns:** `NULL`

**Description:**
Frees the current inference context.

**Example:**

```sql
SELECT llm_context_free();
```

---

## `llm_sampler_create()`

**Returns:** `NULL`

**Description:**
Initializes a new sampling strategy for text generation.
A sampler is the mechanism that determines how the model selects the next token (word or subword) during text generation.
If no sampler is explicitly created, one will be created automatically when needed.

**Example:**

```sql
SELECT llm_sampler_create();
```

---

## `llm_sampler_free()`

**Returns:** `NULL`

**Description:**
Frees resources associated with the current sampler.

**Example:**

```sql
SELECT llm_sampler_free();
```

---

## `llm_lora_load(path TEXT, scale REAL)`

**Returns:** `NULL`

**Description:**
Loads a LoRA adapter from the given file path with a mandatory scale value.
LoRA (Low-Rank Adaptation) is a technique to inject trainable, low-rank layers into a pre-trained model.

**Example:**

```sql
SELECT llm_lora_load('./adapters/adapter.lora', 1.0);
```

---

## `llm_lora_free()`

**Returns:** `NULL`

**Description:**
Unloads any currently loaded LoRA adapter.

**Example:**

```sql
SELECT llm_lora_free();
```

---

## `llm_sampler_init_greedy()`

**Returns:** `NULL`

**Description:**
Configures the sampler to use greedy decoding (always pick most probable token).

**Example:**

```sql
SELECT llm_sampler_init_greedy();
```

---

## `llm_sampler_init_dist(seed INT)`

**Returns:** `NULL`

**Description:**
Initializes a random distribution-based sampler with the given seed.
If a seed value in not specified, a default 0xFFFFFFFF value will be used.

**Example:**

```sql
SELECT llm_sampler_init_dist(42);
```

---

## `llm_sampler_init_top_k(k INT)`

**Returns:** `NULL`

**Description:**
Limits sampling to the top `k` most likely tokens.
Top-K sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751

**Example:**

```sql
SELECT llm_sampler_init_top_k(40);
```

---

## `llm_sampler_init_top_p(p REAL, min_keep INT)`

**Returns:** `NULL`

**Description:**
Top-p sampling retains tokens with cumulative probability >= `p`. Always keeps at least `min_keep` tokens.
Nucleus sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751

**Example:**

```sql
SELECT llm_sampler_init_top_p(0.9, 1);
```

---

## `llm_sampler_init_min_p(p REAL, min_keep INT)`

**Returns:** `NULL`

**Description:**
Like top-p but with a minimum token probability threshold `p`.
Minimum P sampling as described in https://github.com/ggml-org/llama.cpp/pull/3841

**Example:**

```sql
SELECT llm_sampler_init_min_p(0.05, 1);
```

---

## `llm_sampler_init_typical(p REAL, min_keep INT)`

**Returns:** `NULL`

**Description:**
Typical sampling prefers tokens near the expected entropy level.
Locally Typical Sampling implementation described in the paper https://arxiv.org/abs/2202.00666

**Example:**

```sql
SELECT llm_sampler_init_typical(0.95, 1);
```

---

## `llm_sampler_init_temp(t REAL)`

**Returns:** `NULL`

**Description:**
Adjusts the sampling temperature to control randomness.

**Example:**

```sql
SELECT llm_sampler_init_temp(0.8);
```

---

## `llm_sampler_init_temp_ext(t REAL, delta REAL, exponent REAL)`

**Returns:** `NULL`

**Description:**
Advanced temperature control using exponential scaling.
Dynamic temperature implementation (a.k.a. entropy) described in the paper https://arxiv.org/abs/2309.02772

**Example:**

```sql
SELECT llm_sampler_init_temp_ext(0.8, 0.1, 2.0);
```

---

## `llm_sampler_init_xtc(p REAL, t REAL, min_keep INT, seed INT)`

**Returns:** `NULL`

**Description:**
Combines top-p, temperature, and seed-based sampling with a minimum token count.
XTC sampler as described in https://github.com/oobabooga/text-generation-webui/pull/6335

**Example:**

```sql
SELECT llm_sampler_init_xtc(0.9, 0.8, 1, 42);
```

---

## `llm_sampler_init_top_n_sigma(n REAL)`

**Returns:** `NULL`

**Description:**
Limits sampling to tokens within `n` standard deviations.
Top n sigma sampling as described in academic paper "Top-nσ: Not All Logits Are You Need" https://arxiv.org/pdf/2411.07641

**Example:**

```sql
SELECT llm_sampler_init_top_n_sigma(1.5);
```

---

## `llm_sampler_init_mirostat(seed INT, tau REAL, eta REAL, m INT)`

**Returns:** `NULL`

**Description:**
Initializes Mirostat sampling with entropy control.
Mirostat 1.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.

**Example:**

```sql
SELECT llm_sampler_init_mirostat(42, 5.0, 0.1, 100);
```

---

## `llm_sampler_init_mirostat_v2(seed INT, tau REAL, eta REAL)`

**Returns:** `NULL`

**Description:**
Mirostat v2 entropy-based sampling.
Mirostat 2.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.

**Example:**

```sql
SELECT llm_sampler_init_mirostat_v2(42, 5.0, 0.1);
```

---

## `llm_sampler_init_grammar(grammar_str TEXT, grammar_root TEXT)`

**Returns:** `NULL`

**Description:**
Constrains output to match a specified grammar.
Grammar syntax described in https://github.com/ggml-org/llama.cpp/tree/master/grammars

**Example:**

```sql
SELECT llm_sampler_init_grammar('...BNF...', 'root');
```

---

## `llm_sampler_init_infill()`

**Returns:** `NULL`

**Description:**
Enables infill (prefix-suffix) mode for completions.

**Example:**

```sql
SELECT llm_sampler_init_infill();
```

---

## `llm_sampler_init_penalties(n INT, repeat REAL, freq REAL, present REAL)`

**Returns:** `NULL`

**Description:**
Applies repetition, frequency, and presence penalties.

**Example:**

```sql
SELECT llm_sampler_init_penalties(64, 1.2, 0.5, 0.8);
```

---

## `llm_embed_generate(text TEXT, options TEXT)`

**Returns:** `BLOB` or `TEXT`

**Description:**
Generates a text embedding as a BLOB vector, with optional configuration provided as a comma-separated list of key=value pairs.
By default, the embedding is normalized unless `normalize_embedding=0` is specified.
If `json_output=1` is set, the function returns a JSON object instead of a BLOB.

**Example:**

```sql
SELECT llm_embed_generate('hello world', 'json_output=1');
```

---

## `llm_text_generate(text TEXT, options TEXT)`

**Returns:** `TEXT`

**Description:**
Generates a full-text completion based on input, with optional configuration provided as a comma-separated list of key=value pairs.

**Example:**

```sql
SELECT llm_text_generate('Once upon a time', 'n_predict=1024');
```

---

## `llm_chat(prompt TEXT)`

**Returns:** `VIRTUAL TABLE`

**Description:**
Streams a chat-style reply one token per row.

**Example:**

```sql
SELECT reply FROM llm_chat('Tell me a joke.');
```

---

## `llm_chat_create()`

**Returns:** `TEXT`

**Description:**
Starts a new in-memory chat session.
Returns unique chat UUIDv7 value.
If no chat is explicitly created, one will be created automatically when needed.

**Example:**

```sql
SELECT llm_chat_create();
```

---

## `llm_chat_free()`

**Returns:** `NULL`

**Description:**
Ends the current chat session.

**Example:**

```sql
SELECT llm_chat_free();
```

---

## `llm_chat_save(title TEXT, meta TEXT)`

**Returns:** `TEXT`

**Description:**
Saves the current chat session with optional title and meta into the ai_chat_history and ai_chat_messages tables and returns a UUID.

**Example:**

```sql
SELECT llm_chat_save('Support Chat', '{"user": "Marco"}');
```

---

## `llm_chat_restore(uuid TEXT)`

**Returns:** `NULL`

**Description:**
Restores a previously saved chat session by UUID.

**Example:**

```sql
SELECT llm_chat_restore('b59e...');
```

---

## `llm_chat_respond(text TEXT)`

**Returns:** `TEXT`

**Description:**
Generates a context-aware reply using chat memory, returned as a single, complete response.
For a streams model reply, use the llm_chat virtual table.

**Example:**

```sql
SELECT llm_chat_respond('What are the most visited cities in Italy?');
```

---

## Model Metadata

These functions return internal model properties:

```sql
SELECT
  llm_model_n_params(),
  llm_model_size(),
  llm_model_n_ctx_train(),
  llm_model_n_embd(),
  llm_model_n_layer(),
  llm_model_n_head(),
  llm_model_n_head_kv(),
  llm_model_n_swa(),
  llm_model_rope_freq_scale_train(),
  llm_model_n_cls_out(),
  llm_model_cls_label(),
  llm_model_desc(),
  llm_model_has_encoder(),
  llm_model_has_decoder(),
  llm_model_is_recurrent(),
  llm_model_chat_template();
```

All return `INTEGER`, `REAL`, or `TEXT` values depending on the property.

---
