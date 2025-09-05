## Are there cases where llama.cpp can produce more than one embedding from a single text input?

In **llama.cpp**, whether you get **one embedding** or **multiple embeddings** from a text input depends on:

1. **Pooling type (`llama_pooling_type`)**

   * `LLAMA_POOLING_TYPE_NONE` → no pooling is applied.

     * You get an embedding **per token** (so length = number of tokens).
     * You retrieve it with `llama_get_embeddings(ctx)` after `llama_encode()` or `llama_decode()`.
   * `LLAMA_POOLING_TYPE_MEAN`, `LLAMA_POOLING_TYPE_CLS`, `LLAMA_POOLING_TYPE_LAST` → pooling is applied.

     * You get **one embedding per sequence ID**.
     * You retrieve it with `llama_get_embeddings_seq(ctx, seq_id)`.

2. **Number of sequence IDs (`seq_id`) in the batch**

   * Each sequence in the batch (distinguished by `seq_id`) can produce its own pooled embedding if pooling is enabled.
   * Example: if you feed 3 sentences in parallel with `seq_id = 0,1,2` and pooling = MEAN, you’ll get **3 embeddings**.

3. **How you mark logits/outputs in the batch**

   * With pooling enabled, you only need to request an output on one token per sequence (often the last token).
   * Without pooling, you must request logits for every token you want embeddings for.

---

So, to directly answer:

* **Single text input, pooling enabled** → **1 embedding** (sequence-level).
* **Single text input, pooling disabled** → **N embeddings** (token-level, one per token).
* **Multiple texts in one batch** → **N embeddings** (one per sequence if pooling, or token-level if not).

---

In your embedding use case (sounds like you’re doing what `llama.cpp` calls **sentence embedding**), you probably want **pooling enabled** (MEAN or CLS) and therefore **one embedding per text input**.

## Is there a way to force one embedding (sentence embedding)?

Yes, in **llama.cpp** you can force sentence-level embeddings by enabling a **pooling mode** in the context parameters. That way you’ll always get exactly **one embedding per sequence ID**, regardless of how many tokens the text expands into.

---

### How to do it

When you create the context (`llama_context_params`), set:

```c
struct llama_context_params cparams = llama_context_default_params();

// choose your pooling mode:
cparams.pooling_type = LLAMA_POOLING_TYPE_MEAN;  // common for sentence embeddings
// or: LLAMA_POOLING_TYPE_CLS
// or: LLAMA_POOLING_TYPE_LAST

struct llama_context *ctx = llama_new_context_with_model(model, cparams);
```

Then, when you process text with `llama_encode()` or `llama_decode()`:

* Use a single `seq_id` for that text (e.g. `seq_id = 0`).
* After the call, retrieve the **sequence embedding**:

```c
const float *embedding = llama_get_embeddings_seq(ctx, seq_id);
// embedding size = llama_n_embd(model)
```

That’s it: you’ll get **one embedding vector** for the entire input string.

---

### Contrast with token-level embeddings

* If `pooling_type == LLAMA_POOLING_TYPE_NONE` → `llama_get_embeddings(ctx)` gives you **token-level embeddings** (matrix: tokens × dim).
* If `pooling_type != NONE` → `llama_get_embeddings_seq(ctx, seq_id)` gives you **sentence-level embeddings** (one vector per sequence).

---

**So the way to force one embedding per text is:**

* Set `pooling_type` in the context params (`MEAN`, `CLS`, or `LAST`).
* Use `llama_get_embeddings_seq()` instead of `llama_get_embeddings()`.

---

## Is the pooling_type related to the model? Can I be sure that all models support all the pooling types?


The **pooling\_type** in **llama.cpp** is **not a property of the model itself** – it’s purely how **llama.cpp** aggregates the **token embeddings** it already computes. All decoder-only LLaMA-style models output a hidden state per token, so pooling is just a post-processing step that the library offers.

---

### What this means

* Any model you load with **llama.cpp** produces **per-token embeddings**.
* Pooling (`MEAN`, `CLS`, `LAST`) is applied on top of those token embeddings to compress them into **one vector**.
* So yes: **all models supported by llama.cpp can use all pooling types**.
* The only requirement is that you create the context with `cparams.pooling_type` set, so llama.cpp knows to allocate memory for pooled embeddings.

---

### Differences between pooling modes

* **MEAN** → average of all token embeddings in the sequence.

  * Default / most common for sentence embeddings & semantic search.
* **CLS** → use the embedding of the first token (analogous to `[CLS]` in BERT-style models).

  * Works, but LLaMA models weren’t trained with a `[CLS]` objective, so performance may vary.
* **LAST** → use the embedding of the final token.

  * Sometimes useful for autoregressive tasks; not typically the best for embeddings.

---

### Important

The quality of the **sentence embedding** you get *does* depend on the **model training objective**:

* Models explicitly trained for embedding tasks (e.g. `all-MiniLM-L6-v2`, `text-embedding-ada-002`, `nomic-embed`) will usually outperform a raw LLaMA model, even if you apply pooling.
* LLaMA-style models with pooling still give you usable vectors, but they weren’t optimized for semantic similarity.

---

So in summary:

* Pooling is **always available** in llama.cpp.
* All models supported by llama.cpp can use **MEAN / CLS / LAST** pooling.
* The *choice of pooling* affects embedding quality, but you won’t get an error from the library.

---