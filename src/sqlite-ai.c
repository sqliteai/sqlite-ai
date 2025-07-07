//
//  sqlite-ai.c
//  sqliteai
//
//  Created by Marco Bambini on 26/06/25.
//

//  From Georgi Gerganov
//
//  Most embedding models don't have a memory (a.k.a. a KV cache).
//  This is not an error - just a warning telling you that you can simply use llama_encode() instead of llama_decode().
//  If your application is going to support both models with and without a memory, then you should simply call llama_decode() always.
//  The warning message is a debug message (LLAMA_LOG_DEBUG) - it only appears in debug builds and is useful for the llama.cpp developers.
//  It's not something that you should worry about.
//  https://github.com/ggml-org/llama.cpp/discussions/14454

#include "utils.h"
#include "llama.h"
#include "sqlite-ai.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef SQLITE_CORE
SQLITE_EXTENSION_INIT1
#endif

#define DEBUG_AI_ALWAYS(...)                    do {printf(__VA_ARGS__ );printf("\n");} while (0)

#if ENABLE_AI_DEBUG
#define DEBUG_AI(...)                           do {printf(__VA_ARGS__ );printf("\n");} while (0)
#else
#define DEBUG_AI(...)
#endif

#define NPREDICT_DEFAULT_VALUE                  128
#define MIN_ALLOC_TOKEN                         512
#define MAX_PATH                                4096
#define MAX_TOKEN_TEXT_LEN                      128     // according to ChatGPT 32 would be safe for all common tokenizers

#define LOG_TABLE_DECLARATION                   "CREATE TEMP TABLE ai_log (id INTEGER PRIMARY KEY, stamp DATETIME DEFAULT CURRENT_TIMESTAMP, type TEXT, message TEXT);"
#define LOG_TABLE_INSERT_STMT                   "INSERT INTO ai_log (type, message) VALUES (?, ?);"

#define OPTION_KEY_GENERATE_EMBEDDING           "generate_embedding"
#define OPTION_KEY_NORMALIZE_EMBEDDING          "normalize_embedding"
#define OPTION_KEY_MAX_TOKENS                   "max_tokens"
#define OPTION_KEY_JSON_OUTPUT                  "json_output"
#define OPTION_KEY_GPU_LAYERS                   "gpu_layers"
#define OPTION_KEY_CONTEXT_SIZE                 "context_size"
#define OPTION_KEY_N_PREDICT                    "n_predict"

#define AI_COLUMN_REPLY                         0

typedef struct {
    // ** MODEL **
    int32_t                     gpu_layers;         // number of layers to store in VRAM
    int32_t                     main_gpu;           // the GPU that is used for the entire model when split_mode is 0
    enum llama_split_mode       split_mode;         // how to split the model across multiple GPUs
    bool                        vocab_only;         // only load the vocabulary, no weights
    bool                        use_mmap;           // use mmap if possible
    bool                        use_mlock;          // force system to keep model in RAM
    bool                        check_tensors;      // validate model tensor data
    bool                        log_info;           // flag to enable/disable the logging of info
    
    // ** CONTEXT **
    uint32_t                    context_size;       // set both n_ctx and n_batch
    uint32_t                    n_ctx;              // text context, 0 = from model
    uint32_t                    n_batch;            // logical maximum batch size that can be submitted to llama_decode
    uint32_t                    n_ubatch;           // physical maximum batch size
    uint32_t                    n_seq_max;          // max number of sequences (i.e. distinct states for recurrent models)
    int32_t                     n_threads;          // number of threads to use for generation
    int32_t                     n_threads_batch;    // number of threads to use for batch processing
    enum llama_rope_scaling_type rope_scaling_type; // RoPE scaling type, from `enum llama_rope_scaling_type`
    enum llama_pooling_type     pooling_type;       // whether to pool (sum) embedding results by sequence id
    enum llama_attention_type   attention_type;     // attention type to use for embeddings
    float                       rope_freq_base;     // RoPE base frequency, 0 = from model
    float                       rope_freq_scale;    // RoPE frequency scaling factor, 0 = from model
    float                       yarn_ext_factor;    // YaRN extrapolation mix factor, negative = from model
    float                       yarn_attn_factor;   // YaRN magnitude scaling factor
    float                       yarn_beta_fast;     // YaRN low correction dim
    float                       yarn_beta_slow;     // YaRN high correction dim
    uint32_t                    yarn_orig_ctx;      // YaRN original context size
    float                       defrag_thold;       // defragment the KV cache if holes/size > thold, <= 0 disabled (default)
    enum ggml_type              type_k;             // data type for K cache [EXPERIMENTAL]
    enum ggml_type              type_v;             // data type for V cache [EXPERIMENTAL]
    bool                        offload_kqv;        // offload the KQV ops (including the KV cache) to GPU
    bool                        flash_attn;         // use flash attention [EXPERIMENTAL]
    bool                        op_offload;         // offload host tensor operations to device
    bool                        swa_full;           // use full-size SWA cache (https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055)

    // ** SAMPLER **
    int                         n_predict;          // number of tokens to predict
    // EXPOSE FUNCTIONS INSTEAD OF SETTINGS
    // ai_sampler_create
    // ai_sampler_free
    /*
    int32_t                     top_k;              // Top-K sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
    float                       top_p;              // Nucleus sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
    size_t                      top_p_min_keep;
    float                       min_p;              // Minimum P sampling as described in https://github.com/ggml-org/llama.cpp/pull/3841
    size_t                      min_p_min_keep;
    float                       typical;            // Locally Typical Sampling implementation described in the paper https://arxiv.org/abs/2202.00666
    size_t                      typical_min_keep;
    float                       temp;
    float                       temp_ext;           // Dynamic temperature implementation (a.k.a. entropy) described in the paper https://arxiv.org/abs/2309.02772
    float                       temp_ext_delta;
    float                       temp_ext_exponent;
    float                       xtc;                // XTC sampler as described in https://github.com/oobabooga/text-generation-webui/pull/6335
    float                       xtc_t;
    size_t                      xtc_min_keep;
    uint32_t                    xtc_seed;
    float                       top_n_sigma;        // Top n sigma sampling as described in academic paper "Top-nσ: Not All Logits Are You Need" https://arxiv.org/pdf/2411.07641
    
    int32_t                     mirostat_n_vocab;   // Mirostat 1.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
    uint32_t                    mirostat_seed;
    float                       mirostat_tau;
    float                       mirostat_eta;
    int32_t                     mirostat_m;
    
    uint32_t                    mirostatv2_seed;    // Mirostat 2.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
    float                       mirostatv2_tau;
    float                       mirostatv2_eta;
    
    int32_t                     penalty_last_n;     // last n tokens to penalize (0 = disable penalty, -1 = context size)
    float                       penalty_repeat;     // 1.0 = disabled
    float                       penalty_freq;       // 0.0 = disabled
    float                       penalty_present;    // 0.0 = disabled
     */
    
    // ** EMBEDDING **
    bool                        generate_embedding; // if true, extract embeddings (together with logits)
    bool                        normalize_embedding;// if true, embeddings are normalized
    bool                        json_output;        // if true, embedding result is converted to JSON
    
    // ** CUSTOM **
    int32_t                     max_tokens;         // to control max allowed tokens to generate (to control user's input size)
    
} ai_options;

typedef struct {
    sqlite3                     *db;
    struct llama_model          *model;
    struct llama_context        *ctx;
    struct llama_sampler        *sampler;
    
    llama_token                 *token_buffer;
    int32_t                     token_buffer_size;
    ai_options                  options;
} ai_context;

typedef struct {
    sqlite3_vtab                base;               // Base class - must be first
    ai_context                  *ai;
} ai_vtab;

typedef struct {
    llama_chat_message *items;
    size_t count;
    size_t capacity;
} ai_messages;

typedef struct {
    sqlite3_vtab_cursor         base;               // Base class - must be first
    ai_vtab                     *vtab;
    ai_context                  *ai;
    ai_messages                 messages;
    buffer_t                    formatted;
    bool                        is_eog;
    sqlite_int64                rowid;
    int                         prev_len;
} ai_cursor;

// MARK: -

static void ai_options_init (ai_options *options) {
    memset(options, 0, sizeof(ai_options));
    
    options->normalize_embedding = true;
    options->max_tokens = 0;    // no limits
    options->log_info = false;  // disable INFO messages logging
}

static bool ai_options_callback (sqlite3_context *context, void *xdata, const char *key, int key_len, const char *value, int value_len) {
    ai_options *options = (ai_options *)xdata;
    
    // sanity check
    if (!key || key_len == 0) return false;
    if (!value || value_len == 0) return false;
    
    // debug
    // printf("KEY: \"%.*s\", VALUE: \"%.*s\"\n", key_len, key, value_len, value);
    
    // convert value to c-string
    char buffer[256] = {0};
    size_t len = (value_len > sizeof(buffer)-1) ? sizeof(buffer)-1 : value_len;
    memcpy(buffer, value, len);
    
    if (strncasecmp(key, OPTION_KEY_GENERATE_EMBEDDING, key_len) == 0) {
        int value = (int)strtol(buffer, NULL, 0);
        options->generate_embedding = (value != 0);
        return true;
    }
    
    if (strncasecmp(key, OPTION_KEY_NORMALIZE_EMBEDDING, key_len) == 0) {
        int value = (int)strtol(buffer, NULL, 0);
        options->normalize_embedding = (value != 0);
        return true;
    }
    
    if (strncasecmp(key, OPTION_KEY_MAX_TOKENS, key_len) == 0) {
        int value = (int)strtol(buffer, NULL, 0);
        if (value >= 0) options->max_tokens = value;
        return true;
    }
    
    if (strncasecmp(key, OPTION_KEY_JSON_OUTPUT, key_len) == 0) {
        int value = (int)strtol(buffer, NULL, 0);
        options->json_output = (value != 0);
        return true;
    }
    
    if (strncasecmp(key, OPTION_KEY_GPU_LAYERS, key_len) == 0) {
        int value = (int)strtol(buffer, NULL, 0);
        options->gpu_layers = value;
        return true;
    }
    
    if (strncasecmp(key, OPTION_KEY_CONTEXT_SIZE, key_len) == 0) {
        int value = (int)strtol(buffer, NULL, 0);
        options->context_size = value;
        return true;
    }
    
    if (strncasecmp(key, OPTION_KEY_N_PREDICT, key_len) == 0) {
        int value = (int)strtol(buffer, NULL, 0);
        options->n_predict = value;
        return true;
    }
    
    // means ignore unknown keys
    return true;
}

void *ai_sqlite_context_create (sqlite3 *db) {
    ai_context *ai = (ai_context *)sqlite3_malloc(sizeof(ai_context));
    if (ai) {
        ai_options_init(&ai->options);
        ai->db = db;
    }
    return ai;
}

void ai_sqlite_context_free (void *ctx) {
    if (!ctx) return;
    
    ai_context *ai = (ai_context *)ctx;
    if (ai->model) llama_model_free(ai->model);
    if (ai->ctx) llama_free(ai->ctx);
    if (ai->token_buffer) sqlite3_free(ai->token_buffer);
}

void ai_logger (enum ggml_log_level level, const char *text, void *user_data) {
    ai_context *ai = (ai_context *)user_data;
    if ((level == GGML_LOG_LEVEL_INFO) && (ai->options.log_info == false)) return;
    
    const char *type = NULL;
    switch (level) {
        case GGML_LOG_LEVEL_NONE: type = "NONE"; break;
        case GGML_LOG_LEVEL_DEBUG: type = "DEBUG"; break;
        case GGML_LOG_LEVEL_INFO: type = "INFO"; break;
        case GGML_LOG_LEVEL_WARN: type = "WARNING"; break;
        case GGML_LOG_LEVEL_ERROR: type = "ERROR"; break;
        case GGML_LOG_LEVEL_CONT: type = NULL; break;
    }
    
    const char *values[] = {type, text};
    int types[] = {SQLITE_TEXT, SQLITE_TEXT};
    int lens[] = {-1, -1};
    sqlite_db_write(NULL, ai->db, LOG_TABLE_INSERT_STMT, values, types, lens, 2);
}

void ai_set_model_options (struct llama_model_params *model_params, ai_options *options) {
    // number of layers to store in VRAM
    if (options->gpu_layers) model_params->n_gpu_layers = options->gpu_layers;
    if (options->split_mode) model_params->split_mode = options->split_mode;
    if (options->main_gpu) model_params->main_gpu = options->main_gpu;
    if (options->vocab_only) model_params->vocab_only = options->vocab_only;
    if (options->use_mmap) model_params->use_mmap = options->use_mmap;
    if (options->use_mlock) model_params->use_mlock = options->use_mlock;
    if (options->check_tensors) model_params->check_tensors = options->check_tensors;
}

void ai_set_context_options (struct llama_context_params *llama_context, ai_options *options) {
    if (options->generate_embedding) llama_context->embeddings = true;
    if (options->context_size) {
        llama_context->n_ctx = options->context_size;
        llama_context->n_batch = options->context_size;
    }
}

bool ai_model_check (sqlite3_context *context) {
    ai_context *ai = (ai_context *)sqlite3_user_data(context);
    if (!ai) return false;
    return (ai->model != NULL);
}

struct llama_context *ai_context_check (sqlite3_context *context) {
    ai_context *ai = (ai_context *)sqlite3_user_data(context);
    if (ai->ctx) return ai->ctx;
    
    struct llama_context_params ctx_params = llama_context_default_params();
    ai_set_context_options(&ctx_params, &ai->options);
    
    struct llama_context *ctx = llama_init_from_model(ai->model, ctx_params);
    if (!ctx) {
        sqlite_context_result_error(context, SQLITE_ERROR, "Unable to create context from model");
        return NULL;
    }
    
    return ctx;
}

struct llama_sampler *ai_sampler_check (sqlite3_context *context) {
    ai_context *ai = (ai_context *)sqlite3_user_data(context);
    if (ai->sampler) return ai->sampler;
    
    struct llama_sampler_chain_params sampler_params = llama_sampler_chain_default_params();
    struct llama_sampler *sampler = llama_sampler_chain_init(sampler_params);
    if (!sampler) {
        sqlite_context_result_error(context, SQLITE_ERROR, "Unable to create sampler");
        return NULL;
    }
    ai->sampler = sampler;
    return sampler;
}

static bool ai_common_args_check (sqlite3_context *context, const char *function_name, int argc, sqlite3_value **argv, bool check_model) {
    // sanity check arguments
    if (argc == 1) {
        int types[] = {SQLITE_TEXT};
        return sqlite_sanity_function(context, function_name, argc, argv, 1, types, check_model);
    } else if (argc == 2) {
        int types[] = {SQLITE_TEXT, SQLITE_TEXT};
        return sqlite_sanity_function(context, function_name, argc, argv, 2, types, check_model);
    }
    return sqlite_context_result_error(context, SQLITE_ERROR, "Function '%s' expects 1 or 2 arguments, but %d were provided.", function_name, argc);
}

// MARK: -

bool ai_messages_append (ai_messages *list, const char *role, const char *content) {
    if (list->count >= list->capacity) {
        size_t new_cap = list->capacity ? list->capacity * 2 : 4;
        llama_chat_message *new_items = sqlite3_realloc64(list->items, new_cap * sizeof(llama_chat_message));
        if (!new_items) return false;
        
        list->items = new_items;
        list->capacity = new_cap;
    }

    list->items[list->count].role = sqlite_strdup(role);
    list->items[list->count].content = sqlite_strdup(content);
    list->count += 1;
    return true;
}

void ai_messages_free (ai_messages *list) {
    for (size_t i = 0; i < list->count; ++i) {
        sqlite3_free((char *)list->items[i].role);
        sqlite3_free((char *)list->items[i].content);
    }
    sqlite3_free(list->items);
    list->items = NULL;
    list->count = 0;
    list->capacity = 0;
}

// MARK: -

static const char *db_check_model (sqlite3 *db, const char *name, char *path, size_t path_len) {
    // TODO: load path from name inside the ai_models table
    return NULL;
}

// MARK: - Text Embedding -

static void ai_embed_normalize (const float *src, float *dest, int dim) {
    float sum = 0.0f;
    
    // compute L2 norm squared (loop unrolled by 4)
    int i = 0;
    for (; i + 3 < dim; i += 4) {
        sum += src[i] * src[i] + src[i + 1] * src[i + 1] + src[i + 2] * src[i + 2] + src[i + 3] * src[i + 3];
    }
    for (; i < dim; ++i) {
        sum += src[i] * src[i];
    }
    
    float norm = sqrtf(sum);
    if (norm > 0.0f) {
        float inv = 1.0f / norm;
        i = 0;
        for (; i + 3 < dim; i += 4) {
            dest[i]     = src[i]     * inv;
            dest[i + 1] = src[i + 1] * inv;
            dest[i + 2] = src[i + 2] * inv;
            dest[i + 3] = src[i + 3] * inv;
        }
        for (; i < dim; ++i) {
            dest[i] = src[i] * inv;
        }
    } else {
        // if norm is zero, copy zeros
        for (int j = 0; j < dim; ++j) {
            dest[j] = 0.0f;
        }
    }
}

static void ai_text_embed_run (sqlite3_context *context, const char *text, int32_t text_len) {
    ai_context *ai = (ai_context *)sqlite3_user_data(context);
    
    // sanity check model
    if (llama_model_has_encoder(ai->model) && llama_model_has_decoder(ai->model)) {
        sqlite_context_result_error(context, SQLITE_ERROR, "Computing embeddings in encoder-decoder models is not supported");
        return;
    }
    
    // sanity check model type (decode is used to create embeddings)
    if (llama_model_has_decoder(ai->model) == false) {
        sqlite_context_result_error(context, SQLITE_ERROR, "Model does not support decoding (required for embedding)");
        return;
    }
    
    // sanity check vocab
    const struct llama_vocab *vocab = llama_model_get_vocab(ai->model);
    if (!vocab) {
        sqlite_context_result_error(context, SQLITE_ERROR, "Failed to extract vocabulary from the model");
        return;
    }
    
    // sanity check context
    if (!ai->ctx) {
        ai->options.generate_embedding = true;
        ai->ctx = ai_context_check(context);
    }
    if (!ai->ctx) return;
    
    struct llama_context *ctx = ai->ctx;
    llama_set_embeddings(ctx, true);
    
    // sanity check tokens
    const int n_ctx_train = llama_model_n_ctx_train(ai->model);
    const int n_ctx = llama_n_ctx(ctx);
    if (n_ctx > n_ctx_train) {
        char buffer[512];
        snprintf(buffer, sizeof(buffer), "Model was trained on only %d context tokens (%d specified)", n_ctx_train, n_ctx);
        ai_logger(GGML_LOG_LEVEL_WARN, buffer, sqlite3_context_db_handle(context));
    }
    
    /*
    if (llama_vocab_get_add_sep(vocab)) {
        const char *sep = llama_vocab_get_text(vocab, llama_vocab_sep(vocab));
        printf("sep: %s\n", sep);
    }
    
    if (llama_vocab_get_add_eos(vocab)) {
        const char *eos = llama_vocab_get_text(vocab, llama_vocab_eos(vocab));
        printf("eos: %s\n", eos);
    }
     */
    
    // sanity check embedding memory
    int dimension = llama_model_n_embd(llama_get_model(ctx));
    float *embedding = (float *)sqlite3_malloc64(sizeof(float) * dimension);
    if (!embedding) {
        sqlite_context_result_error(context, SQLITE_NOMEM, "Out of memory: failed to allocate embedding buffer of dimension %d", dimension);
        return;
    }
    
    // get token count
    int32_t n_tokens = -llama_tokenize(vocab, text, text_len, NULL, 0, true, false);
    if (n_tokens == 0) {
        sqlite3_free(embedding);
        sqlite_context_result_error(context, SQLITE_ERROR, "Tokenization failed: returned %d tokens", n_tokens);
        return;
    }
    if (ai->options.max_tokens > 0 && n_tokens > ai->options.max_tokens) {
        sqlite3_free(embedding);
        sqlite_context_result_error(context, SQLITE_TOOBIG, "Input too large: %d tokens exceeds max allowed (%d)", n_tokens, ai->options.max_tokens);
        return;
    }
    
    // check if memory allocation is needed
    if (n_tokens > ai->token_buffer_size) {
        int32_t n_count = (n_tokens > MIN_ALLOC_TOKEN) ? n_tokens : MIN_ALLOC_TOKEN;
        if (ai->token_buffer) sqlite3_free(ai->token_buffer);
        ai->token_buffer = sqlite3_malloc64(n_count * sizeof(llama_token));
        if (!ai->token_buffer) {
            sqlite3_free(embedding);
            sqlite_context_result_error(context, SQLITE_NOMEM, "Out of memory: failed to allocate %d tokens", n_count);
            return;
        }
        ai->token_buffer_size = n_count;
    }
    llama_token *tokens = ai->token_buffer;
    
    // tokenize input
    int n_actual = llama_tokenize(vocab, text, text_len, tokens, n_tokens, true, true);
    if (n_actual != n_tokens) {
        sqlite3_free(embedding);
        sqlite_context_result_error(context, SQLITE_ERROR, "Tokenization size mismatch: got %d tokens, expected %d", n_actual, n_tokens);
        return;
    }
    
    // set up batch for processing
    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    llama_seq_id seq_id = 0;
    for (int i = 0; i < n_tokens; ++i) {
        batch.token[batch.n_tokens] = tokens[i];
        batch.pos[batch.n_tokens] = i;
        batch.n_seq_id[batch.n_tokens]= 1;
        batch.seq_id[batch.n_tokens][0] = seq_id;
        batch.logits[batch.n_tokens]  = true;
        batch.n_tokens++;
    }
    
    // do real processing
    llama_memory_t memory = llama_get_memory(ctx);
    int32_t rc = (memory) ? llama_decode(ctx, batch) : llama_encode(ctx, batch);
    if (rc < 0) {
        sqlite3_free(embedding);
        llama_batch_free(batch);
        sqlite_context_result_error(context, SQLITE_ERROR, "Model decode failed during embedding generation");
        return;
    }
    
    // retrieve embeddings
    const float *result = llama_get_embeddings(ctx);
    if (result == NULL) {
        sqlite3_free(embedding);
        llama_batch_free(batch);
        sqlite_context_result_error(context, SQLITE_ERROR, "Failed to retrieve embedding vector from model");
        return;
    }
    
    // check if normalization is needed (default true)
    (ai->options.normalize_embedding) ? ai_embed_normalize(result, embedding, dimension) : memcpy(embedding, result, sizeof(float) * dimension);
    
    // check if JSON output is set
    if (ai->options.json_output) {
        sqlite3_str *s = sqlite3_str_new(NULL);
        sqlite3_str_appendchar(s, 1, '[');
        for (int i = 0; i < dimension; i++) {
            if (i != 0) sqlite3_str_appendchar(s, 1, ',');
            sqlite3_str_appendf(s, "%f", embedding[i]);
        }
        sqlite3_str_appendchar(s, 1, ']');
        
        char *json = sqlite3_str_finish(s);
        (json) ? sqlite3_result_text(context, json, -1, sqlite3_free) : sqlite3_result_null(context);
        sqlite3_free(embedding);
    } else {
        sqlite3_result_blob(context, embedding, sizeof(float) * dimension, sqlite3_free);
    }
    
    llama_batch_free(batch);
}

static void ai_text_embed (sqlite3_context *context, int argc, sqlite3_value **argv) {
    if (ai_common_args_check(context, "ai_text_embed", argc, argv, true) == false) return;
    
    const char *text = (const char *)sqlite3_value_text(argv[0]);
    int32_t text_len = (int32_t)sqlite3_value_bytes(argv[0]);
    const char *model_options = (argc == 2) ? (const char *)sqlite3_value_text(argv[1]) : NULL;
    
    ai_context *ai = (ai_context *)sqlite3_user_data(context);
    if (parse_keyvalue_string(context, model_options, ai_options_callback, &ai->options) == false) return;
    
    if (!text || text_len == 0) return;
    ai_text_embed_run(context, text, text_len);
}

// MARK: - Text Generation -

static void ai_text_run (sqlite3_context *context, const char *text, int32_t text_len) {
    ai_context *ai = (ai_context *)sqlite3_user_data(context);
    
    // sanity check vocab
    const struct llama_vocab *vocab = llama_model_get_vocab(ai->model);
    if (!vocab) {
        sqlite_context_result_error(context, SQLITE_ERROR, "Failed to extract vocabulary from the model");
        return;
    }
    
    // find the number of tokens in the prompt
    const int n_prompt = -llama_tokenize(vocab, text, text_len, NULL, 0, true, true);
    if (n_prompt == 0) {
        sqlite_context_result_error(context, SQLITE_ERROR, "Unable to extract number of tokens from prompt");
        return;
    }
    
    // allocate space for the tokens and tokenize the prompt
    llama_token *tokens = (llama_token *)sqlite3_malloc(n_prompt * sizeof(llama_token));
    if (!tokens) {
        sqlite_context_result_error(context, SQLITE_NOMEM, "Out of memory: failed to allocate %d tokens", n_prompt);
        return;
    }
    
    int n_actual = llama_tokenize(vocab, text, text_len, tokens, n_prompt, true, true); // TODO: false, false?
    if (n_actual != n_prompt) {
        sqlite3_free(tokens);
        sqlite_context_result_error(context, SQLITE_ERROR, "Tokenization size mismatch: got %d tokens, expected %d", n_actual, n_prompt);
        return;
    }
    
    // sanity check context
    int n_predict = (ai->options.n_predict > 0) ? ai->options.n_predict : NPREDICT_DEFAULT_VALUE;
    if (!ai->ctx) {
        ai->options.context_size = n_prompt + n_predict - 1; // set both n_ctx and n_batch
        ai->ctx = ai_context_check(context);
    }
    if (!ai->ctx) return;
    
    struct llama_context *ctx = ai->ctx;
    if (ctx == NULL) {
        sqlite3_free(tokens);
        sqlite_context_result_error(context, SQLITE_ERROR, "Failed to create the llama_context");
        return;
    }
    
    // initialize the sampler
    bool sampler_already_setup = (ai->sampler != NULL);
    struct llama_sampler *sampler = ai_sampler_check(context);
    if (!sampler) return;
    if (!sampler_already_setup) {
        // no sampler was setup, so initialize it with some default values
        llama_sampler_chain_add(sampler, llama_sampler_init_penalties(64, 1.1, 0, 0));
        llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
    }
    
    // prepare a batch for the prompt
    struct llama_batch batch = llama_batch_get_one(tokens, n_prompt);
    
    int n_decode = 0;
    llama_token new_token_id;
    buffer_t buffer;
    uint32_t buffer_size = ((n_prompt + n_predict) * MAX_TOKEN_TEXT_LEN); // should be more than enough to avoid a reallocation
    if (!buffer_create(&buffer, buffer_size)) {
        sqlite_context_result_error(context, SQLITE_NOMEM, "Out of memory: failed to allocate buffer (%d bytes)", buffer_size);
        goto cleanup;
    }
    
    for (int n_pos = 0; n_pos + batch.n_tokens < n_prompt + n_predict; ) {
        // evaluate the current batch with the transformer model
        if (llama_decode(ctx, batch)) {
            sqlite_context_result_error(context, SQLITE_ERROR, "Failed to execute the decoding function");
            goto cleanup;
        }

        n_pos += batch.n_tokens;

        // sample the next token
        new_token_id = llama_sampler_sample(sampler, ctx, -1);

        // is it an end of generation?
        if (llama_vocab_is_eog(vocab, new_token_id)) {
            break;
        }

        char buf[MAX_TOKEN_TEXT_LEN];
        int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
        if (n < 0) {
            sqlite_context_result_error(context, SQLITE_ERROR, "Failed to convert token to piece (%d)", n);
            goto cleanup;
        }
        
        if (buffer_append(&buffer, buf, n) == false) {
            sqlite_context_result_error(context, SQLITE_NOMEM, "Out of memory: failed to allocate and append buffer (%d bytes)", buffer.capacity + n);
            goto cleanup;
        }
        
        // print token as string
        // fwrite(buf, 1, n, stdout);
        // fflush(stdout);

        // prepare the next batch with the sampled token
        batch = llama_batch_get_one(&new_token_id, 1);

        n_decode += 1;
    }
    
    sqlite3_result_text(context, buffer.data, buffer.length, sqlite3_free);
    
cleanup:
    sqlite3_free(tokens);
    if (!sampler_already_setup) llama_sampler_free(sampler);
}

static void ai_text_generate (sqlite3_context *context, int argc, sqlite3_value **argv) {
    if (ai_common_args_check(context, "ai_text_generate", argc, argv, true) == false) return;
    
    const char *text = (const char *)sqlite3_value_text(argv[0]);
    int32_t text_len = (int32_t)sqlite3_value_bytes(argv[0]);
    const char *options = (argc == 2) ? (const char *)sqlite3_value_text(argv[1]) : NULL;
    
    ai_context *ai = (ai_context *)sqlite3_user_data(context);
    if (parse_keyvalue_string(context, options, ai_options_callback, &ai->options) == false) return;
    
    if (!text || text_len == 0) return;
    ai_text_run(context, text, text_len);
}

// MARK: - Chat Virtual Table -

/*
    (NO) INSERT INTO ai_chat('Hello')
    (NO) INSERT INTO ai_chat(prompt) VALUES ('Hello')
    ---> SELECT reply FROM ai_chat('Hello', opt);
 */

static int vt_chat_connect (sqlite3 *db, void *pAux, int argc, const char *const *argv, sqlite3_vtab **ppVtab, char **pzErr) {
    int rc = sqlite3_declare_vtab(db, "CREATE TABLE x(reply);");
    if (rc != SQLITE_OK) return rc;
    
    ai_vtab *vtab = (ai_vtab *)sqlite3_malloc(sizeof(ai_vtab));
    if (!vtab) return SQLITE_NOMEM;
    
    memset(vtab, 0, sizeof(ai_vtab));
    vtab->ai = (ai_context *)pAux;
    
    *ppVtab = (sqlite3_vtab *)vtab;
    return SQLITE_OK;
}

static int vt_chat_disconnect (sqlite3_vtab *pVtab) {
    ai_vtab *vtab = (ai_vtab *)pVtab;
    sqlite3_free(vtab);
    return SQLITE_OK;
}

static int vt_chat_best_index (sqlite3_vtab *tab, sqlite3_index_info *pIdxInfo) {
    pIdxInfo->estimatedCost = (double)1;
    pIdxInfo->estimatedRows = 100;
    pIdxInfo->orderByConsumed = 1;
    pIdxInfo->idxNum = 1;
    return SQLITE_OK;
}

static int vt_chat_cursor_open (sqlite3_vtab *pVtab, sqlite3_vtab_cursor **ppCursor) {
    ai_cursor *c = (ai_cursor *)sqlite3_malloc(sizeof(ai_cursor));
    if (!c) return SQLITE_NOMEM;
    
    memset(c, 0, sizeof(ai_cursor));
    ai_vtab *vtab = (ai_vtab *)pVtab;
    c->ai = vtab->ai;
    
    ai_context *ai = c->ai;
    int n_ctx = llama_n_ctx(ai->ctx);
    buffer_create(&c->formatted, n_ctx);
    
    *ppCursor = (sqlite3_vtab_cursor *)c;
    return SQLITE_OK;
}

static int vt_chat_cursor_close (sqlite3_vtab_cursor *cur){
    ai_cursor *c = (ai_cursor *)cur;
    if (c->formatted.data) sqlite3_free(c->formatted.data);
    
    if (c->messages.items) ai_messages_free(&c->messages);
    if (c->formatted.data) sqlite3_free(c->formatted.data);
    
    sqlite3_free(c);
    return SQLITE_OK;
}

static int vt_chat_cursor_next (sqlite3_vtab_cursor *cur){
    ai_cursor *c = (ai_cursor *)cur;
    c->rowid++;
    return SQLITE_OK;
}

static int vt_chat_cursor_eof (sqlite3_vtab_cursor *cur){
    ai_cursor *c = (ai_cursor *)cur;
    return (int)c->is_eog;
}

static int vt_chat_cursor_column (sqlite3_vtab_cursor *cur, sqlite3_context *context, int iCol) {
    ai_cursor *c = (ai_cursor *)cur;
    if (iCol == AI_COLUMN_REPLY) {
        ai_messages *messages = &c->messages;
        sqlite3_result_text(context, messages->items[messages->count-1].content, -1, SQLITE_TRANSIENT);
    }
    return SQLITE_OK;
}

static int vt_chat_cursor_rowid (sqlite3_vtab_cursor *cur, sqlite_int64 *pRowid) {
    ai_cursor *c = (ai_cursor *)cur;
    *pRowid = c->rowid;
    return SQLITE_OK;
}

static char *vt_chat_cursor_generate (sqlite3_vtab_cursor *cur, const char *text) {
    return NULL;
}

static int vt_chat_cursor_filter (sqlite3_vtab_cursor *cur, int idxNum, const char *idxStr, int argc, sqlite3_value **argv) {
    ai_cursor *c = (ai_cursor *)cur;
    ai_context *ai = c->ai;
    ai_vtab *vtab = c->vtab;
    
    // sanity check arguments
    if (argc != 1) {
        return sqlite_vtab_set_error(&vtab->base, "ai_chat expects %d arguments, but %d were provided.", 1, argc);
    }
    if (sqlite3_value_type(argv[0]) != SQLITE_TEXT) {
        return sqlite_vtab_set_error(&vtab->base, "ai_chat argument must be of type TEXT");
    }
    
    const char *input = (const char *)sqlite3_value_text(argv[0]);
    const char *template = llama_model_chat_template(ai->model, NULL);
    ai_messages *messages = &c->messages;
    buffer_t *formatted = &c->formatted;
    
    if (!ai_messages_append(messages, "user", input)) {
        return sqlite_vtab_set_error(&vtab->base, "failed to append message");
    }
    
    int new_len = llama_chat_apply_template(template, messages->items, messages->count, true, formatted->data, formatted->capacity);
    
    if (new_len < 0) {
        return sqlite_vtab_set_error(&vtab->base, "failed to apply chat template");
    }
    
    // extract just the new prompt
    int prompt_len = new_len - c->prev_len;
    char *prompt = sqlite3_malloc(prompt_len + 1);
    if (!prompt) {
        return sqlite_vtab_set_error(&vtab->base, "failed to allocate prompt buffer");
        
    }
    memcpy(prompt, formatted + c->prev_len, prompt_len);
    prompt[prompt_len] = 0;
    
    char *response = vt_chat_cursor_generate(cur, prompt);
    sqlite3_free(prompt);
    
    if (!ai_messages_append(messages, "assistant", response)) {
        sqlite_vtab_set_error(&vtab->base, "failed to append response");
        sqlite3_free(response);
        return SQLITE_ERROR;
    }
    sqlite3_free(response);
    
    c->prev_len = llama_chat_apply_template(template, messages->items, messages->count, false, NULL, 0);
    if (c->prev_len < 0) return sqlite_vtab_set_error(&vtab->base, "failed to apply chat template\n");
    
    return SQLITE_OK;
}

static sqlite3_module vt_chat = {
  /* iVersion    */ 0,
  /* xCreate     */ 0,
  /* xConnect    */ vt_chat_connect,
  /* xBestIndex  */ vt_chat_best_index,
  /* xDisconnect */ vt_chat_disconnect,
  /* xDestroy    */ 0,
  /* xOpen       */ vt_chat_cursor_open,
  /* xClose      */ vt_chat_cursor_close,
  /* xFilter     */ vt_chat_cursor_filter,
  /* xNext       */ vt_chat_cursor_next,
  /* xEof        */ vt_chat_cursor_eof,
  /* xColumn     */ vt_chat_cursor_column,
  /* xRowid      */ vt_chat_cursor_rowid,
  /* xUpdate     */ 0,
  /* xBegin      */ 0,
  /* xSync       */ 0,
  /* xCommit     */ 0,
  /* xRollback   */ 0,
  /* xFindMethod */ 0,
  /* xRename     */ 0,
  /* xSavepoint  */ 0,
  /* xRelease    */ 0,
  /* xRollbackTo */ 0,
  /* xShadowName */ 0,
  /* xIntegrity  */ 0
};

// MARK: -

static void ai_sampler_free (sqlite3_context *context, int argc, sqlite3_value **argv) {
    ai_context *ai = (ai_context *)sqlite3_user_data(context);
    if (ai->sampler) llama_sampler_free(ai->sampler);
    ai->sampler = NULL;
}

static void ai_sampler_create (sqlite3_context *context, int argc, sqlite3_value **argv) {
    ai_context *ai = (ai_context *)sqlite3_user_data(context);
    if (ai->sampler) llama_sampler_free(ai->sampler);
    ai->sampler = NULL;
    ai_sampler_check(context);
}

static void ai_sampler_init_greedy (sqlite3_context *context, int argc, sqlite3_value **argv) {
    ai_context *ai = (ai_context *)sqlite3_user_data(context);
    ai_sampler_check(context);
    if (ai->sampler) llama_sampler_chain_add(ai->sampler, llama_sampler_init_greedy());
}

static void ai_sampler_init_dist (sqlite3_context *context, int argc, sqlite3_value **argv) {
    if (argc != 0) {
        int types[] = {SQLITE_INTEGER};
        if (sqlite_sanity_function(context, "ai_sampler_init_dist", argc, argv, 1, types, true) == false) return;
    }
    
    ai_context *ai = (ai_context *)sqlite3_user_data(context);
    ai_sampler_check(context);
    if (ai->sampler) {
        int32_t seed = (argc == 1) ? (int32_t)sqlite3_value_int64(argv[0]) : (int32_t)LLAMA_DEFAULT_SEED;
        llama_sampler_chain_add(ai->sampler, llama_sampler_init_dist(seed));
    }
}

static void ai_sampler_init_top_k (sqlite3_context *context, int argc, sqlite3_value **argv) {
    int types[] = {SQLITE_INTEGER};
    if (sqlite_sanity_function(context, "ai_sampler_init_top_k", argc, argv, 1, types, true) == false) return;
    
    ai_context *ai = (ai_context *)sqlite3_user_data(context);
    ai_sampler_check(context);
    if (ai->sampler) {
        int32_t k = (int32_t)sqlite3_value_int64(argv[0]);
        llama_sampler_chain_add(ai->sampler, llama_sampler_init_top_k(k));
    }
}

static void ai_sampler_init_top_p (sqlite3_context *context, int argc, sqlite3_value **argv) {
    int types[] = {SQLITE_FLOAT, SQLITE_INTEGER};
    if (sqlite_sanity_function(context, "ai_sampler_init_top_p", argc, argv, 2, types, true) == false) return;
    
    ai_context *ai = (ai_context *)sqlite3_user_data(context);
    ai_sampler_check(context);
    if (ai->sampler) {
        float p = (float)sqlite3_value_double(argv[0]);
        size_t min_keep = (size_t)sqlite3_value_int64(argv[1]);
        llama_sampler_chain_add(ai->sampler, llama_sampler_init_top_p(p, min_keep));
    }
}

static void ai_sampler_init_min_p (sqlite3_context *context, int argc, sqlite3_value **argv) {
    int types[] = {SQLITE_FLOAT, SQLITE_INTEGER};
    if (sqlite_sanity_function(context, "ai_sampler_init_min_p", argc, argv, 2, types, true) == false) return;
    
    ai_context *ai = (ai_context *)sqlite3_user_data(context);
    ai_sampler_check(context);
    if (ai->sampler) {
        float p = (float)sqlite3_value_double(argv[0]);
        size_t min_keep = (size_t)sqlite3_value_int64(argv[1]);
        llama_sampler_chain_add(ai->sampler, llama_sampler_init_min_p(p, min_keep));
    }
}

static void ai_sampler_init_typical (sqlite3_context *context, int argc, sqlite3_value **argv) {
    int types[] = {SQLITE_FLOAT, SQLITE_INTEGER};
    if (sqlite_sanity_function(context, "llama_sampler_init_typical", argc, argv, 2, types, true) == false) return;
    
    ai_context *ai = (ai_context *)sqlite3_user_data(context);
    ai_sampler_check(context);
    if (ai->sampler) {
        float p = (float)sqlite3_value_double(argv[0]);
        size_t min_keep = (size_t)sqlite3_value_int64(argv[1]);
        llama_sampler_chain_add(ai->sampler, llama_sampler_init_typical(p, min_keep));
    }
}

static void ai_sampler_init_temp (sqlite3_context *context, int argc, sqlite3_value **argv) {
    int types[] = {SQLITE_FLOAT};
    if (sqlite_sanity_function(context, "llama_sampler_init_temp", argc, argv, 1, types, true) == false) return;
    
    ai_context *ai = (ai_context *)sqlite3_user_data(context);
    ai_sampler_check(context);
    if (ai->sampler) {
        float t = (float)sqlite3_value_double(argv[0]);
        llama_sampler_chain_add(ai->sampler, llama_sampler_init_temp(t));
    }
}

static void ai_sampler_init_temp_ext (sqlite3_context *context, int argc, sqlite3_value **argv) {
    int types[] = {SQLITE_FLOAT, SQLITE_FLOAT, SQLITE_FLOAT};
    if (sqlite_sanity_function(context, "ai_sampler_init_temp_ext", argc, argv, 3, types, true) == false) return;
    
    ai_context *ai = (ai_context *)sqlite3_user_data(context);
    ai_sampler_check(context);
    if (ai->sampler) {
        float t = (float)sqlite3_value_double(argv[0]);
        float delta = (float)sqlite3_value_double(argv[1]);
        float exponent = (float)sqlite3_value_double(argv[2]);
        llama_sampler_chain_add(ai->sampler, llama_sampler_init_temp_ext(t, delta, exponent));
    }
}

static void ai_sampler_init_xtc (sqlite3_context *context, int argc, sqlite3_value **argv) {
    int types[] = {SQLITE_FLOAT, SQLITE_FLOAT, SQLITE_INTEGER, SQLITE_INTEGER};
    if (sqlite_sanity_function(context, "ai_sampler_init_xtc", argc, argv, 4, types, true) == false) return;
    
    ai_context *ai = (ai_context *)sqlite3_user_data(context);
    ai_sampler_check(context);
    if (ai->sampler) {
        float p = (float)sqlite3_value_double(argv[0]);
        float t = (float)sqlite3_value_double(argv[1]);
        size_t min_keep = (size_t)sqlite3_value_int64(argv[2]);
        uint32_t seed = (uint32_t)sqlite3_value_int64(argv[3]);
        llama_sampler_chain_add(ai->sampler, llama_sampler_init_xtc(p, t, min_keep, seed));
    }
}

static void ai_sampler_init_top_n_sigma (sqlite3_context *context, int argc, sqlite3_value **argv) {
    int types[] = {SQLITE_FLOAT};
    if (sqlite_sanity_function(context, "ai_sampler_init_top_n_sigma", argc, argv, 1, types, true) == false) return;
    
    ai_context *ai = (ai_context *)sqlite3_user_data(context);
    ai_sampler_check(context);
    if (ai->sampler) {
        float n = (float)sqlite3_value_double(argv[0]);
        llama_sampler_chain_add(ai->sampler, llama_sampler_init_top_n_sigma(n));
    }
}

static void ai_sampler_init_mirostat (sqlite3_context *context, int argc, sqlite3_value **argv) {
    int types[] = {SQLITE_INTEGER, SQLITE_FLOAT, SQLITE_FLOAT, SQLITE_INTEGER};
    if (sqlite_sanity_function(context, "ai_sampler_init_mirostat", argc, argv, 4, types, true) == false) return;
    
    ai_context *ai = (ai_context *)sqlite3_user_data(context);
    const struct llama_vocab *vocab = llama_model_get_vocab(ai->model);
    if (!vocab) {
        sqlite_context_result_error(context, SQLITE_ERROR, "Unable to get vocab from current model.");
        return;
    }
    
    ai_sampler_check(context);
    if (ai->sampler) {
        uint32_t seed = (uint32_t)sqlite3_value_int64(argv[0]);
        float tau = (float)sqlite3_value_double(argv[1]);
        float eta = (float)sqlite3_value_double(argv[2]);
        int32_t m = (int32_t)sqlite3_value_int64(argv[3]);
        llama_sampler_chain_add(ai->sampler, llama_sampler_init_mirostat(llama_vocab_n_tokens(vocab), seed, tau, eta, m));
    }
}

static void ai_sampler_init_mirostat_v2 (sqlite3_context *context, int argc, sqlite3_value **argv) {
    int types[] = {SQLITE_INTEGER, SQLITE_FLOAT, SQLITE_FLOAT};
    if (sqlite_sanity_function(context, "ai_sampler_init_mirostat_v2", argc, argv, 3, types, true) == false) return;
    
    ai_context *ai = (ai_context *)sqlite3_user_data(context);
    ai_sampler_check(context);
    if (ai->sampler) {
        uint32_t seed = (uint32_t)sqlite3_value_int64(argv[0]);
        float tau = (float)sqlite3_value_double(argv[1]);
        float eta = (float)sqlite3_value_double(argv[2]);
        llama_sampler_chain_add(ai->sampler, llama_sampler_init_mirostat_v2(seed, tau, eta));
    }
}

static void ai_sampler_init_grammar (sqlite3_context *context, int argc, sqlite3_value **argv) {
    int types[] = {SQLITE_TEXT, SQLITE_TEXT};
    if (sqlite_sanity_function(context, "ai_sampler_init_grammar", argc, argv, 2, types, true) == false) return;
    
    ai_context *ai = (ai_context *)sqlite3_user_data(context);
    const struct llama_vocab *vocab = llama_model_get_vocab(ai->model);
    if (!vocab) {
        sqlite_context_result_error(context, SQLITE_ERROR, "Unable to get vocab from current model.");
        return;
    }
    
    ai_sampler_check(context);
    if (ai->sampler) {
        const char *grammar_str = (const char *)sqlite3_value_text(argv[0]);
        const char *grammar_root = (const char *)sqlite3_value_text(argv[1]);
        llama_sampler_chain_add(ai->sampler, llama_sampler_init_grammar(vocab, grammar_str, grammar_root));
    }
}

static void ai_sampler_init_infill (sqlite3_context *context, int argc, sqlite3_value **argv) {
    ai_context *ai = (ai_context *)sqlite3_user_data(context);
    const struct llama_vocab *vocab = llama_model_get_vocab(ai->model);
    if (!vocab) {
        sqlite_context_result_error(context, SQLITE_ERROR, "Unable to get vocab from current model.");
        return;
    }
    
    ai_sampler_check(context);
    if (ai->sampler) {
        llama_sampler_chain_add(ai->sampler, llama_sampler_init_infill(vocab));
    }
}


// llama_sampler_init_dry(const struct llama_vocab *  vocab, int32_t    n_ctx_train, float    dry_multiplier, float    dry_base, int32_t    dry_allowed_length, int32_t    dry_penalty_last_n, const char ** seq_breakers, size_t    num_breakers)

// llama_sampler_init_logit_bias( int32_t   n_vocab, int32_t   n_logit_bias, const llama_logit_bias * logit_bias)


static void ai_sampler_init_penalties (sqlite3_context *context, int argc, sqlite3_value **argv) {
    int types[] = {SQLITE_INTEGER, SQLITE_FLOAT, SQLITE_FLOAT, SQLITE_FLOAT};
    if (sqlite_sanity_function(context, "ai_sampler_init_penalties", argc, argv, 4, types, true) == false) return;
    
    ai_context *ai = (ai_context *)sqlite3_user_data(context);
    ai_sampler_check(context);
    if (ai->sampler) {
        int32_t penalty_last_n = (int32_t)sqlite3_value_int64(argv[0]);
        float penalty_repeat = (float)sqlite3_value_double(argv[1]);
        float penalty_freq = (float)sqlite3_value_double(argv[2]);
        float penalty_present = (float)sqlite3_value_double(argv[3]);
        llama_sampler_chain_add(ai->sampler, llama_sampler_init_penalties(penalty_last_n, penalty_repeat, penalty_freq, penalty_present));
    }
}

// MARK: -

static void ai_context_free (sqlite3_context *context, int argc, sqlite3_value **argv) {
    ai_context *ai = (ai_context *)sqlite3_user_data(context);
    if (ai->ctx) llama_free(ai->ctx);
    ai->ctx = NULL;
}

static void ai_context_create (sqlite3_context *context, int argc, sqlite3_value **argv) {
    // sanity check arguments
    if ((argc > 0) && ai_common_args_check(context, "ai_context_create", argc, argv, true) == false) return;
    const char *options = (argc == 1) ? (const char *)sqlite3_value_text(argv[0]) : NULL;
    
    ai_context *ai = (ai_context *)sqlite3_user_data(context);
    if (parse_keyvalue_string(context, options, ai_options_callback, &ai->options) == false) return;
    
    if (ai->ctx) ai_context_free(context, 0, NULL);
    ai->ctx = ai_context_check(context);
}

static void ai_model_set (sqlite3_context *context, int argc, sqlite3_value **argv) {
    // sanity check arguments
    if (ai_common_args_check(context, "ai_model_set", argc, argv, false) == false) return;
    
    // TODO: model_arg can be a path or a name inside the ai_models table
    char path[MAX_PATH];
    const char *model_arg = (const char *)sqlite3_value_text(argv[0]);
    const char *p = db_check_model(sqlite3_context_db_handle(context), model_arg, path, sizeof(path));
    const char *model_path = (p) ? p : model_arg;
    const char *model_options = (argc == 2) ? (const char *)sqlite3_value_text(argv[1]) : NULL;
    
    ai_context *ai = (ai_context *)sqlite3_user_data(context);
    if (parse_keyvalue_string(context, model_options, ai_options_callback, &ai->options) == false) return;
    
    struct llama_model_params model_params = llama_model_default_params();
    ai_set_model_options(&model_params, &ai->options);
    struct llama_model *model = llama_model_load_from_file(model_path, model_params);
    if (!model) {
        sqlite_context_result_error(context, SQLITE_ERROR, "Unable to load model from file %s", model_path);
        return;
    }
    
    if (ai->model) llama_model_free(ai->model);
    ai->model = model;
}

static void ai_log_info (sqlite3_context *context, int argc, sqlite3_value **argv) {
    bool info_value = false;
    if ((argc == 1) && (sqlite3_value_type(argv[0]) == SQLITE_INTEGER)) info_value = !(sqlite3_value_int(argv[0]) == 0);
    
    ai_context *ai = (ai_context *)sqlite3_user_data(context);
    ai->options.log_info = info_value;
}

static void ai_version (sqlite3_context *context, int argc, sqlite3_value **argv) {
    sqlite3_result_text(context, SQLITE_AI_VERSION, -1, NULL);
}

// MARK: -

SQLITE_AI_API int sqlite3_ai_init (sqlite3 *db, char **pzErrMsg, const sqlite3_api_routines *pApi) {
    #ifndef SQLITE_CORE
    SQLITE_EXTENSION_INIT2(pApi);
    #endif
    
    // initialize the llama + ggml backend
    static bool once = false;
    if (once == false) {
        llama_backend_init();
        once = true;
    }
    
    // create temp log table
    sqlite3_exec(db, LOG_TABLE_DECLARATION, NULL, NULL, NULL);
    
    // init context
    void *ctx = ai_sqlite_context_create(db);
    if (!ctx) {
        if (pzErrMsg) *pzErrMsg = sqlite3_mprintf("Out of memory: failed to allocate AI extension context.");
        return SQLITE_NOMEM;
    }
    
    // set logger
    llama_log_set(ai_logger, ctx);
    
    // register public functions
    int rc = SQLITE_OK;
    rc = sqlite3_create_function_v2(db, "ai_version", 0, SQLITE_UTF8, ctx, ai_version, NULL, NULL, ai_sqlite_context_free);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function_v2(db, "ai_log_info", 1, SQLITE_UTF8, ctx, ai_log_info, NULL, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function(db, "ai_model_set", 1, SQLITE_UTF8, ctx, ai_model_set, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function(db, "ai_model_set", 2, SQLITE_UTF8, ctx, ai_model_set, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function(db, "ai_context_create", 0, SQLITE_UTF8, ctx, ai_context_create, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function(db, "ai_context_create", 1, SQLITE_UTF8, ctx, ai_context_create, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function(db, "ai_context_free", 0, SQLITE_UTF8, ctx, ai_context_free, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function(db, "ai_sampler_create", 0, SQLITE_UTF8, ctx, ai_sampler_create, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function(db, "ai_sampler_free", 0, SQLITE_UTF8, ctx, ai_sampler_free, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function(db, "ai_sampler_init_greedy", 0, SQLITE_UTF8, ctx, ai_sampler_init_greedy, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function(db, "ai_sampler_init_dist", 0, SQLITE_UTF8, ctx, ai_sampler_init_dist, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function(db, "ai_sampler_init_dist", 1, SQLITE_UTF8, ctx, ai_sampler_init_dist, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function(db, "ai_sampler_init_top_k", 1, SQLITE_UTF8, ctx, ai_sampler_init_top_k, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function(db, "ai_sampler_init_top_p", 2, SQLITE_UTF8, ctx, ai_sampler_init_top_p, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function(db, "ai_sampler_init_min_p", 2, SQLITE_UTF8, ctx, ai_sampler_init_min_p, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function(db, "ai_sampler_init_typical", 2, SQLITE_UTF8, ctx, ai_sampler_init_typical, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function(db, "ai_sampler_init_temp", 1, SQLITE_UTF8, ctx, ai_sampler_init_temp, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function(db, "ai_sampler_init_temp_ext", 3, SQLITE_UTF8, ctx, ai_sampler_init_temp_ext, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function(db, "ai_sampler_init_xtc", 4, SQLITE_UTF8, ctx, ai_sampler_init_xtc, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function(db, "ai_sampler_init_top_n_sigma", 1, SQLITE_UTF8, ctx, ai_sampler_init_top_n_sigma, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function(db, "ai_sampler_init_mirostat", 4, SQLITE_UTF8, ctx, ai_sampler_init_mirostat, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function(db, "ai_sampler_init_mirostat_v2", 3, SQLITE_UTF8, ctx, ai_sampler_init_mirostat_v2, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function(db, "ai_sampler_init_grammar", 2, SQLITE_UTF8, ctx, ai_sampler_init_grammar, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function(db, "ai_sampler_init_infill", 0, SQLITE_UTF8, ctx, ai_sampler_init_infill, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function(db, "ai_sampler_init_penalties", 4, SQLITE_UTF8, ctx, ai_sampler_init_penalties, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    // TODO
    // llama_sampler_init_dry(const struct llama_vocab *  vocab, int32_t    n_ctx_train, float    dry_multiplier, float    dry_base, int32_t    dry_allowed_length, int32_t    dry_penalty_last_n, const char ** seq_breakers, size_t    num_breakers)
    // llama_sampler_init_logit_bias( int32_t   n_vocab, int32_t   n_logit_bias, const llama_logit_bias * logit_bias)
    
    rc = sqlite3_create_function(db, "ai_text_embed", 1, SQLITE_UTF8, ctx, ai_text_embed, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function(db, "ai_text_embed", 2, SQLITE_UTF8, ctx, ai_text_embed, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function(db, "ai_text_generate", 1, SQLITE_UTF8, ctx, ai_text_generate, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function(db, "ai_text_generate", 2, SQLITE_UTF8, ctx, ai_text_generate, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    //rc = sqlite3_create_module(db, "ai_chat", &vt_chat, ctx);
    //if (rc != SQLITE_OK) goto cleanup;
    
cleanup:
    return rc;
}
