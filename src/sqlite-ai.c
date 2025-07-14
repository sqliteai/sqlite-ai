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
#define MIN_ALLOC_TOKEN                         4096
#define MIN_ALLOC_PROMPT                        4096
#define MIN_ALLOC_RESPONSE                      4096
#define MAX_PATH                                4096
#define MAX_TOKEN_TEXT_LEN                      128     // according to ChatGPT 32 would be safe for all common tokenizers
#define MIN_ALLOC_MESSAGES                      256

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
    
    // ** EMBEDDING **
    bool                        generate_embedding; // if true, extract embeddings (together with logits)
    bool                        normalize_embedding;// if true, embeddings are normalized
    bool                        json_output;        // if true, embedding result is converted to JSON
    
    // ** CUSTOM **
    int32_t                     max_tokens;         // to control max allowed tokens to generate (to control user's input size)
    
} llm_options;

typedef struct {
    llama_chat_message *items;
    size_t count;
    size_t capacity;
} ai_messages;

typedef struct {
    // sqlite
    sqlite3                     *db;
    sqlite3_vtab                *vtab;
    sqlite3_context             *context;
    
    // llama
    struct llama_model          *model;
    struct llama_context        *ctx;
    struct llama_sampler        *sampler;
    llm_options                 options;
    
    // chat
    struct {
        char                    uuid[UUID_STR_MAXLEN];
        const char              *template;
        const struct llama_vocab*vocab;
        
        ai_messages             messages;
        buffer_t                formatted;
        buffer_t                response;
        char                    *prompt;
        int32_t                 prev_len;
        llama_token             *tokens;
        int32_t                 ntokens;
        llama_batch             batch;
        
        llama_token             token_id;
        char                    token_text[MAX_TOKEN_TEXT_LEN];
        int32_t                 token_len;
        int32_t                 token_count;
    } chat;
} ai_context;

typedef struct {
    sqlite3_vtab                base;               // Base class - must be first
    ai_context                  *ai;
} ai_vtab;

typedef struct {
    sqlite3_vtab_cursor         base;               // Base class - must be first
    ai_vtab                     *vtab;
    ai_context                  *ai;
    
    bool                        is_eog;
    sqlite_int64                rowid;
} ai_cursor;

const char *ROLE_USER       = "user";
const char *ROLE_ASSISTANT  = "assistant";

// MARK: -

void llm_set_model_options (struct llama_model_params *model_params, llm_options *options) {
    // number of layers to store in VRAM
    if (options->gpu_layers) model_params->n_gpu_layers = options->gpu_layers;
    if (options->split_mode) model_params->split_mode = options->split_mode;
    if (options->main_gpu) model_params->main_gpu = options->main_gpu;
    if (options->vocab_only) model_params->vocab_only = options->vocab_only;
    if (options->use_mmap) model_params->use_mmap = options->use_mmap;
    if (options->use_mlock) model_params->use_mlock = options->use_mlock;
    if (options->check_tensors) model_params->check_tensors = options->check_tensors;
}

void llm_set_context_options (struct llama_context_params *llama_context, llm_options *options) {
    if (options->generate_embedding) llama_context->embeddings = true;
    if (options->context_size) {
        llama_context->n_ctx = options->context_size;
        llama_context->n_batch = options->context_size;
    }
}

static void llm_options_init (llm_options *options) {
    memset(options, 0, sizeof(llm_options));
    
    options->normalize_embedding = true;
    options->max_tokens = 0;    // no limits
    options->log_info = false;  // disable INFO messages logging
}

static bool llm_options_callback (void *xdata, const char *key, int key_len, const char *value, int value_len) {
    llm_options *options = (llm_options *)xdata;
    
    // sanity check (ignore malformed key/value)
    if (!key || key_len == 0) return true;
    if (!value || value_len == 0) return true;
    
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

struct llama_sampler *llm_sampler_check (ai_context *ai) {
    if (ai->sampler) return ai->sampler;
    
    struct llama_sampler_chain_params sampler_params = llama_sampler_chain_default_params();
    struct llama_sampler *sampler = llama_sampler_chain_init(sampler_params);
    if (!sampler) {
        sqlite_common_set_error(ai->context, ai->vtab, SQLITE_ERROR, "Unable to create sampler");
        return NULL;
    }
    ai->sampler = sampler;
    return sampler;
}

// MARK: -

void *ai_create (sqlite3 *db) {
    ai_context *ai = (ai_context *)sqlite3_malloc(sizeof(ai_context));
    if (ai) {
        llm_options_init(&ai->options);
        ai->db = db;
    }
    return ai;
}

static void ai_cleanup (void *ctx) {
    if (!ctx) return;
    ai_context *ai = (ai_context *)ctx;
    
    // disable logger first
    ai->db = NULL;
    
    if (ai->model) llama_model_free(ai->model);
    if (ai->ctx) llama_free(ai->ctx);
    if (ai->sampler) llama_sampler_free(ai->sampler);
    llm_options_init(&ai->options);
    
    ai->model = NULL;
    ai->ctx = NULL;
    ai->sampler = NULL;
}


void ai_logger (enum ggml_log_level level, const char *text, void *user_data) {
    ai_context *ai = (ai_context *)user_data;
    if (ai->db == NULL) return;
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
    
    // DEBUG
    // printf("%s %s\n", type, text);
    
    const char *values[] = {type, text};
    int types[] = {SQLITE_TEXT, SQLITE_TEXT};
    int lens[] = {-1, -1};
    sqlite_db_write(NULL, ai->db, LOG_TABLE_INSERT_STMT, values, types, lens, 2);
}

bool ai_model_check (sqlite3_context *context) {
    ai_context *ai = (ai_context *)sqlite3_user_data(context);
    if (!ai) return false;
    return (ai->model != NULL);
}

struct llama_context *ai_context_check (ai_context *ai) {
    if (ai->ctx) return ai->ctx;
    
    struct llama_context_params ctx_params = llama_context_default_params();
    llm_set_context_options(&ctx_params, &ai->options);
    
    struct llama_context *ctx = llama_init_from_model(ai->model, ctx_params);
    if (!ctx) {
        sqlite_common_set_error(ai->context, ai->vtab, SQLITE_ERROR, "Unable to create context from model");
        return NULL;
    }
    
    return ctx;
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

// MARK: - Chat Messages -

bool llm_messages_append (ai_messages *list, const char *role, const char *content, bool duplicate_role, bool duplicate_content) {
    if (list->count >= list->capacity) {
        size_t new_cap = list->capacity ? list->capacity * 2 : MIN_ALLOC_MESSAGES;
        llama_chat_message *new_items = sqlite3_realloc64(list->items, new_cap * sizeof(llama_chat_message));
        if (!new_items) return false;
        
        list->items = new_items;
        list->capacity = new_cap;
    }

    list->items[list->count].role = (duplicate_role) ? sqlite_strdup(role) : role;
    list->items[list->count].content = (duplicate_content) ? sqlite_strdup(content) : content;
    list->items[list->count].role_tofree = duplicate_role;
    list->count += 1;
    return true;
}

void llm_messages_free (ai_messages *list) {
    for (size_t i = 0; i < list->count; ++i) {
        // content is always to free
        if (list->items[list->count].role_tofree) sqlite3_free((char *)list->items[i].role);
        sqlite3_free((char *)list->items[i].content);
    }
    sqlite3_free(list->items);
    
    list->items = NULL;
    list->count = 0;
    list->capacity = 0;
}

// MARK: - Text Embedding -

static void llm_embed_normalize (const float *src, float *dest, int dim) {
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

static void llm_embed_generate_run (sqlite3_context *context, const char *text, int32_t text_len) {
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
        ai->ctx = ai_context_check(ai);
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
    
    // allocate memory for tokens
    llama_token *tokens = sqlite3_malloc64(n_tokens * sizeof(llama_token));
    if (!tokens) {
        sqlite3_free(embedding);
        sqlite_context_result_error(context, SQLITE_NOMEM, "Out of memory: failed to allocate tokens memory of size %lld", (long long)(n_tokens * sizeof(llama_token)));
        return;
    }
    
    // tokenize input
    int n_actual = llama_tokenize(vocab, text, text_len, tokens, n_tokens, true, true);
    if (n_actual != n_tokens) {
        sqlite3_free(tokens);
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
        sqlite3_free(tokens);
        sqlite3_free(embedding);
        llama_batch_free(batch);
        sqlite_context_result_error(context, SQLITE_ERROR, "Model decode failed during embedding generation");
        return;
    }
    
    // retrieve embeddings
    const float *result = llama_get_embeddings(ctx);
    if (result == NULL) {
        sqlite3_free(tokens);
        sqlite3_free(embedding);
        llama_batch_free(batch);
        sqlite_context_result_error(context, SQLITE_ERROR, "Failed to retrieve embedding vector from model");
        return;
    }
    
    // check if normalization is needed (default true)
    (ai->options.normalize_embedding) ? llm_embed_normalize(result, embedding, dimension) : memcpy(embedding, result, sizeof(float) * dimension);
    
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
    
    sqlite3_free(tokens);
    llama_batch_free(batch);
}

static void llm_embed_generate (sqlite3_context *context, int argc, sqlite3_value **argv) {
    if (ai_common_args_check(context, "llm_embed_generate", argc, argv, true) == false) return;
    
    const char *text = (const char *)sqlite3_value_text(argv[0]);
    int32_t text_len = (int32_t)sqlite3_value_bytes(argv[0]);
    const char *model_options = (argc == 2) ? (const char *)sqlite3_value_text(argv[1]) : NULL;
    
    ai_context *ai = (ai_context *)sqlite3_user_data(context);
    if (parse_keyvalue_string(model_options, llm_options_callback, &ai->options) == false) return;
    
    if (!text || text_len == 0) return;
    llm_embed_generate_run(context, text, text_len);
}

// MARK: - Text Generation -

static void llm_text_run (sqlite3_context *context, const char *text, int32_t text_len) {
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
    
    int n_actual = llama_tokenize(vocab, text, text_len, tokens, n_prompt, true, true);
    if (n_actual != n_prompt) {
        sqlite3_free(tokens);
        sqlite_context_result_error(context, SQLITE_ERROR, "Tokenization size mismatch: got %d tokens, expected %d", n_actual, n_prompt);
        return;
    }
    
    // sanity check context
    int n_predict = (ai->options.n_predict > 0) ? ai->options.n_predict : NPREDICT_DEFAULT_VALUE;
    if (!ai->ctx) {
        ai->options.context_size = n_prompt + n_predict - 1; // set both n_ctx and n_batch
        ai->ctx = ai_context_check(ai);
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
    struct llama_sampler *sampler = llm_sampler_check(ai);
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
        
        if (buffer_append(&buffer, buf, n, true) == false) {
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

static void llm_text_generate (sqlite3_context *context, int argc, sqlite3_value **argv) {
    if (ai_common_args_check(context, "llm_text_generate", argc, argv, true) == false) return;
    
    const char *text = (const char *)sqlite3_value_text(argv[0]);
    int32_t text_len = (int32_t)sqlite3_value_bytes(argv[0]);
    const char *options = (argc == 2) ? (const char *)sqlite3_value_text(argv[1]) : NULL;
    
    ai_context *ai = (ai_context *)sqlite3_user_data(context);
    if (parse_keyvalue_string(options, llm_options_callback, &ai->options) == false) {
        sqlite_context_result_error(context, SQLITE_ERROR, "An error occurred while parsing options (%s)", options);
        return;
    }
        
    if (!text || text_len == 0) return;
    llm_text_run(context, text, text_len);
}

// MARK: - Chat -

static bool llm_chat_check_context (ai_context *ai) {
    // check context
    if (!ai->ctx) {
        const char *options = "context_size=4096,n_gpu_layers=99";
        if (parse_keyvalue_string(options, llm_options_callback, &ai->options) == false) {
            sqlite_common_set_error(ai->context, ai->vtab, SQLITE_ERROR, "An error occurred while parsing options (%s)", options);
            return false;
        }
        ai->ctx = ai_context_check(ai);
        if (!ai->ctx) return false;
    }
    
    // check sampler
    if (!ai->sampler) {
        llm_sampler_check(ai);
        if (ai->sampler == NULL) return false;
        llama_sampler_chain_add(ai->sampler, llama_sampler_init_min_p(0.05, 1));
        llama_sampler_chain_add(ai->sampler, llama_sampler_init_temp(0.8));
        llama_sampler_chain_add(ai->sampler, llama_sampler_init_dist((uint32_t)LLAMA_DEFAULT_SEED));
    }
    
    // create history structs
    ai_uuid_v7_string(ai->chat.uuid, true);
    
    int n_ctx = llama_n_ctx(ai->ctx);
    buffer_create(&ai->chat.formatted, n_ctx);
    buffer_create(&ai->chat.response, MIN_ALLOC_RESPONSE);
    
    // do not report an error in case of malloc failure (it will be reported later by the function or the vtab)
    ai->chat.prompt = (char *)sqlite3_malloc(MIN_ALLOC_PROMPT);
    ai->chat.tokens = (llama_token *)sqlite3_malloc(sizeof(llama_token) * MIN_ALLOC_TOKEN);
    if (ai->chat.tokens) ai->chat.ntokens = MIN_ALLOC_TOKEN;
    
    return true;
}

static bool llm_chat_save_response (ai_context *ai, ai_messages *messages, const char *template) {
    char *response = ai->chat.response.data;
    if (!response) return false;
    
    if (!llm_messages_append(messages, ROLE_ASSISTANT, response, false, false)) {
        sqlite_common_set_error (ai->context, ai->vtab, SQLITE_ERROR, "Failed to append response");
        return false;
    }
    
    ai->chat.prev_len = llama_chat_apply_template(template, messages->items, messages->count, false, NULL, 0);
    if (ai->chat.prev_len < 0) {
        sqlite_common_set_error (ai->context, ai->vtab, SQLITE_ERROR, "Failed to finalize chat template");
        return false;
    }
    
    return true;
}

static bool llm_chat_generate_response (ai_context *ai, ai_cursor *c, bool *is_eog) {
    struct llama_context *ctx = ai->ctx;
    struct llama_sampler *sampler = ai->sampler;
    const struct llama_vocab *vocab = ai->chat.vocab;
    llama_batch batch = ai->chat.batch;
    char *tok = ai->chat.token_text;
    
    // check context space
    uint32_t n_ctx = llama_n_ctx(ctx);
    int32_t n_ctx_used = llama_memory_seq_pos_max(llama_get_memory(ctx), 0);
    if (n_ctx_used + batch.n_tokens > n_ctx) {
        sqlite_common_set_error (ai->context, ai->vtab, SQLITE_ERROR, "Context size exceeded (%d, %d)", n_ctx, n_ctx_used + batch.n_tokens);
        return false;
    }
    
    if (llama_decode(ctx, batch)) {
        sqlite_common_set_error (ai->context, ai->vtab, SQLITE_ERROR, "Failed to decode prompt batch");
        return false;
    }
    
    // sample next token
    ai->chat.token_id = llama_sampler_sample(sampler, ctx, -1);
    
    // DEBUG
    // printf("%d ", ai->chat.token_id);
    
    if (llama_vocab_is_eog(vocab, ai->chat.token_id)) {
        if (c) c->is_eog = true;
        if (is_eog) *is_eog = true;
        return true;
    }
    
    // convert token to string
    int32_t n = llama_token_to_piece(vocab, ai->chat.token_id, tok, MAX_TOKEN_TEXT_LEN, 0, true);
    if (n < 0) {
        sqlite_common_set_error (ai->context, ai->vtab, SQLITE_ERROR, "Failed to convert token to string");
        return false;
    }
    ai->chat.token_len = n;
    
    // append converted token to response buffer
    if (buffer_append(&ai->chat.response, tok, n, true) == false) {
        sqlite_common_set_error (ai->context, ai->vtab, SQLITE_ERROR, "Failed to grow response buffer");
        return false;
    }
    
    // DEBUG
    // fwrite(buf, 1, n, stdout);
    // fflush(stdout);
    
    // prepare next batch
    ai->chat.batch = llama_batch_get_one(&ai->chat.token_id, 1);
    ai->chat.token_count++;
    
    return true;
}

static bool llm_chat_tokenize_input (ai_context *ai, const char *prompt) {
    struct llama_context *ctx = ai->ctx;
    const struct llama_vocab *vocab = ai->chat.vocab;
    
    // check if first execution
    bool is_first = (llama_memory_seq_pos_max(llama_get_memory(ctx), 0) == -1);
    
    // count how many tokens prompt generates
    int32_t prompt_len = (int32_t)strlen(prompt);
    int32_t n_prompt_tokens = -llama_tokenize(vocab, prompt, prompt_len, NULL, 0, is_first, true);
    if (n_prompt_tokens <= 0) {
        sqlite_common_set_error (ai->context, ai->vtab, SQLITE_ERROR, "Failed to determine prompt token count");
        return false;
    }
    
    // check if a new token allocation is required
    if (ai->chat.ntokens < n_prompt_tokens) {
        llama_token *prompt_tokens = sqlite3_malloc(sizeof(llama_token) * n_prompt_tokens);
        if (!prompt_tokens) {
            sqlite_common_set_error (ai->context, ai->vtab, SQLITE_NOMEM, "Failed to allocate prompt token buffer");
            return false;
        }
        if (ai->chat.tokens) sqlite3_free(ai->chat.tokens);
        ai->chat.tokens = prompt_tokens;
        ai->chat.ntokens = n_prompt_tokens;
    }
    
    // tokenize prompt
    llama_token *prompt_tokens = ai->chat.tokens;
    if (llama_tokenize(vocab, prompt, prompt_len, prompt_tokens, n_prompt_tokens, is_first, true) < 0) {
        sqlite_common_set_error (ai->context, ai->vtab, SQLITE_ERROR, "Failed to tokenize the prompt");
        return false;
    }
    
    // create initial batch
    ai->chat.batch = llama_batch_get_one(prompt_tokens, n_prompt_tokens);
    
    return true;
}

static bool llm_chat_run (ai_context *ai, ai_cursor *c, const char *user_prompt) {
    // TODO: what to do if template is not available?
    const char *template = llama_model_chat_template(ai->model, NULL);
    if (!template) {
        sqlite_common_set_error (ai->context, ai->vtab, SQLITE_ERROR, "Template not available");
        return false;
    }
    
    const struct llama_vocab *vocab = llama_model_get_vocab(ai->model);
    if (!vocab) {
        sqlite_common_set_error (ai->context, ai->vtab, SQLITE_ERROR, "Model vocab not available");
        return false;
    }
    
    // setup context
    ai->chat.vocab = vocab;
    ai->chat.template = template;
    ai_messages *messages = &ai->chat.messages;
    buffer_t *formatted = &ai->chat.formatted;
    
    // save prompt input in history
    if (!llm_messages_append(messages, ROLE_USER, user_prompt, false, true)) {
        sqlite_common_set_error (ai->context, ai->vtab, SQLITE_ERROR, "Failed to append message");
        return false;
    }
    
    // transform a list of messages (the context) into
    // <|user|>What is AI?<|end|><|assistant|>AI stands for Artificial Intelligence...<|end|><|user|>Can you give an example?<|end|><|assistant|>...
    int32_t new_len = llama_chat_apply_template(template, messages->items, messages->count, true, formatted->data, formatted->capacity);
    if (new_len > formatted->capacity) {
        if (buffer_resize(formatted, new_len * 2) == false) return false;
        new_len = llama_chat_apply_template(template, messages->items, messages->count, true, formatted->data, formatted->capacity);
    }
    if ((new_len < 0) || (new_len > formatted->capacity)) {
        sqlite_common_set_error (ai->context, ai->vtab, SQLITE_ERROR, "failed to apply chat template");
        return false;
    }
    
    // check if there is enough space for the new formatted prompt
    int32_t prompt_len = new_len - ai->chat.prev_len;
    int32_t current_len = (int32_t)sqlite3_msize(ai->chat.prompt); // safe even if ai->chat.prompt is NULL
    if (current_len < prompt_len + 1) {
        char *buffer = (char *)sqlite3_malloc64(prompt_len + MIN_ALLOC_PROMPT);
        if (!buffer) {
            sqlite_common_set_error (ai->context, ai->vtab, SQLITE_NOMEM, "Failed to allocate prompt buffer");
            return false;
        }
        if (ai->chat.prompt) sqlite3_free(ai->chat.prompt);
        ai->chat.prompt = buffer;
    }
    
    // build templated version of the user prompt
    memcpy(ai->chat.prompt, formatted->data + ai->chat.prev_len, prompt_len);
    ai->chat.prompt[prompt_len] = 0;
    
    // tokenize input prompt
    if (!llm_chat_tokenize_input(ai, ai->chat.prompt)) return false;
    
    // if c is not NULL it means that reply must be streamed
    if (c) return true;
    
    // do not stream response and incrementally build the buffer
    bool is_eog = false;
    while (1) {
        if (!llm_chat_generate_response (ai, NULL, &is_eog)) return false;
        if (is_eog) break;
    }
    
    // save response
    if (llm_chat_save_response(ai, messages, template) == false) return false;
    
    // return full respond
    char *response = ai->chat.response.data;
    sqlite3_result_text(ai->context, response, -1, SQLITE_TRANSIENT);
    return true;
}

// MARK: -

static int llm_chat_connect (sqlite3 *db, void *pAux, int argc, const char *const *argv, sqlite3_vtab **ppVtab, char **pzErr) {
    int rc = sqlite3_declare_vtab(db, "CREATE TABLE x(reply, dummy hidden);");
    if (rc != SQLITE_OK) return rc;
    
    ai_vtab *vtab = (ai_vtab *)sqlite3_malloc(sizeof(ai_vtab));
    if (!vtab) return SQLITE_NOMEM;
    
    memset(vtab, 0, sizeof(ai_vtab));
    ai_context *ai = (ai_context *)pAux;
    
    vtab->ai = ai;
    ai->db = db;
    ai->context = NULL;
    ai->vtab = (sqlite3_vtab *)vtab;
    
    *ppVtab = (sqlite3_vtab *)vtab;
    return SQLITE_OK;
}

static int llm_chat_disconnect (sqlite3_vtab *pVtab) {
    ai_vtab *vtab = (ai_vtab *)pVtab;
    sqlite3_free(vtab);
    return SQLITE_OK;
}

static int llm_chat_best_index (sqlite3_vtab *tab, sqlite3_index_info *pIdxInfo) {
    pIdxInfo->idxNum = 1;
    pIdxInfo->orderByConsumed = 1;
    pIdxInfo->estimatedCost = (double)1;
    
    for (int i = 0; i < pIdxInfo->nConstraint; i++) {
        if (pIdxInfo->aConstraint[i].usable && pIdxInfo->aConstraint[i].op == SQLITE_INDEX_CONSTRAINT_EQ) {
            // tell SQLite we'll use this argument
            pIdxInfo->aConstraintUsage[i].argvIndex = 1;
            pIdxInfo->aConstraintUsage[i].omit = 1;
            pIdxInfo->idxNum = 1;
            break;
        }
    }
    
    return SQLITE_OK;
}

static int llm_chat_cursor_open (sqlite3_vtab *pVtab, sqlite3_vtab_cursor **ppCursor) {
    ai_cursor *c = (ai_cursor *)sqlite3_malloc(sizeof(ai_cursor));
    if (!c) return SQLITE_NOMEM;
    
    memset(c, 0, sizeof(ai_cursor));
    ai_vtab *vtab = (ai_vtab *)pVtab;
    c->vtab = vtab;
    c->ai = vtab->ai;
    
    ai_context *ai = c->ai;
    if (llm_chat_check_context(ai) == false) return SQLITE_ERROR;
    
    *ppCursor = (sqlite3_vtab_cursor *)c;
    return SQLITE_OK;
}

static int llm_chat_cursor_close (sqlite3_vtab_cursor *cur) {
    ai_cursor *c = (ai_cursor *)cur;
    ai_context *ai = c->ai;
    
    // save response when cursor closes
    sqlite3_free(c);
    
    ai_messages *messages = &ai->chat.messages;
    const char *template = ai->chat.template;
    if (llm_chat_save_response(ai, messages, template) == false) return SQLITE_ERROR;
    
    return SQLITE_OK;
}

static int llm_chat_cursor_next (sqlite3_vtab_cursor *cur) {
    ai_cursor *c = (ai_cursor *)cur;
    if (!llm_chat_generate_response (c->ai, c, NULL)) return SQLITE_ERROR;
    c->rowid++;
    return SQLITE_OK;
}

static int llm_chat_cursor_eof (sqlite3_vtab_cursor *cur) {
    ai_cursor *c = (ai_cursor *)cur;
    return (int)c->is_eog;
}

static int llm_chat_cursor_column (sqlite3_vtab_cursor *cur, sqlite3_context *context, int iCol) {
    ai_cursor *c = (ai_cursor *)cur;
    if (iCol == AI_COLUMN_REPLY) {
        sqlite3_result_text(context, c->ai->chat.token_text, c->ai->chat.token_len, SQLITE_TRANSIENT);
    }
    return SQLITE_OK;
}

static int llm_chat_cursor_rowid (sqlite3_vtab_cursor *cur, sqlite_int64 *pRowid) {
    ai_cursor *c = (ai_cursor *)cur;
    *pRowid = c->rowid;
    return SQLITE_OK;
}

static int llm_chat_cursor_filter (sqlite3_vtab_cursor *cur, int idxNum, const char *idxStr, int argc, sqlite3_value **argv) {
    ai_cursor *c = (ai_cursor *)cur;
    ai_context *ai = c->ai;
    ai_vtab *vtab = c->vtab;
    
    // sanity check arguments
    if (argc != 1) {
        return sqlite_vtab_set_error(&vtab->base, "llm_chat expects %d arguments, but %d were provided.", 1, argc);
    }
    if (sqlite3_value_type(argv[0]) != SQLITE_TEXT) {
        return sqlite_vtab_set_error(&vtab->base, "llm_chat argument must be of type TEXT");
    }
    
    ai->chat.token_count = 0;
    buffer_reset(&ai->chat.formatted);
    buffer_reset(&ai->chat.response);
    
    const char *user_prompt = (const char *)sqlite3_value_text(argv[0]);
    bool result = llm_chat_run(ai, c, user_prompt);
    return (result) ? SQLITE_OK : SQLITE_ERROR;
}

static sqlite3_module llm_chat = {
  /* iVersion    */ 0,
  /* xCreate     */ 0,
  /* xConnect    */ llm_chat_connect,
  /* xBestIndex  */ llm_chat_best_index,
  /* xDisconnect */ llm_chat_disconnect,
  /* xDestroy    */ 0,
  /* xOpen       */ llm_chat_cursor_open,
  /* xClose      */ llm_chat_cursor_close,
  /* xFilter     */ llm_chat_cursor_filter,
  /* xNext       */ llm_chat_cursor_next,
  /* xEof        */ llm_chat_cursor_eof,
  /* xColumn     */ llm_chat_cursor_column,
  /* xRowid      */ llm_chat_cursor_rowid,
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

static void llm_chat_free (sqlite3_context *context, int argc, sqlite3_value **argv) {
    ai_context *ai = (ai_context *)sqlite3_user_data(context);
    
    // reset UUID and cleanup chat related memory
    memset(ai->chat.uuid, 0, UUID_STR_MAXLEN);
    
    buffer_destroy(&ai->chat.response);
    buffer_destroy(&ai->chat.formatted);
    llm_messages_free(&ai->chat.messages);
    
    if (ai->chat.tokens) sqlite3_free(ai->chat.tokens);
    ai->chat.ntokens = 0;
    
    if (ai->chat.prompt) sqlite3_free(ai->chat.prompt);
    ai->chat.prev_len = 0;
}

static void llm_chat_create (sqlite3_context *context, int argc, sqlite3_value **argv) {
    ai_context *ai = (ai_context *)sqlite3_user_data(context);
    
    // clean-up old chat (if any)
    llm_chat_free(context, argc, argv);
    if (llm_chat_check_context(ai) == false) return;
    
    // returns chat UUID
    sqlite3_result_text(context, ai->chat.uuid, -1, SQLITE_TRANSIENT);
}

static bool llm_chat_check_tables (sqlite3_context *context) {
    const char *sql = "CREATE TABLE IF NOT EXISTS ai_chat_history (id INTEGER PRIMARY KEY AUTOINCREMENT, uuid TEXT UNIQUE, title TEXT, metadata TEXT, created_at DATETIME DEFAULT CURRENT_TIMESTAMP);";
    
    sqlite3 *db = sqlite3_context_db_handle(context);
    int rc = sqlite_db_write_simple(context, db, sql);
    if (rc != SQLITE_OK) return false;
    
    sql = "CREATE TABLE IF NOT EXISTS ai_chat_messages (id INTEGER PRIMARY KEY AUTOINCREMENT, chat_id INTEGER NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL);";
    rc = sqlite_db_write_simple(context, db, sql);
    if (rc != SQLITE_OK) return false;
    
    return true;
}

static void llm_chat_save (sqlite3_context *context, int argc, sqlite3_value **argv) {
    ai_context *ai = (ai_context *)sqlite3_user_data(context);
    if (llm_chat_check_tables(context) == false) return;
    
    // sanity check if there is something to save
    if (ai->chat.uuid[0] == 0) return;
    if (ai->chat.messages.count == 0) return;
    
    // title, metadata
    const char *title = ((argc >= 1) && (sqlite3_value_type(argv[0]) == SQLITE3_TEXT)) ? (const char *)sqlite3_value_text(argv[0]) : NULL;
    const char *meta =  ((argc >= 2) && (sqlite3_value_type(argv[1]) == SQLITE3_TEXT)) ? (const char *)sqlite3_value_text(argv[1]) : NULL;
    
    sqlite3 *db = sqlite3_context_db_handle(context);
    ai_messages *messages = &ai->chat.messages;
    
    // start transaction
    sqlite_db_write_simple(context, db, "BEGIN;");
    
    // save chat
    const char *sql = "INSERT INTO ai_chat_history (uuid, title, metadata) VALUES (?, ?, ?);";
    const char *values[] = {ai->chat.uuid, title, meta};
    int types[] = {SQLITE_TEXT, SQLITE_TEXT, SQLITE_TEXT};
    int lens[] = {-1, -1, -1};
    
    int rc = sqlite_db_write(context, db, sql, values, types, lens, 3);
    if (rc != SQLITE_OK) goto abort_save;
        
    // loop to save messages (the context)
    char rowid_s[256];
    sqlite3_int64 rowid = sqlite3_last_insert_rowid(db);
    snprintf(rowid_s, sizeof(rowid_s), "%lld", (long long)rowid);
    
    sql = "INSERT INTO ai_chat_messages (chat_id, role, content) VALUES (?, ?, ?);";
    int types2[] = {SQLITE_INTEGER, SQLITE_TEXT, SQLITE_TEXT};
    
    for (int i=0; i < messages->count; i++) {
        const char *role = messages->items[i].role;
        const char *content = messages->items[i].content;
        const char *values2[] = {rowid_s, role, content};
        int rc = sqlite_db_write(context, db, sql, values2, types2, lens, 3);
        if (rc != SQLITE_OK) goto abort_save;
    }
    
    // commit transaction and returns chat UUID
    sqlite_db_write_simple(context, db, "COMMIT;");
    sqlite3_result_text(context, ai->chat.uuid, -1, SQLITE_TRANSIENT);
    return;
    
abort_save:
    sqlite_db_write_simple(context, db, "ROLLBACK;");
}

static void llm_chat_restore (sqlite3_context *context, int argc, sqlite3_value **argv) {
    // sanity check argument
    int types[] = {SQLITE_TEXT};
    if (sqlite_sanity_function(context, "llm_chat_restore", argc, argv, 1, types, false) == false) return;
    
    // free old chat (if any)
    llm_chat_free(context, 0, NULL);
    
    // UUID
    const char *uuid = (const char *)sqlite3_value_text(argv[0]);
    ai_context *ai = (ai_context *)sqlite3_user_data(context);
    sqlite3 *db = sqlite3_context_db_handle(context);
    
    const char *sql = "SELECT m.role, m.content FROM ai_chat_messages m JOIN ai_chat_history h ON m.chat_id = h.id WHERE h.uuid = ? ORDER BY m.id ASC;";
    sqlite3_stmt *vm = NULL;
    int rc = sqlite3_prepare_v2(db, sql, -1, &vm, NULL);
    if (rc != SQLITE_OK) goto abort_restore;
    
    rc = sqlite3_bind_text(vm, 1, uuid, -1, SQLITE_STATIC);
    if (rc != SQLITE_OK) goto abort_restore;
    
    int counter = 0;
    ai_messages *messages = &ai->chat.messages;
    while (1) {
        rc = sqlite3_step(vm);
        if (rc == SQLITE_DONE) {rc = SQLITE_OK; break;}
        if (rc != SQLITE_ROW) goto abort_restore;
        
        const char *role = (const char *)sqlite3_column_text(vm, 0);
        const char *content = (const char *)sqlite3_column_text(vm, 1);
        
        if (!llm_messages_append(messages, role, content, true, true)) {
            sqlite_common_set_error (ai->context, ai->vtab, SQLITE_ERROR, "Failed to append response");
            rc = SQLITE_OK;
            goto abort_restore;
        }
        ++counter;
    }
    
    sqlite3_result_int(context, counter);
    if (vm) sqlite3_finalize(vm);
    return;
    
abort_restore:
    if (rc != SQLITE_OK) sqlite3_result_error(context, sqlite3_errmsg(db), rc);
    if (vm) sqlite3_finalize(vm);
}

static void llm_chat_respond (sqlite3_context *context, int argc, sqlite3_value **argv) {
    int types[] = {SQLITE_TEXT};
    if (sqlite_sanity_function(context, "llm_chat_respond", argc, argv, 1, types, true) == false) return;
    
    ai_context *ai = (ai_context *)sqlite3_user_data(context);
    if (llm_chat_check_context(ai) == false) return;
    
    const char *user_prompt = (const char *)sqlite3_value_text(argv[0]);
    ai->context = context;
    ai->vtab = NULL;
    
    ai->chat.token_count = 0;
    buffer_reset(&ai->chat.formatted);
    buffer_reset(&ai->chat.response);
    llm_chat_run(ai, NULL, user_prompt);
}

// MARK: - LLM Sampler -

static void llm_sampler_init_greedy (sqlite3_context *context, int argc, sqlite3_value **argv) {
    ai_context *ai = (ai_context *)sqlite3_user_data(context);
    llm_sampler_check(ai);
    if (ai->sampler) llama_sampler_chain_add(ai->sampler, llama_sampler_init_greedy());
}

static void llm_sampler_init_dist (sqlite3_context *context, int argc, sqlite3_value **argv) {
    if (argc != 0) {
        int types[] = {SQLITE_INTEGER};
        if (sqlite_sanity_function(context, "llm_sampler_init_dist", argc, argv, 1, types, true) == false) return;
    }
    
    ai_context *ai = (ai_context *)sqlite3_user_data(context);
    llm_sampler_check(ai);
    if (ai->sampler) {
        int32_t seed = (argc == 1) ? (int32_t)sqlite3_value_int64(argv[0]) : (int32_t)LLAMA_DEFAULT_SEED;
        llama_sampler_chain_add(ai->sampler, llama_sampler_init_dist(seed));
    }
}

static void llm_sampler_init_top_k (sqlite3_context *context, int argc, sqlite3_value **argv) {
    // Top-K sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
    
    int types[] = {SQLITE_INTEGER};
    if (sqlite_sanity_function(context, "llm_sampler_init_top_k", argc, argv, 1, types, true) == false) return;
    
    ai_context *ai = (ai_context *)sqlite3_user_data(context);
    llm_sampler_check(ai);
    if (ai->sampler) {
        int32_t k = (int32_t)sqlite3_value_int64(argv[0]);
        llama_sampler_chain_add(ai->sampler, llama_sampler_init_top_k(k));
    }
}

static void llm_sampler_init_top_p (sqlite3_context *context, int argc, sqlite3_value **argv) {
    // Nucleus sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
    
    int types[] = {SQLITE_FLOAT, SQLITE_INTEGER};
    if (sqlite_sanity_function(context, "llm_sampler_init_top_p", argc, argv, 2, types, true) == false) return;
    
    ai_context *ai = (ai_context *)sqlite3_user_data(context);
    llm_sampler_check(ai);
    if (ai->sampler) {
        float p = (float)sqlite3_value_double(argv[0]);
        size_t min_keep = (size_t)sqlite3_value_int64(argv[1]);
        llama_sampler_chain_add(ai->sampler, llama_sampler_init_top_p(p, min_keep));
    }
}

static void llm_sampler_init_min_p (sqlite3_context *context, int argc, sqlite3_value **argv) {
    // Minimum P sampling as described in https://github.com/ggml-org/llama.cpp/pull/3841
    
    int types[] = {SQLITE_FLOAT, SQLITE_INTEGER};
    if (sqlite_sanity_function(context, "llm_sampler_init_min_p", argc, argv, 2, types, true) == false) return;
    
    ai_context *ai = (ai_context *)sqlite3_user_data(context);
    llm_sampler_check(ai);
    if (ai->sampler) {
        float p = (float)sqlite3_value_double(argv[0]);
        size_t min_keep = (size_t)sqlite3_value_int64(argv[1]);
        llama_sampler_chain_add(ai->sampler, llama_sampler_init_min_p(p, min_keep));
    }
}

static void llm_sampler_init_typical (sqlite3_context *context, int argc, sqlite3_value **argv) {
    // Locally Typical Sampling implementation described in the paper https://arxiv.org/abs/2202.00666
    
    int types[] = {SQLITE_FLOAT, SQLITE_INTEGER};
    if (sqlite_sanity_function(context, "llm_sampler_init_typical", argc, argv, 2, types, true) == false) return;
    
    ai_context *ai = (ai_context *)sqlite3_user_data(context);
    llm_sampler_check(ai);
    if (ai->sampler) {
        float p = (float)sqlite3_value_double(argv[0]);
        size_t min_keep = (size_t)sqlite3_value_int64(argv[1]);
        llama_sampler_chain_add(ai->sampler, llama_sampler_init_typical(p, min_keep));
    }
}

static void llm_sampler_init_temp (sqlite3_context *context, int argc, sqlite3_value **argv) {
    int types[] = {SQLITE_FLOAT};
    if (sqlite_sanity_function(context, "llm_sampler_init_temp", argc, argv, 1, types, true) == false) return;
    
    ai_context *ai = (ai_context *)sqlite3_user_data(context);
    llm_sampler_check(ai);
    if (ai->sampler) {
        float t = (float)sqlite3_value_double(argv[0]);
        llama_sampler_chain_add(ai->sampler, llama_sampler_init_temp(t));
    }
}

static void llm_sampler_init_temp_ext (sqlite3_context *context, int argc, sqlite3_value **argv) {
    // Dynamic temperature implementation (a.k.a. entropy) described in the paper https://arxiv.org/abs/2309.02772
    
    int types[] = {SQLITE_FLOAT, SQLITE_FLOAT, SQLITE_FLOAT};
    if (sqlite_sanity_function(context, "llm_sampler_init_temp_ext", argc, argv, 3, types, true) == false) return;
    
    ai_context *ai = (ai_context *)sqlite3_user_data(context);
    llm_sampler_check(ai);
    if (ai->sampler) {
        float t = (float)sqlite3_value_double(argv[0]);
        float delta = (float)sqlite3_value_double(argv[1]);
        float exponent = (float)sqlite3_value_double(argv[2]);
        llama_sampler_chain_add(ai->sampler, llama_sampler_init_temp_ext(t, delta, exponent));
    }
}

static void llm_sampler_init_xtc (sqlite3_context *context, int argc, sqlite3_value **argv) {
    // XTC sampler as described in https://github.com/oobabooga/text-generation-webui/pull/6335
    
    int types[] = {SQLITE_FLOAT, SQLITE_FLOAT, SQLITE_INTEGER, SQLITE_INTEGER};
    if (sqlite_sanity_function(context, "llm_sampler_init_xtc", argc, argv, 4, types, true) == false) return;
    
    ai_context *ai = (ai_context *)sqlite3_user_data(context);
    llm_sampler_check(ai);
    if (ai->sampler) {
        float p = (float)sqlite3_value_double(argv[0]);
        float t = (float)sqlite3_value_double(argv[1]);
        size_t min_keep = (size_t)sqlite3_value_int64(argv[2]);
        uint32_t seed = (uint32_t)sqlite3_value_int64(argv[3]);
        llama_sampler_chain_add(ai->sampler, llama_sampler_init_xtc(p, t, min_keep, seed));
    }
}

static void llm_sampler_init_top_n_sigma (sqlite3_context *context, int argc, sqlite3_value **argv) {
    // Top n sigma sampling as described in academic paper "Top-nσ: Not All Logits Are You Need" https://arxiv.org/pdf/2411.07641
    
    int types[] = {SQLITE_FLOAT};
    if (sqlite_sanity_function(context, "llm_sampler_init_top_n_sigma", argc, argv, 1, types, true) == false) return;
    
    ai_context *ai = (ai_context *)sqlite3_user_data(context);
    llm_sampler_check(ai);
    if (ai->sampler) {
        float n = (float)sqlite3_value_double(argv[0]);
        llama_sampler_chain_add(ai->sampler, llama_sampler_init_top_n_sigma(n));
    }
}

static void llm_sampler_init_mirostat (sqlite3_context *context, int argc, sqlite3_value **argv) {
    // Mirostat 1.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
    
    int types[] = {SQLITE_INTEGER, SQLITE_FLOAT, SQLITE_FLOAT, SQLITE_INTEGER};
    if (sqlite_sanity_function(context, "llm_sampler_init_mirostat", argc, argv, 4, types, true) == false) return;
    
    ai_context *ai = (ai_context *)sqlite3_user_data(context);
    const struct llama_vocab *vocab = llama_model_get_vocab(ai->model);
    if (!vocab) {
        sqlite_context_result_error(context, SQLITE_ERROR, "Unable to get vocab from current model.");
        return;
    }
    
    llm_sampler_check(ai);
    if (ai->sampler) {
        uint32_t seed = (uint32_t)sqlite3_value_int64(argv[0]);
        float tau = (float)sqlite3_value_double(argv[1]);
        float eta = (float)sqlite3_value_double(argv[2]);
        int32_t m = (int32_t)sqlite3_value_int64(argv[3]);
        llama_sampler_chain_add(ai->sampler, llama_sampler_init_mirostat(llama_vocab_n_tokens(vocab), seed, tau, eta, m));
    }
}

static void llm_sampler_init_mirostat_v2 (sqlite3_context *context, int argc, sqlite3_value **argv) {
    // Mirostat 2.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
    
    int types[] = {SQLITE_INTEGER, SQLITE_FLOAT, SQLITE_FLOAT};
    if (sqlite_sanity_function(context, "llm_sampler_init_mirostat_v2", argc, argv, 3, types, true) == false) return;
    
    ai_context *ai = (ai_context *)sqlite3_user_data(context);
    llm_sampler_check(ai);
    if (ai->sampler) {
        uint32_t seed = (uint32_t)sqlite3_value_int64(argv[0]);
        float tau = (float)sqlite3_value_double(argv[1]);
        float eta = (float)sqlite3_value_double(argv[2]);
        llama_sampler_chain_add(ai->sampler, llama_sampler_init_mirostat_v2(seed, tau, eta));
    }
}

static void llm_sampler_init_grammar (sqlite3_context *context, int argc, sqlite3_value **argv) {
    int types[] = {SQLITE_TEXT, SQLITE_TEXT};
    if (sqlite_sanity_function(context, "llm_sampler_init_grammar", argc, argv, 2, types, true) == false) return;
    
    ai_context *ai = (ai_context *)sqlite3_user_data(context);
    const struct llama_vocab *vocab = llama_model_get_vocab(ai->model);
    if (!vocab) {
        sqlite_context_result_error(context, SQLITE_ERROR, "Unable to get vocab from current model.");
        return;
    }
    
    llm_sampler_check(ai);
    if (ai->sampler) {
        const char *grammar_str = (const char *)sqlite3_value_text(argv[0]);
        const char *grammar_root = (const char *)sqlite3_value_text(argv[1]);
        llama_sampler_chain_add(ai->sampler, llama_sampler_init_grammar(vocab, grammar_str, grammar_root));
    }
}

static void llm_sampler_init_infill (sqlite3_context *context, int argc, sqlite3_value **argv) {
    ai_context *ai = (ai_context *)sqlite3_user_data(context);
    const struct llama_vocab *vocab = llama_model_get_vocab(ai->model);
    if (!vocab) {
        sqlite_context_result_error(context, SQLITE_ERROR, "Unable to get vocab from current model.");
        return;
    }
    
    llm_sampler_check(ai);
    if (ai->sampler) {
        llama_sampler_chain_add(ai->sampler, llama_sampler_init_infill(vocab));
    }
}

static void llm_sampler_init_penalties (sqlite3_context *context, int argc, sqlite3_value **argv) {
    int types[] = {SQLITE_INTEGER, SQLITE_FLOAT, SQLITE_FLOAT, SQLITE_FLOAT};
    if (sqlite_sanity_function(context, "llm_sampler_init_penalties", argc, argv, 4, types, true) == false) return;
    
    ai_context *ai = (ai_context *)sqlite3_user_data(context);
    llm_sampler_check(ai);
    if (ai->sampler) {
        int32_t penalty_last_n = (int32_t)sqlite3_value_int64(argv[0]);
        float penalty_repeat = (float)sqlite3_value_double(argv[1]);
        float penalty_freq = (float)sqlite3_value_double(argv[2]);
        float penalty_present = (float)sqlite3_value_double(argv[3]);
        llama_sampler_chain_add(ai->sampler, llama_sampler_init_penalties(penalty_last_n, penalty_repeat, penalty_freq, penalty_present));
    }
}

// MARK: - LLM General -

static void llm_sampler_free (sqlite3_context *context, int argc, sqlite3_value **argv) {
    ai_context *ai = (ai_context *)sqlite3_user_data(context);
    if (ai->sampler) llama_sampler_free(ai->sampler);
    ai->sampler = NULL;
}

static void llm_sampler_create (sqlite3_context *context, int argc, sqlite3_value **argv) {
    ai_context *ai = (ai_context *)sqlite3_user_data(context);
    if (ai->sampler) llama_sampler_free(ai->sampler);
    ai->sampler = NULL;
    llm_sampler_check(ai);
}

static void llm_context_free (sqlite3_context *context, int argc, sqlite3_value **argv) {
    ai_context *ai = (ai_context *)sqlite3_user_data(context);
    if (ai->ctx) llama_free(ai->ctx);
    ai->ctx = NULL;
}

static void llm_context_create (sqlite3_context *context, int argc, sqlite3_value **argv) {
    // sanity check arguments
    if ((argc > 0) && ai_common_args_check(context, "llm_context_create", argc, argv, true) == false) return;
    const char *options = (argc == 1) ? (const char *)sqlite3_value_text(argv[0]) : NULL;
    
    ai_context *ai = (ai_context *)sqlite3_user_data(context);
    if (parse_keyvalue_string(options, llm_options_callback, &ai->options) == false) {
        sqlite_context_result_error(context, SQLITE_ERROR, "An error occurred while parsing options (%s)", options);
        return;
    }
    
    if (ai->ctx) llm_context_free(context, 0, NULL);
    ai->ctx = ai_context_check(ai);
}

static void llm_model_free (sqlite3_context *context, int argc, sqlite3_value **argv) {
    ai_context *ai = (ai_context *)sqlite3_user_data(context);
    if (ai->model) llama_model_free(ai->model);
    if (ai->ctx) llama_free(ai->ctx);
    ai->model = NULL;
    ai->ctx = NULL;
}

static void llm_model_load (sqlite3_context *context, int argc, sqlite3_value **argv) {
    // sanity check arguments
    if (ai_common_args_check(context, "llm_model_load", argc, argv, false) == false) return;
    
    const char *model_path = (const char *)sqlite3_value_text(argv[0]);
    const char *model_options = (argc == 2) ? (const char *)sqlite3_value_text(argv[1]) : NULL;
    
    ai_context *ai = (ai_context *)sqlite3_user_data(context);
    if (parse_keyvalue_string(model_options, llm_options_callback, &ai->options) == false) {
        sqlite_context_result_error(context, SQLITE_ERROR, "An error occurred while parsing options (%s)", model_options);
        return;
    }
    
    struct llama_model_params model_params = llama_model_default_params();
    llm_set_model_options(&model_params, &ai->options);
    struct llama_model *model = llama_model_load_from_file(model_path, model_params);
    if (!model) {
        sqlite_context_result_error(context, SQLITE_ERROR, "Unable to load model from file %s", model_path);
        return;
    }
    
    ai_cleanup((void *)ai);
    ai->model = model;
}

// MARK: -

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
    void *ctx = ai_create(db);
    if (!ctx) {
        if (pzErrMsg) *pzErrMsg = sqlite3_mprintf("Out of memory: failed to allocate AI extension context.");
        return SQLITE_NOMEM;
    }
    
    // set logger
    llama_log_set(ai_logger, ctx);
    
    // register public functions
    int rc = SQLITE_OK;
    rc = sqlite3_create_function_v2(db, "ai_version", 0, SQLITE_UTF8, ctx, ai_version, NULL, NULL, ai_cleanup);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function_v2(db, "ai_log_info", 1, SQLITE_UTF8, ctx, ai_log_info, NULL, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function(db, "llm_model_load", 1, SQLITE_UTF8, ctx, llm_model_load, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function(db, "llm_model_load", 2, SQLITE_UTF8, ctx, llm_model_load, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function(db, "llm_model_free", 0, SQLITE_UTF8, ctx, llm_model_free, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function(db, "llm_context_create", 0, SQLITE_UTF8, ctx, llm_context_create, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function(db, "llm_context_create", 1, SQLITE_UTF8, ctx, llm_context_create, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function(db, "llm_context_free", 0, SQLITE_UTF8, ctx, llm_context_free, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function(db, "llm_sampler_create", 0, SQLITE_UTF8, ctx, llm_sampler_create, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function(db, "llm_sampler_free", 0, SQLITE_UTF8, ctx, llm_sampler_free, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function(db, "llm_sampler_init_greedy", 0, SQLITE_UTF8, ctx, llm_sampler_init_greedy, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function(db, "llm_sampler_init_dist", 0, SQLITE_UTF8, ctx, llm_sampler_init_dist, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function(db, "llm_sampler_init_dist", 1, SQLITE_UTF8, ctx, llm_sampler_init_dist, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function(db, "llm_sampler_init_top_k", 1, SQLITE_UTF8, ctx, llm_sampler_init_top_k, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function(db, "llm_sampler_init_top_p", 2, SQLITE_UTF8, ctx, llm_sampler_init_top_p, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function(db, "llm_sampler_init_min_p", 2, SQLITE_UTF8, ctx, llm_sampler_init_min_p, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function(db, "llm_sampler_init_typical", 2, SQLITE_UTF8, ctx, llm_sampler_init_typical, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function(db, "llm_sampler_init_temp", 1, SQLITE_UTF8, ctx, llm_sampler_init_temp, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function(db, "llm_sampler_init_temp_ext", 3, SQLITE_UTF8, ctx, llm_sampler_init_temp_ext, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function(db, "llm_sampler_init_xtc", 4, SQLITE_UTF8, ctx, llm_sampler_init_xtc, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function(db, "llm_sampler_init_top_n_sigma", 1, SQLITE_UTF8, ctx, llm_sampler_init_top_n_sigma, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function(db, "llm_sampler_init_mirostat", 4, SQLITE_UTF8, ctx, llm_sampler_init_mirostat, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function(db, "llm_sampler_init_mirostat_v2", 3, SQLITE_UTF8, ctx, llm_sampler_init_mirostat_v2, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function(db, "llm_sampler_init_grammar", 2, SQLITE_UTF8, ctx, llm_sampler_init_grammar, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function(db, "llm_sampler_init_infill", 0, SQLITE_UTF8, ctx, llm_sampler_init_infill, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function(db, "llm_sampler_init_penalties", 4, SQLITE_UTF8, ctx, llm_sampler_init_penalties, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function(db, "llm_embed_generate", 1, SQLITE_UTF8, ctx, llm_embed_generate, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function(db, "llm_embed_generate", 2, SQLITE_UTF8, ctx, llm_embed_generate, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function(db, "llm_text_generate", 1, SQLITE_UTF8, ctx, llm_text_generate, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function(db, "llm_text_generate", 2, SQLITE_UTF8, ctx, llm_text_generate, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_module(db, "llm_chat", &llm_chat, ctx);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function(db, "llm_chat_create", 0, SQLITE_UTF8, ctx, llm_chat_create, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function(db, "llm_chat_free", 0, SQLITE_UTF8, ctx, llm_chat_free, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function(db, "llm_chat_save", 0, SQLITE_UTF8, ctx, llm_chat_save, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function(db, "llm_chat_save", 1, SQLITE_UTF8, ctx, llm_chat_save, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function(db, "llm_chat_save", 2, SQLITE_UTF8, ctx, llm_chat_save, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function(db, "llm_chat_restore", 1, SQLITE_UTF8, ctx, llm_chat_restore, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function(db, "llm_chat_respond", 1, SQLITE_UTF8, ctx, llm_chat_respond, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
cleanup:
    return rc;
}
