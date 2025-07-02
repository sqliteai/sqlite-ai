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

#define MIN_ALLOC_TOKEN                         512
#define MAX_PATH                                4096

#define OPTION_KEY_GENERATE_EMBEDDING           "generate_embedding"
#define OPTION_KEY_NORMALIZE_EMBEDDING          "normalize_embedding"
#define OPTION_KEY_MAX_TOKENS                   "max_tokens"
#define OPTION_KEY_JSON_OUTPUT                  "json_output"
#define OPTION_KEY_GPU_LAYERS                   "gpu_layers"
#define OPTION_KEY_CONTEXT_SIZE                 "context_size"

typedef struct {
    int32_t                 max_tokens;
    int                     gpu_layers;
    int                     context_size;
    
    bool                    generate_embedding;
    bool                    normalize_embedding;
    bool                    json_output;
} ai_options;

typedef struct {
    sqlite3                 *db;
    struct llama_model      *model;
    struct llama_context    *ctx;
    
    llama_token             *token_buffer;
    int32_t                 token_buffer_size;
    ai_options              options;
} ai_context;

typedef struct {
    sqlite3_vtab    base;                   // Base class - must be first
    sqlite3         *db;
    ai_context      *ai;
} ai_vtab;

typedef struct {
    sqlite3_vtab_cursor base;               // Base class - must be first
} ai_cursor;

// MARK: -

static void ai_options_init (ai_options *options) {
    memset(options, 0, sizeof(ai_options));
    
    options->normalize_embedding = true;
    options->max_tokens = 0; // no limits
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
    
    // means ignore unknown keys
    return true;
}

void *ai_context_create (sqlite3 *db) {
    ai_context *ai = (ai_context *)sqlite3_malloc(sizeof(ai_context));
    if (ai) {
        ai_options_init(&ai->options);
        ai->db = db;
    }
    return ai;
}

void ai_context_free (void *ctx) {
    if (!ctx) return;
    
    ai_context *ai = (ai_context *)ctx;
    sqlite_clear_ptr(ai->db);
    if (ai->model) llama_model_free(ai->model);
    if (ai->ctx) llama_free(ai->ctx);
    if (ai->token_buffer) sqlite3_free(ai->token_buffer);
}

void ai_logger (enum ggml_log_level level, const char *text, void *user_data) {
    if (level == GGML_LOG_LEVEL_INFO) return;
    
    const char *prefix = NULL;
    switch (level) {
        case GGML_LOG_LEVEL_NONE: prefix = "NONE"; break;
        case GGML_LOG_LEVEL_DEBUG: prefix = "DEBUG"; break;
        case GGML_LOG_LEVEL_INFO: prefix = "INFO"; break;
        case GGML_LOG_LEVEL_WARN: prefix = "WARNING"; break;
        case GGML_LOG_LEVEL_ERROR: prefix = "ERROR"; break;
        case GGML_LOG_LEVEL_CONT: prefix = NULL; break;
    }
    
    (prefix) ? printf("\n[%s]\t%s", prefix, text) : printf("%s", text);
}

void ai_set_model_options (struct llama_model_params *model_params, ai_options *options) {
    if (options->gpu_layers) model_params->n_gpu_layers = options->gpu_layers;
}

void ai_set_llama_options (struct llama_context_params *llama_context, ai_options *options) {
    if (options->generate_embedding) llama_context->embeddings = true;
    if (options->context_size) {
        llama_context->n_ctx = options->context_size;
        llama_context->n_batch = options->context_size;
    }
}

bool ai_check_model (sqlite3_context *context) {
    ai_context *ai = (ai_context *)sqlite3_user_data(context);
    if (!ai) return false;
    return (ai->model != NULL);
}

// MARK: -

static const char *db_check_model (sqlite3 *db, const char *name, char *path, size_t path_len) {
    // TODO: load path from name inside the ai_models table
    return NULL;
}

// MARK: - Embedding Generation -

static void ai_embedding_normalize (const float *src, float *dest, int dim) {
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

static void ai_embed_generate (sqlite3_context *context, const char *text, int32_t text_len) {
    // step 0: sanity check context
    ai_context *ai = (ai_context *)sqlite3_user_data(context);
    struct llama_context *ctx = ai->ctx;
    llama_set_embeddings(ctx, true);
    
    if (llama_model_has_encoder(ai->model) && llama_model_has_decoder(ai->model)) {
        sqlite_context_result_error(context, SQLITE_ERROR, "Computing embeddings in encoder-decoder models is not supported");
        return;
    }
    
    // sanity check model type (decode is used to create embeddings)
    if (llama_model_has_decoder(ai->model) == false) {
        sqlite_context_result_error(context, SQLITE_ERROR, "Model does not support decoding (required for embedding)");
        return;
    }
    
    const int n_ctx_train = llama_model_n_ctx_train(ai->model);
    const int n_ctx = llama_n_ctx(ctx);
    
    if (n_ctx > n_ctx_train) {
        printf("--> warning: model was trained on only %d context tokens (%d specified)\n", n_ctx_train, n_ctx);
    }
    
    // sanity check vocab
    const struct llama_vocab *vocab = llama_model_get_vocab(ai->model);
    if (!vocab) {
        sqlite_context_result_error(context, SQLITE_ERROR, "Failed to extract vocabulary from the model");
        return;
    }
    
    if (llama_vocab_get_add_sep(vocab)) {
        const char *sep = llama_vocab_get_text(vocab, llama_vocab_sep(vocab));
        printf("sep: %s\n", sep);
    }
    
    if (llama_vocab_get_add_eos(vocab)) {
        const char *eos = llama_vocab_get_text(vocab, llama_vocab_eos(vocab));
        printf("eos: %s\n", eos);
    }
    
    // sanity check embedding memory
    int dimension = llama_model_n_embd(llama_get_model(ctx));
    float *embedding = (float *)sqlite3_malloc64(sizeof(float) * dimension);
    if (!embedding) {
        sqlite_context_result_error(context, SQLITE_NOMEM, "Out of memory: failed to allocate embedding buffer of dimension %d", dimension);
        return;
    }
    
    // step 1: get token count needed
    int32_t n_tokens = llama_tokenize(vocab, text, text_len, NULL, 0, true, false);
    if (n_tokens < 0) n_tokens *= -1; // if negative number => the number of tokens that would have been returned
    
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
    
    // step2: check if memory allocation is needed
    if (n_tokens > ai->token_buffer_size) {
        int32_t n_count = (n_tokens > MIN_ALLOC_TOKEN) ? n_tokens : MIN_ALLOC_TOKEN;
        if (ai->token_buffer) sqlite3_free(ai->token_buffer);
        ai->token_buffer = sqlite3_malloc64(n_count * sizeof(llama_token));
        if (!ai->token_buffer) {
            sqlite_context_result_error(context, SQLITE_NOMEM, "Out of memory: failed to allocate %d tokens", n_count);
            return;
        }
        ai->token_buffer_size = n_count;
    }
    llama_token *tokens = ai->token_buffer;
    
    // step3: tokenize
    int n_actual = llama_tokenize(vocab, text, text_len, tokens, n_tokens, true, true);
    if (n_actual != n_tokens) {
        sqlite_context_result_error(context, SQLITE_ERROR, "Tokenization size mismatch: got %d tokens, expected %d", n_actual, n_tokens);
        return;
    }
    
    // step 4: set up batch
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
    
    // step5: do real processing
    llama_memory_t memory = llama_get_memory(ctx);
    int32_t rc = (memory) ? llama_decode(ctx, batch) : llama_encode(ctx, batch);
    if (rc < 0) {
        sqlite_context_result_error(context, SQLITE_ERROR, "Model decode failed during embedding generation");
        llama_batch_free(batch);
        return;
    }
    
    // step6: retrieve embeddings
    const float *result = llama_get_embeddings(ctx);
    if (result == NULL) {
        sqlite_context_result_error(context, SQLITE_ERROR, "Failed to retrieve embedding vector from model");
        llama_batch_free(batch);
        return;
    }
    
    (ai->options.normalize_embedding) ? ai_embedding_normalize(result, embedding, dimension) : memcpy(embedding, result, sizeof(float) * dimension);
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

static void ai_embed (sqlite3_context *context, int argc, sqlite3_value **argv) {
    // sanity check arguments
    const char *function_name = "ai_embed";
    if (argc == 1) {
        int types[] = {SQLITE_TEXT};
        if (sqlite_sanity_function(context, function_name, argc, argv, 1, types, true) == false) return;
    } else if (argc == 2) {
        int types[] = {SQLITE_TEXT, SQLITE_TEXT};
        if (sqlite_sanity_function(context, function_name, argc, argv, 2, types, true) == false) return;
    } else {
        sqlite_context_result_error(context, SQLITE_ERROR, "Function '%s' expects 1 or 2 arguments, but %d were provided.", function_name, argc);
    }
    
    const char *text = (const char *)sqlite3_value_text(argv[0]);
    int32_t text_len = (int32_t)sqlite3_value_bytes(argv[0]);
    const char *model_options = (argc == 2) ? (const char *)sqlite3_value_text(argv[1]) : NULL;
    
    ai_context *ai = (ai_context *)sqlite3_user_data(context);
    if (parse_keyvalue_string(context, model_options, ai_options_callback, &ai->options) == false) return;
    
    ai_embed_generate(context, text, text_len);
}

// MARK: - Chat Virtual Table -

/*
    (NO) INSERT INTO ai_chat('Hello')
    (NO) INSERT INTO ai_chat(prompt) VALUES ('Hello')
    ---> SELECT reply FROM ai_chat('Hello', opt);
 */

static int vt_chat_connect (sqlite3 *db, void *pAux, int argc, const char *const *argv, sqlite3_vtab **ppVtab, char **pzErr) {
    int rc = sqlite3_declare_vtab(db, "CREATE TABLE x(response);");
    if (rc != SQLITE_OK) return rc;
    
    ai_vtab *vtab = (ai_vtab *)sqlite3_malloc(sizeof(ai_vtab));
    if (!vtab) return SQLITE_NOMEM;
    
    memset(vtab, 0, sizeof(ai_vtab));
    vtab->db = db;
    vtab->ai = (ai_context *)pAux;
    
    *ppVtab = (sqlite3_vtab *)vtab;
    return SQLITE_OK;
    
    struct llama_sampler *smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_min_p(0.05f, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.8f));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    
    
    return SQLITE_OK;
}

static int vt_chat_disconnect (sqlite3_vtab *pVtab) {
    return SQLITE_OK;
}

static int vt_chat_best_index (sqlite3_vtab *tab, sqlite3_index_info *pIdxInfo) {
    return SQLITE_OK;
}

static int vt_chat_cursor_open (sqlite3_vtab *pVtab, sqlite3_vtab_cursor **ppCursor) {
    return SQLITE_OK;
}

static int vt_chat_cursor_close (sqlite3_vtab_cursor *cur){
    return SQLITE_OK;
}

static int vt_chat_cursor_next (sqlite3_vtab_cursor *cur){
    return SQLITE_OK;
}

static int vt_chat_cursor_eof (sqlite3_vtab_cursor *cur){
    return 1;
}

static int vt_chat_cursor_column (sqlite3_vtab_cursor *cur, sqlite3_context *context, int iCol) {
    return SQLITE_OK;
}

static int vt_chat_cursor_rowid (sqlite3_vtab_cursor *cur, sqlite_int64 *pRowid) {
    return SQLITE_OK;
}

static int vt_chat_cursor_filter (sqlite3_vtab_cursor *cur, int idxNum, const char *idxStr, int argc, sqlite3_value **argv) {
    return SQLITE_OK;
}

/*
static int vt_chat_cursor_update (sqlite3_vtab *vtab, int argc, sqlite3_value **argv, sqlite3_int64 *rowid) {
    return SQLITE_OK;
}
 */

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
  /* xUpdate     */ 0, //vt_chat_cursor_update,
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

static void ai_set_model (sqlite3_context *context, int argc, sqlite3_value **argv) {
    // sanity check arguments
    const char *function_name = "ai_set_model";
    if (argc == 1) {
        int types[] = {SQLITE_TEXT};
        if (sqlite_sanity_function(context, function_name, argc, argv, 1, types, false) == false) return;
    } else if (argc == 2) {
        int types[] = {SQLITE_TEXT, SQLITE_TEXT};
        if (sqlite_sanity_function(context, function_name, argc, argv, 2, types, false) == false) return;
    } else {
        sqlite_context_result_error(context, SQLITE_ERROR, "Function '%s' expects 1 or 2 arguments, but %d were provided.", function_name, argc);
    }
    
    // model_arg can be a path or a name inside the ai_models table
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
    
    struct llama_context_params ctx_params = llama_context_default_params();
    ai_set_llama_options(&ctx_params, &ai->options);
    
    // for non-causal models, batch size must be equal to ubatch size
    // ctx_params.n_ubatch = ctx_params.n_batch;
    // ctx_params.n_ctx = 512;
    
    struct llama_context *ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        if (p == NULL) llama_model_free(model); // free only if loaded from file
        sqlite_context_result_error(context, SQLITE_ERROR, "Unable to create context from model %s", model_arg);
        return;
    }
    
    ai->model = model;
    ai->ctx = ctx;
}

static void ai_version (sqlite3_context *context, int argc, sqlite3_value **argv) {
    sqlite3_result_text(context, SQLITE_AI_VERSION, -1, NULL);
}

// MARK: -

SQLITE_AI_API int sqlite3_ai_init (sqlite3 *db, char **pzErrMsg, const sqlite3_api_routines *pApi) {
    #ifndef SQLITE_CORE
    SQLITE_EXTENSION_INIT2(pApi);
    #endif
    
    // set logger
    sqlite_utils_init();
    llama_log_set(ai_logger, db);
    
    // initialize the llama + ggml backend
    static bool once = false;
    if (once == false) {
        llama_backend_init();
        once = true;
    }
    
    // init context
    void *ctx = ai_context_create(db);
    if (!ctx) {
        if (pzErrMsg) *pzErrMsg = sqlite3_mprintf("Out of memory: failed to allocate AI extension context.");
        return SQLITE_NOMEM;
    }
    
    /*
    // bound ctx to db to be accessible also to virtual tables
    if (sqlite_set_ptr(db, ctx) == false) {
        ai_context_free((ai_context *)ctx);
        if (pzErrMsg) *pzErrMsg = sqlite3_mprintf("Unable to store AI extension context.");
        return SQLITE_ERROR;
    }
     */
    
    // register public functions
    int rc = SQLITE_OK;
    rc = sqlite3_create_function_v2(db, "ai_version", 0, SQLITE_UTF8, ctx, ai_version, NULL, NULL, ai_context_free);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function(db, "ai_set_model", 1, SQLITE_UTF8, ctx, ai_set_model, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function(db, "ai_set_model", 2, SQLITE_UTF8, ctx, ai_set_model, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function(db, "ai_embed", 1, SQLITE_UTF8, ctx, ai_embed, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function(db, "ai_embed", 2, SQLITE_UTF8, ctx, ai_embed, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_module(db, "ai_chat", &vt_chat, ctx);
    if (rc != SQLITE_OK) goto cleanup;
    
cleanup:
    return rc;
}
