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
#include "whisper.h"
#include "sqlite-ai.h"
#include "fp16/fp16.h"

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
#define MAX_LORAS                               64      // max 2 or 3 LoRa adapters are used (usually just one)

#define LOG_TABLE_DECLARATION                   "CREATE TEMP TABLE IF NOT EXISTS ai_log (id INTEGER PRIMARY KEY, stamp DATETIME DEFAULT CURRENT_TIMESTAMP, type TEXT, message TEXT);"
#define LOG_TABLE_INSERT_STMT                   "INSERT INTO ai_log (type, message) VALUES (?, ?);"

// CONTEXT OPTIONS
#define OPTION_KEY_CONTEXT_SIZE                 "context_size"
#define OPTION_KEY_N_CTX                        "n_ctx"
#define OPTION_KEY_N_BATCH                      "n_batch"
#define OPTION_KEY_N_UBATCH                     "n_ubatch"
#define OPTION_KEY_N_SEQ_MAX                    "n_seq_max"
#define OPTION_KEY_N_THREADS                    "n_threads"
#define OPTION_KEY_N_THREADS_BATCH              "n_threads_batch"
#define OPTION_KEY_ROPE_SCALING_TYPE            "rope_scaling_type"
#define OPTION_KEY_POOLING_TYPE                 "pooling_type"
#define OPTION_KEY_ATTENTION_TYPE               "attention_type"
#define OPTION_KEY_FLASH_ATTN_TYPE              "flash_attn_type"

#define OPTION_KEY_ROPE_FREQ_BASE               "rope_freq_base"
#define OPTION_KEY_ROPE_FREQ_SCALE              "rope_freq_scale"
#define OPTION_KEY_YARN_EXT_FACTOR              "yarn_ext_factor"
#define OPTION_KEY_YARN_ATTN_FACTOR             "yarn_attn_factor"
#define OPTION_KEY_YARN_BETA_FAST               "yarn_beta_fast"
#define OPTION_KEY_YARN_BETA_SLOW               "yarn_beta_slow"
#define OPTION_KEY_YARN_ORIG_CTX                "yarn_orig_ctx"
#define OPTION_KEY_DEFRAG_THOLD                 "defrag_thold"
#define OPTION_KEY_TYPE_K                       "type_k"
#define OPTION_KEY_TYPE_V                       "type_v"
#define OPTION_KEY_OFFLOAD_KQV                  "offload_kqv"
#define OPTION_KEY_OP_OFFLOAD                   "op_offload"
#define OPTION_KEY_SWA_FULL                     "swa_full"
#define OPTION_KEY_TYPE_KV_UNIFIED              "kv_unified"

#define OPTION_KEY_GENERATE_EMBEDDING           "generate_embedding"
#define OPTION_KEY_NORMALIZE_EMBEDDING          "normalize_embedding"
#define OPTION_KEY_JSON_OUTPUT                  "json_output"
#define OPTION_KEY_MAX_TOKENS                   "max_tokens"
#define OPTION_KEY_N_PREDICT                    "n_predict"
#define OPTION_KEY_EMBEDDING_TYPE               "embedding_type"


// MODEL OPTIONS
#define OPTION_KEY_GPU_LAYERS                   "gpu_layers"
#define OPTION_KEY_MAIN_GPU                     "main_gpu"
#define OPTION_KEY_SPLIT_MODE                   "split_mode"
#define OPTION_KEY_VOCAB_ONLY                   "vocab_only"
#define OPTION_KEY_USE_MMAP                     "use_mmap"
#define OPTION_KEY_USE_MLOCK                    "use_mlock"
#define OPTION_KEY_CHECK_TENSORS                "check_tensors"
#define OPTION_KEY_LOG_INFO                     "log_info"

#define AI_COLUMN_REPLY                         0

#define AI_DEFAULT_MODEL_OPTIONS                "gpu_layers=99"
#define AI_DEFAULT_CONTEXT_EMBEDDING_OPTIONS    "generate_embedding=1,normalize_embedding=1,pooling_type=mean"
#define AI_DEFAULT_CONTEXT_CHAT_OPTIONS         "context_size=4096"
#define AI_DEFAULT_CONTEXT_TEXTGEN_OPTIONS      "context_size=4096"

typedef enum {
    EMBEDDING_TYPE_F32 = 1,
    EMBEDDING_TYPE_F16,
    EMBEDDING_TYPE_BF16,
    EMBEDDING_TYPE_U8,
    EMBEDDING_TYPE_I8
} embedding_type; // same as vector_type from sqlite-vector

typedef struct {
    bool                        log_info;               // flag to enable/disable the logging of info (MODEL)
    uint32_t                    context_size;           // set both n_ctx and n_batch (CONTEXT)
    int                         n_predict;              // number of tokens to predict (SAMPLER)
    int32_t                     max_tokens;             // to control max allowed tokens to generate (to control user's input size) (CUSTOM)
    struct {
        embedding_type          type;
        bool                    normalize;              // if true, embeddings are normalized
        bool                    json_output;            // if true, embedding result is converted to JSON
    } embedding;
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
    struct llama_adapter_lora   *lora[MAX_LORAS];
    float                       lora_scale[MAX_LORAS];
    
    llm_options                 options;
    
    // whisper
    struct whisper_context      *whisper;
    
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

typedef enum {
    AI_MODEL_SETTING_N_PARAMS = 1,
    AI_MODEL_SIZE,
    AI_MODEL_N_CTX_TRAIN,
    AI_MODEL_N_EMBED,
    AI_MODEL_N_LAYER,
    AI_MODEL_N_HEAD,
    AI_MODEL_N_HEAD_KV,
    AI_MODEL_N_SWA,
    AI_MODEL_FREQ_SCALE_TRAIN,
    AI_MODEL_N_CLS_OUT,
    AI_MODEL_HAS_ENCODER,
    AI_MODEL_HAS_DECODER,
    AI_MODEL_IS_RECURRENT,
    AI_MODEL_CHAT_TEMPLATE
} ai_model_setting;

const char *ROLE_USER       = "user";
const char *ROLE_ASSISTANT  = "assistant";

void ai_logger (enum ggml_log_level level, const char *text, void *user_data);

// MARK: - NUMERICS -
// typedef uint16_t bfloat16_t;    // don't typedef to bfloat16_t to avoid mix with <arm_neon.h>’s native bfloat16_t

// float <-> uint32_t bit casts
static inline uint32_t f32_to_bits (float f) {
    #if defined(HAVE_BUILTIN_BIT_CAST)
    return __builtin_bit_cast(uint32_t, f);
    #else
    union { float f; uint32_t u; } v = { .f = f };
    return v.u;
    #endif
}

static inline float bits_to_f32 (uint32_t u) {
    #if defined(HAVE_BUILTIN_BIT_CAST)
    return __builtin_bit_cast(float, u);
    #else
    union { uint32_t u; float f; } v = { .u = u };
    return v.f;
    #endif
}

#if 0
// bfloat16 (stored as uint16_t) -> float32, and back (RNE)
static inline bool bfloat16_is_nan (uint16_t h) {      /* exp==0xFF && frac!=0 */
    return ((h & 0x7F80u) == 0x7F80u) && ((h & 0x007Fu) != 0);
}
static inline bool bfloat16_is_inf (uint16_t h) {      /* exp==0xFF && frac==0 */
    return ((h & 0x7F80u) == 0x7F80u) && ((h & 0x007Fu) == 0);
}
static inline bool bfloat16_is_zero (uint16_t h) {     /* ±0 */
    return (h & 0x7FFFu) == 0;
}
static inline int bfloat16_sign (uint16_t h) {
    return (h >> 15) & 1;
}
#endif

static inline float bfloat16_to_float32 (uint16_t bf) {
    return bits_to_f32((uint32_t)bf << 16);
}
static inline uint16_t float32_to_bfloat16 (float f) {
    uint32_t x = f32_to_bits(f);
    uint32_t lsb = (x >> 16) & 1u;      /* ties-to-even */
    uint32_t rnd = 0x7FFFu + lsb;
    return (uint16_t)((x + rnd) >> 16);
}

#if 0
// ---- float16 (binary16) classifiers (work on raw uint16_t bits)
static inline bool f16_is_nan (uint16_t h) {      /* exp==0x1F && frac!=0 */
    return ( (h & 0x7C00u) == 0x7C00u ) && ((h & 0x03FFu) != 0);
}
static inline bool f16_is_inf (uint16_t h) {      /* exp==0x1F && frac==0 */
    return ( (h & 0x7C00u) == 0x7C00u ) && ((h & 0x03FFu) == 0);
}
static inline int  f16_sign (uint16_t h) {
    return (h >> 15) & 1;
}
static inline bool f16_is_zero (uint16_t h) {     /* ±0 */
    return (h & 0x7FFFu) == 0;
}
#endif

static inline uint16_t float32_to_float16 (float f) {
    return fp16_ieee_from_fp32_value(f);
}
static inline float float16_to_float32 (uint16_t h) {
    return fp16_ieee_to_fp32_value(h);
}

static inline uint8_t sat_u8_from_f32 (float x) {
    if (!isfinite(x)) return x > 0 ? 255 : 0;
    long v = lrintf(x);                   // round to nearest, ties-to-even
    if (v < 0)   return 0;
    if (v > 255) return 255;
    return (uint8_t)v;
}

static inline int8_t sat_i8_from_f32 (float x) {
    if (!isfinite(x)) return x > 0 ? 127 : -128;
    long v = lrintf(x);
    if (v < -128) return -128;
    if (v > 127)  return 127;
    return (int8_t)v;
}

static size_t embedding_type_to_size (embedding_type type) {
    switch (type) {
        case EMBEDDING_TYPE_F32: return sizeof(float);
        case EMBEDDING_TYPE_F16: return sizeof(uint16_t);
        case EMBEDDING_TYPE_BF16: return sizeof(uint16_t);
        case EMBEDDING_TYPE_U8: return sizeof(uint8_t);
        case EMBEDDING_TYPE_I8: return sizeof(int8_t);
    }
    return 0;
}

static embedding_type embedding_name_to_type (const char *vname) {
    if (strcasecmp(vname, "FLOAT32") == 0) return EMBEDDING_TYPE_F32;
    if (strcasecmp(vname, "FLOAT16") == 0) return EMBEDDING_TYPE_F16;
    if (strcasecmp(vname, "FLOATB16") == 0) return EMBEDDING_TYPE_BF16;
    if (strcasecmp(vname, "UINT8") == 0) return EMBEDDING_TYPE_U8;
    if (strcasecmp(vname, "INT8") == 0) return EMBEDDING_TYPE_I8;
    return 0;
}

const char *embedding_type_to_name (embedding_type type) {
    switch (type) {
        case EMBEDDING_TYPE_F32: return "FLOAT32";
        case EMBEDDING_TYPE_F16: return "FLOAT16";
        case EMBEDDING_TYPE_BF16: return "FLOATB16";
        case EMBEDDING_TYPE_U8: return "UINT8";
        case EMBEDDING_TYPE_I8: return "INT8";
    }
    return "N/A";
}

// MARK: -

static void llm_options_init (llm_options *options) {
    memset(options, 0, sizeof(llm_options));
    
    options->embedding.normalize = true;
    options->max_tokens = 0;    // no limits
    options->log_info = false;  // disable INFO messages logging
}

static bool llm_model_options_callback (void *ctx, void *xdata, const char *key, int key_len, const char *value, int value_len) {
    struct llama_model_params *options = (struct llama_model_params *)xdata;
    ai_context *ai = (ai_context *)ctx;
    
    // sanity check (ignore malformed key/value)
    if (!key || key_len == 0) return true;
    if (!value || value_len == 0) return true;
    
    // debug
    // printf("KEY: \"%.*s\", VALUE: \"%.*s\"\n", key_len, key, value_len, value);
    
    // convert value to c-string
    char buffer[256] = {0};
    size_t len = (value_len > sizeof(buffer)-1) ? sizeof(buffer)-1 : value_len;
    memcpy(buffer, value, len);
    
    // MODEL OPTIONS
    if (strncasecmp(key, OPTION_KEY_GPU_LAYERS, key_len) == 0) {
        int value = (int)strtol(buffer, NULL, 0);
        options->n_gpu_layers = value;
        return true;
    }
    
    if (strncasecmp(key, OPTION_KEY_MAIN_GPU, key_len) == 0) {
        int value = (int)strtol(buffer, NULL, 0);
        options->main_gpu = value;
        return true;
    }
    
    if (strncasecmp(key, OPTION_KEY_SPLIT_MODE, key_len) == 0) {
        int value = (int)strtol(buffer, NULL, 0);
        if (value >= 0 && value <= 2) options->split_mode = value;
        return true;
    }
    
    if (strncasecmp(key, OPTION_KEY_VOCAB_ONLY, key_len) == 0) {
        int value = (int)strtol(buffer, NULL, 0);
        options->vocab_only = (value != 0);
        return true;
    }
    
    if (strncasecmp(key, OPTION_KEY_USE_MMAP, key_len) == 0) {
        int value = (int)strtol(buffer, NULL, 0);
        options->use_mmap = (value != 0);
        return true;
    }
    
    if (strncasecmp(key, OPTION_KEY_USE_MLOCK, key_len) == 0) {
        int value = (int)strtol(buffer, NULL, 0);
        options->use_mlock = (value != 0);
        return true;
    }
    
    if (strncasecmp(key, OPTION_KEY_CHECK_TENSORS, key_len) == 0) {
        int value = (int)strtol(buffer, NULL, 0);
        options->check_tensors = (value != 0);
        return true;
    }
    
    if (strncasecmp(key, OPTION_KEY_LOG_INFO, key_len) == 0) {
        int value = (int)strtol(buffer, NULL, 0);
        ai->options.log_info = (value != 0);
        return true;
    }
    
    return true;
}

static bool llm_context_options_callback (void *ctx, void *xdata, const char *key, int key_len, const char *value, int value_len) {
    struct llama_context_params *options = (struct llama_context_params *)xdata;
    ai_context *ai = (ai_context *)ctx;
    
    // sanity check (ignore malformed key/value)
    if (!key || key_len == 0) return true;
    if (!value || value_len == 0) return true;
    
    // debug
    // printf("KEY: \"%.*s\", VALUE: \"%.*s\"\n", key_len, key, value_len, value);
    
    // convert value to c-string
    char buffer[256] = {0};
    size_t len = (value_len > sizeof(buffer)-1) ? sizeof(buffer)-1 : value_len;
    memcpy(buffer, value, len);
    
    // AI CONTEXT (OPTIONS can be NULL)
    if (strncasecmp(key, OPTION_KEY_NORMALIZE_EMBEDDING, key_len) == 0) {
        int value = (int)strtol(buffer, NULL, 0);
        ai->options.embedding.normalize = (value != 0);
        return true;
    }
    
    if (strncasecmp(key, OPTION_KEY_JSON_OUTPUT, key_len) == 0) {
        int value = (int)strtol(buffer, NULL, 0);
        ai->options.embedding.json_output = (value != 0);
        return true;
    }
    
    if (strncasecmp(key, OPTION_KEY_MAX_TOKENS, key_len) == 0) {
        int value = (int)strtol(buffer, NULL, 0);
        if (value >= 0) ai->options.max_tokens = value;
        return true;
    }
    
    if (strncasecmp(key, OPTION_KEY_N_PREDICT, key_len) == 0) {
        int value = (int)strtol(buffer, NULL, 0);
        if (value >= 0) ai->options.n_predict = value;
        return true;
    }
    
    if (strncasecmp(key, OPTION_KEY_EMBEDDING_TYPE, key_len) == 0) {
        int value = embedding_name_to_type(buffer);
        if (value > 0) ai->options.embedding.type = value;
        return true;
    }
    
    // CONTEXT OPTIONS
    if (options == NULL) {
        char buffer[512];
        snprintf(buffer, sizeof(buffer), "key %.*s ignored because context was already created", key_len, key);
        ai_logger(GGML_LOG_LEVEL_WARN, buffer, ai);
        return true;
    }
    
    if (strncasecmp(key, OPTION_KEY_GENERATE_EMBEDDING, key_len) == 0) {
        // https://github.com/ggml-org/llama.cpp/discussions/15093
        int value = (int)strtol(buffer, NULL, 0);
        options->embeddings = (value != 0);
        options->pooling_type = LLAMA_POOLING_TYPE_MEAN;
        
        // for non-causal models, batch size must be equal to ubatch size
        // when generating embeddings, always tie them together.
        options->n_ubatch = options->n_batch;
        return true;
    }
    
    if (strncasecmp(key, OPTION_KEY_CONTEXT_SIZE, key_len) == 0) {
        int value = (int)strtol(buffer, NULL, 0);
        if (value >= 0) {
            options->n_ctx = value;
            options->n_batch = value;
        }
        return true;
    }
    
    if (strncasecmp(key, OPTION_KEY_N_CTX, key_len) == 0) {
        int value = (int)strtol(buffer, NULL, 0);
        if (value >= 0) options->n_ctx = value;
        return true;
    }
    
    if (strncasecmp(key, OPTION_KEY_N_BATCH, key_len) == 0) {
        int value = (int)strtol(buffer, NULL, 0);
        if (value >= 0) options->n_batch = value;
        return true;
    }
    
    if (strncasecmp(key, OPTION_KEY_N_UBATCH, key_len) == 0) {
        int value = (int)strtol(buffer, NULL, 0);
        if (value >= 0) options->n_ubatch = value;
        return true;
    }
    
    if (strncasecmp(key, OPTION_KEY_N_SEQ_MAX, key_len) == 0) {
        int value = (int)strtol(buffer, NULL, 0);
        if (value >= 0) options->n_seq_max = value;
        return true;
    }
    
    if (strncasecmp(key, OPTION_KEY_N_THREADS, key_len) == 0) {
        int value = (int)strtol(buffer, NULL, 0);
        if (value >= 0) options->n_threads = value;
        return true;
    }
    
    if (strncasecmp(key, OPTION_KEY_N_THREADS_BATCH, key_len) == 0) {
        int value = (int)strtol(buffer, NULL, 0);
        if (value >= 0) options->n_threads_batch = value;
        return true;
    }
    
    if (strncasecmp(key, OPTION_KEY_POOLING_TYPE, key_len) == 0) {
        // pooling_type mean is not supported and so in this version we forced it to be really mean so ONE EMBEDDING will be generated
        if (strcasecmp(buffer, "none") == 0) options->pooling_type = LLAMA_POOLING_TYPE_MEAN;
        else if (strcasecmp(buffer, "unspecified") == 0) options->pooling_type = LLAMA_POOLING_TYPE_MEAN;
        else if (strcasecmp(buffer, "mean") == 0) options->pooling_type = LLAMA_POOLING_TYPE_MEAN;
        else if (strcasecmp(buffer, "cls") == 0) options->pooling_type = LLAMA_POOLING_TYPE_CLS;
        else if (strcasecmp(buffer, "last") == 0) options->pooling_type = LLAMA_POOLING_TYPE_LAST;
        else if (strcasecmp(buffer, "rank") == 0) options->pooling_type = LLAMA_POOLING_TYPE_RANK;
        return true;
    }
    
    if (strncasecmp(key, OPTION_KEY_ATTENTION_TYPE, key_len) == 0) {
        if (strcasecmp(buffer, "unspecified") == 0) options->attention_type = LLAMA_ATTENTION_TYPE_UNSPECIFIED;
        else if (strcasecmp(buffer, "causal") == 0) options->attention_type = LLAMA_ATTENTION_TYPE_CAUSAL;
        else if (strcasecmp(buffer, "non_causal") == 0) options->attention_type = LLAMA_ATTENTION_TYPE_NON_CAUSAL;
        return true;
    }
    
    if (strncasecmp(key, OPTION_KEY_ROPE_SCALING_TYPE, key_len) == 0) {
        if (strcasecmp(buffer, "unspecified") == 0) options->rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED;
        else if (strcasecmp(buffer, "none") == 0) options->rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_NONE;
        else if (strcasecmp(buffer, "linear") == 0) options->rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_LINEAR;
        else if (strcasecmp(buffer, "yarn") == 0) options->rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_YARN;
        else if (strcasecmp(buffer, "longrope") == 0) options->rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_LONGROPE;
        return true;
    }
    
    if (strncasecmp(key, OPTION_KEY_FLASH_ATTN_TYPE, key_len) == 0) {
        if (strcasecmp(buffer, "auto") == 0) options->flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
        else if (strcasecmp(buffer, "disabled") == 0) options->flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        else if (strcasecmp(buffer, "enabled") == 0) options->flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        return true;
    }
    
    if (strncasecmp(key, OPTION_KEY_ROPE_FREQ_BASE, key_len) == 0) {
        float value = strtof(buffer, NULL);
        options->rope_freq_base = value;
        return true;
    }
    
    if (strncasecmp(key, OPTION_KEY_ROPE_FREQ_SCALE, key_len) == 0) {
        float value = strtof(buffer, NULL);
        options->rope_freq_scale = value;
        return true;
    }
    
    if (strncasecmp(key, OPTION_KEY_YARN_EXT_FACTOR, key_len) == 0) {
        float value = strtof(buffer, NULL);
        options->yarn_ext_factor = value;
        return true;
    }
    
    if (strncasecmp(key, OPTION_KEY_YARN_ATTN_FACTOR, key_len) == 0) {
        float value = strtof(buffer, NULL);
        options->yarn_attn_factor = value;
        return true;
    }
    
    if (strncasecmp(key, OPTION_KEY_YARN_BETA_FAST, key_len) == 0) {
        float value = strtof(buffer, NULL);
        options->yarn_beta_fast = value;
        return true;
    }
    
    if (strncasecmp(key, OPTION_KEY_YARN_BETA_SLOW, key_len) == 0) {
        float value = strtof(buffer, NULL);
        options->yarn_beta_slow = value;
        return true;
    }
    
    if (strncasecmp(key, OPTION_KEY_DEFRAG_THOLD, key_len) == 0) {
        float value = strtof(buffer, NULL);
        options->defrag_thold = value;
        return true;
    }
    
    if (strncasecmp(key, OPTION_KEY_YARN_ORIG_CTX, key_len) == 0) {
        int value = (int)strtol(buffer, NULL, 0);
        if (value >= 0) options->yarn_orig_ctx = value;
        return true;
    }
    
    if (strncasecmp(key, OPTION_KEY_OFFLOAD_KQV, key_len) == 0) {
        int value = (int)strtol(buffer, NULL, 0);
        options->offload_kqv = (value != 0);
        return true;
    }
    
    if (strncasecmp(key, OPTION_KEY_OP_OFFLOAD, key_len) == 0) {
        int value = (int)strtol(buffer, NULL, 0);
        options->op_offload = (value != 0);
        return true;
    }
    
    if (strncasecmp(key, OPTION_KEY_SWA_FULL, key_len) == 0) {
        int value = (int)strtol(buffer, NULL, 0);
        options->swa_full = (value != 0);
        return true;
    }
    
    if (strncasecmp(key, OPTION_KEY_TYPE_K, key_len) == 0) {
        int value = (int)strtol(buffer, NULL, 0);
        if (value >= 0) options->type_k = value;
        return true;
    }
    
    if (strncasecmp(key, OPTION_KEY_TYPE_V, key_len) == 0) {
        int value = (int)strtol(buffer, NULL, 0);
        if (value >= 0) options->type_v = value;
        return true;
    }
    
    if (strncasecmp(key, OPTION_KEY_TYPE_KV_UNIFIED, key_len) == 0) {
        int value = (int)strtol(buffer, NULL, 0);
        if (value >= 0) options->kv_unified = (value != 0);
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

static int llm_lora_push (ai_context *ai, struct llama_adapter_lora *lora, float scale) {
    for (int i=0; i<MAX_LORAS; ++i) {
        if (ai->lora[i] == NULL) {
            ai->lora[i] = lora;
            ai->lora_scale[i] = scale;
            return i;
        }
    }
    return -1;
}

// MARK: -

static bool whisper_model_options_callback (void *ctx, void *xdata, const char *key, int key_len, const char *value, int value_len) {
    //struct whisper_context_params *whisper_params = (struct whisper_context_params *)xdata;
    //ai_context *ai = (ai_context *)ctx;
    return true;
}

static bool whisper_full_params_options_callback (void *ctx, void *xdata, const char *key, int key_len, const char *value, int value_len) {
    //struct whisper_full_params *params = (struct whisper_full_params *)xdata;
    //ai_context *ai = (ai_context *)ctx;
    return true;
}

// MARK: -

void *ai_create (sqlite3 *db) {
    ai_context *ai = (ai_context *)sqlite3_malloc(sizeof(ai_context));
    if (ai) {
        memset(ai, 0, sizeof(ai_context));
        llm_options_init(&ai->options);
        ai->db = db;
    }
    return ai;
}

static void ai_free (void *ctx, bool free_ai, bool free_llm, bool free_audio) {
    if (!ctx) return;
    ai_context *ai = (ai_context *)ctx;
    
    // disable logger first
    if (free_ai) {
        ai->db = NULL;
        free_llm = true;
        free_audio = true;
    }
    
    if (free_llm) {
        memset(ai->lora, 0, sizeof(struct llama_adapter_lora *)*MAX_LORAS);
        memset(ai->lora_scale, 0, sizeof(float)*MAX_LORAS);
        if (ai->ctx) llama_clear_adapter_lora(ai->ctx);
        if (ai->ctx) llama_free(ai->ctx);
        if (ai->model) llama_model_free(ai->model);
        // important: do not free if the sampler has been added to a llama_sampler_chain (via llama_sampler_chain_add)
        // if (ai->sampler) llama_sampler_free(ai->sampler);
        llm_options_init(&ai->options);
        
        ai->model = NULL;
        ai->ctx = NULL;
        ai->sampler = NULL;
    }
    
    if (free_audio) {
        if (ai->whisper) whisper_free(ai->whisper);
        ai->whisper = NULL;
    }
    
    if (free_ai) {
        sqlite3_free(ai);
    }
}

static void ai_cleanup (void *ctx, bool free_llm, bool free_audio) {
    ai_free(ctx, false, free_llm, free_audio);
}

static void ai_destroy (void *ctx) {
    ai_free(ctx, true, true, true);
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
    int types[] = {(type == NULL) ? SQLITE_NULL : SQLITE_TEXT, SQLITE_TEXT};
    int lens[] = {-1, -1};
    sqlite_db_write(NULL, ai->db, LOG_TABLE_INSERT_STMT, values, types, lens, 2);
}

bool ai_model_check (sqlite3_context *context, bool check_llm, bool check_audio) {
    ai_context *ai = (ai_context *)sqlite3_user_data(context);
    if (!ai) return false;
    if (check_llm && (ai->model == NULL)) return false;
    if (check_audio && (ai->whisper == NULL)) return false;
    return true;
}

static bool llm_common_args_check (sqlite3_context *context, const char *function_name, int argc, sqlite3_value **argv, bool check_llm_model) {
    // sanity check arguments
    if (argc == 1) {
        int types[] = {SQLITE_TEXT};
        return sqlite_sanity_function(context, function_name, argc, argv, 1, types, check_llm_model, false);
    } else if (argc == 2) {
        int types[] = {SQLITE_TEXT, SQLITE_TEXT};
        return sqlite_sanity_function(context, function_name, argc, argv, 2, types, check_llm_model, false);
    }
    
    return sqlite_context_result_error(context, SQLITE_ERROR, "Function '%s' expects 1 or 2 arguments, but %d were provided.", function_name, argc);
}

static bool llm_check_context (sqlite3_context *context) {
    ai_context *ai = (ai_context *)sqlite3_user_data(context);
    if (!ai || !ai->ctx) {
        return sqlite_context_result_error(context, SQLITE_MISUSE, "No context found. Please call llm_context_create() before using this function.");
    }
    
    return true;
}

// MARK: - Chat Messages -

bool llm_messages_append (ai_messages *list, const char *role, const char *content) {
    if (list->count >= list->capacity) {
        size_t new_cap = list->capacity ? list->capacity * 2 : MIN_ALLOC_MESSAGES;
        llama_chat_message *new_items = sqlite3_realloc64(list->items, new_cap * sizeof(llama_chat_message));
        if (!new_items) return false;
        
        list->items = new_items;
        list->capacity = new_cap;
    }

    bool duplicate_role = ((role != ROLE_USER) && (role != ROLE_ASSISTANT));
    list->items[list->count].role = (duplicate_role) ? sqlite_strdup(role) : role;
    list->items[list->count].content = sqlite_strdup(content);
    list->count += 1;
    return true;
}

void llm_messages_free (ai_messages *list) {
    for (size_t i = 0; i < list->count; ++i) {
        // check if rule is static
        const char *role = list->items[i].role;
        bool role_tofree = ((role != ROLE_USER) && (role != ROLE_ASSISTANT));
        if (role_tofree) sqlite3_free((char *)list->items[i].role);
        // content is always to free
        sqlite3_free((char *)list->items[i].content);
    }
    sqlite3_free(list->items);
    
    list->items = NULL;
    list->count = 0;
    list->capacity = 0;
}

// MARK: - Text Embedding and Normalization -

static inline float llm_common_f32_sum (const float *src, int dim) {
    float sum = 0.0f;
    
    // compute L2 norm squared (loop unrolled by 4)
    int i = 0;
    for (; i + 3 < dim; i += 4) {
        sum += src[i] * src[i] + src[i + 1] * src[i + 1] + src[i + 2] * src[i + 2] + src[i + 3] * src[i + 3];
    }
    for (; i < dim; ++i) {
        sum += src[i] * src[i];
    }
    
    return sum;
}

static int llm_embed_normalize_f32 (const float *src, float *dest, int dim) {
    float sum = llm_common_f32_sum(src, dim);
    
    float norm = sqrtf(sum);
    if (norm > 0.0f) {
        float inv = 1.0f / norm;
        int i = 0;
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
    
    return 0;
}

static int llm_embed_normalize_f16 (const float *src, uint16_t *dest, int dim) {
    float sum = llm_common_f32_sum(src, dim);

    if (sum > 0.0f) {
        float inv = 1.0f / sqrtf(sum);
        int i = 0;
        for (; i + 3 < dim; i += 4) {
            dest[i]     = float32_to_float16(src[i]     * inv);
            dest[i + 1] = float32_to_float16(src[i + 1] * inv);
            dest[i + 2] = float32_to_float16(src[i + 2] * inv);
            dest[i + 3] = float32_to_float16(src[i + 3] * inv);
        }
        for (; i < dim; ++i) {
            dest[i] = float32_to_float16(src[i] * inv);
        }
    } else {
        for (int j = 0; j < dim; ++j) dest[j] = 0; // +0.0
    }
    return 0;
}

static int llm_embed_normalize_bf16 (const float *src, uint16_t *dest, int dim) {
    float sum = llm_common_f32_sum(src, dim);

    if (sum > 0.0f) {
        float inv = 1.0f / sqrtf(sum);
        int i = 0;
        for (; i + 3 < dim; i += 4) {
            dest[i]     = float32_to_bfloat16(src[i]     * inv);
            dest[i + 1] = float32_to_bfloat16(src[i + 1] * inv);
            dest[i + 2] = float32_to_bfloat16(src[i + 2] * inv);
            dest[i + 3] = float32_to_bfloat16(src[i + 3] * inv);
        }
        for (; i < dim; ++i) {
            dest[i] = float32_to_bfloat16(src[i] * inv);
        }
    } else {
        for (int j = 0; j < dim; ++j) dest[j] = 0; // +0.0
    }
    return 0;
}

static int llm_embed_normalize_i8 (const float *src, int8_t *dest, int dim) {
    float sum = llm_common_f32_sum(src, dim);

    if (sum > 0.0f) {
        float inv = 1.0f / sqrtf(sum);
        int i = 0;
        for (; i + 3 < dim; i += 4) {
            int q0 = (int)lrintf(src[i]     * inv * 127.0f);
            int q1 = (int)lrintf(src[i + 1] * inv * 127.0f);
            int q2 = (int)lrintf(src[i + 2] * inv * 127.0f);
            int q3 = (int)lrintf(src[i + 3] * inv * 127.0f);
            if (q0 > 127) q0 = 127; else if (q0 < -127) q0 = -127;
            if (q1 > 127) q1 = 127; else if (q1 < -127) q1 = -127;
            if (q2 > 127) q2 = 127; else if (q2 < -127) q2 = -127;
            if (q3 > 127) q3 = 127; else if (q3 < -127) q3 = -127;
            dest[i]     = (int8_t)q0;
            dest[i + 1] = (int8_t)q1;
            dest[i + 2] = (int8_t)q2;
            dest[i + 3] = (int8_t)q3;
        }
        for (; i < dim; ++i) {
            int q = (int)lrintf(src[i] * inv * 127.0f);
            if (q > 127) q = 127; else if (q < -127) q = -127;
            dest[i] = (int8_t)q;
        }
    } else {
        for (int j = 0; j < dim; ++j) dest[j] = 0;
    }
    return 0;
}

static int llm_embed_normalize_u8 (const float *src, uint8_t *dest, int dim) {
    float sum = llm_common_f32_sum(src, dim);

    if (sum > 0.0f) {
        float inv = 1.0f / sqrtf(sum);
        int i = 0;
        for (; i + 3 < dim; i += 4) {
            int q0 = (int)lrintf(src[i]     * inv * 127.0f + 128.0f);
            int q1 = (int)lrintf(src[i + 1] * inv * 127.0f + 128.0f);
            int q2 = (int)lrintf(src[i + 2] * inv * 127.0f + 128.0f);
            int q3 = (int)lrintf(src[i + 3] * inv * 127.0f + 128.0f);
            if (q0 < 0) q0 = 0; else if (q0 > 255) q0 = 255;
            if (q1 < 0) q1 = 0; else if (q1 > 255) q1 = 255;
            if (q2 < 0) q2 = 0; else if (q2 > 255) q2 = 255;
            if (q3 < 0) q3 = 0; else if (q3 > 255) q3 = 255;
            dest[i]     = (uint8_t)q0;
            dest[i + 1] = (uint8_t)q1;
            dest[i + 2] = (uint8_t)q2;
            dest[i + 3] = (uint8_t)q3;
        }
        for (; i < dim; ++i) {
            int q = (int)lrintf(src[i] * inv * 127.0f + 128.0f);
            if (q < 0) q = 0; else if (q > 255) q = 255;
            dest[i] = (uint8_t)q;
        }
    } else {
        // represent zero as the zero-point
        for (int j = 0; j < dim; ++j) dest[j] = 128;
    }
    return 0;
}

static int llm_embed_normalize (const float *src, void *dest, embedding_type type, int dim) {
    switch (type) {
        case EMBEDDING_TYPE_F32: return llm_embed_normalize_f32(src, (float *)dest, dim);
        case EMBEDDING_TYPE_F16: return llm_embed_normalize_f16(src, (uint16_t *)dest, dim);
        case EMBEDDING_TYPE_BF16: return llm_embed_normalize_bf16(src, (uint16_t *)dest, dim);
        case EMBEDDING_TYPE_U8: return llm_embed_normalize_u8(src, (uint8_t *)dest, dim);
        case EMBEDDING_TYPE_I8: return llm_embed_normalize_i8(src, (int8_t *)dest, dim);
    }
    return 0;
}

static int llm_embed_copy (const float *src, void *dest, embedding_type type, int dim, int bsize) {
    switch (type) {
        case EMBEDDING_TYPE_F32:
            // if embedding size is the same as src then just copy src into dest
            memcpy(dest, src, bsize);
            break;
            
        case EMBEDDING_TYPE_F16: {
            uint16_t *buffer = (uint16_t *)dest;
            int i = 0;
            for (; i + 3 < dim; i += 4) {
                buffer[i+0] = float32_to_float16(src[i+0]);
                buffer[i+1] = float32_to_float16(src[i+1]);
                buffer[i+2] = float32_to_float16(src[i+2]);
                buffer[i+3] = float32_to_float16(src[i+3]);
            }
            for (; i < dim; ++i) buffer[i] = float32_to_float16(src[i]);
            break;
        }
            
        case EMBEDDING_TYPE_BF16: {
            uint16_t *buffer = (uint16_t *)dest;
            int i = 0;
            for (; i + 3 < dim; i += 4) {
                buffer[i+0] = float32_to_bfloat16(src[i+0]);
                buffer[i+1] = float32_to_bfloat16(src[i+1]);
                buffer[i+2] = float32_to_bfloat16(src[i+2]);
                buffer[i+3] = float32_to_bfloat16(src[i+3]);
            }
            for (; i < dim; ++i) buffer[i] = float32_to_bfloat16(src[i]);
            break;
        }
            
        case EMBEDDING_TYPE_U8: {
            uint8_t *buffer = (uint8_t *)dest;
            int i = 0;
            for (; i + 3 < dim; i += 4) {
                buffer[i+0] = sat_u8_from_f32(src[i+0]);
                buffer[i+1] = sat_u8_from_f32(src[i+1]);
                buffer[i+2] = sat_u8_from_f32(src[i+2]);
                buffer[i+3] = sat_u8_from_f32(src[i+3]);
            }
            for (; i < dim; ++i) buffer[i] = sat_u8_from_f32(src[i]);
            break;
        }
            
        case EMBEDDING_TYPE_I8: {
            int8_t *buffer = (int8_t *)dest;
            int i = 0;
            for (; i + 3 < dim; i += 4) {
                buffer[i+0] = sat_i8_from_f32(src[i+0]);
                buffer[i+1] = sat_i8_from_f32(src[i+1]);
                buffer[i+2] = sat_i8_from_f32(src[i+2]);
                buffer[i+3] = sat_i8_from_f32(src[i+3]);
            }
            for (; i < dim; ++i) buffer[i] = sat_i8_from_f32(src[i]);
            break;
        }
    }
    
    return 0;
}

static void llm_batch_clear (struct llama_batch *batch) {
    batch->n_tokens = 0;
}

static void llm_batch_add (struct llama_batch *batch, llama_token id, llama_pos pos, const llama_seq_id *seq_ids, size_t n_seq_ids, bool logits) {
    batch->token   [batch->n_tokens] = id;
    batch->pos     [batch->n_tokens] = pos;
    batch->n_seq_id[batch->n_tokens] = (int32_t)n_seq_ids;
    
    for (size_t i = 0; i < n_seq_ids; ++i) {
        batch->seq_id[batch->n_tokens][i] = seq_ids[i];
    }
    
    batch->logits[batch->n_tokens] = logits ? 1 : 0;
    batch->n_tokens++;
}

#if 0
static void llm_embed_debug (const void *buf, embedding_type type, int n) {
    printf("type: %s - dim: %d [", embedding_type_to_name(type), n);
    for (int i=0; i<n; ++i) {
        switch (type) {
            case EMBEDDING_TYPE_F32: {
                float *f = (float *)buf;
                printf("%f,", f[i]);
            }
            break;
                
            case EMBEDDING_TYPE_F16: {
                uint16_t *f = (uint16_t *)buf;
                printf("%f,", float16_to_float32(f[i]));
            }
            break;
                
            case EMBEDDING_TYPE_BF16: {
                uint16_t *f = (uint16_t *)buf;
                printf("%f,", bfloat16_to_float32(f[i]));
            }
            break;
                
            case EMBEDDING_TYPE_U8: {
                uint8_t *u = (uint8_t *)buf;
                printf("%d,", u[i]);
            }
            break;
                
            case EMBEDDING_TYPE_I8: {
                int8_t *u = (int8_t *)buf;
                printf("%d,", u[i]);
            }
            break;
        }
    }
    printf("]\n");
}
#endif

static void llm_embed_generate_run (sqlite3_context *context, const char *text, int32_t text_len) {
    ai_context *ai = (ai_context *)sqlite3_user_data(context);
    struct llama_model *model = ai->model;
    
    // sanity check model
    if (llama_model_has_encoder(model) && llama_model_has_decoder(model)) {
        sqlite_context_result_error(context, SQLITE_ERROR, "Computing embeddings in encoder-decoder models is not supported");
        return;
    }
    
    // sanity check model type (decode is used to create embeddings)
    if (llama_model_has_decoder(model) == false) {
        sqlite_context_result_error(context, SQLITE_ERROR, "Model does not support decoding (required for embedding)");
        return;
    }
    
    // sanity check vocab
    const struct llama_vocab *vocab = llama_model_get_vocab(model);
    if (!vocab) {
        sqlite_context_result_error(context, SQLITE_ERROR, "Failed to extract vocabulary from the model");
        return;
    }
    
    // pooling is NOT NONE -> one sentence-level embedding
    // more details in notes/EMBEDDING.md
    struct llama_context *ctx = ai->ctx;
    llama_set_embeddings(ctx, true);
    
    // sanity check context / training window info (warn only)
    const int n_ctx_train = llama_model_n_ctx_train(model);
    const int n_ctx = llama_n_ctx(ctx);
    if (n_ctx > n_ctx_train) {
        char buffer[512];
        snprintf(buffer, sizeof(buffer), "Model was trained on only %d context tokens (%d specified)", n_ctx_train, n_ctx);
        ai_logger(GGML_LOG_LEVEL_WARN, buffer, sqlite3_context_db_handle(context));
    }
    
    // allocate embedding buffer
    int dimension = llama_model_n_embd(llama_get_model(ctx));
    embedding_type type = ai->options.embedding.type;
    int embedding_size = (int)embedding_type_to_size(type) * dimension;
    void *embedding = (void *)sqlite3_malloc64(embedding_size);
    if (!embedding) {
        sqlite_context_result_error(context, SQLITE_NOMEM, "Out of memory: failed to allocate embedding buffer of size %d", embedding_size);
        return;
    }
    
    // get token count (negative return encodes needed size)
    int32_t n_tokens = -llama_tokenize(vocab, text, text_len, NULL, 0, true, true);
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
    
    // even with chunking, decoder embeddings need the full sequence to be in the KV once
    if (n_tokens > n_ctx) {
        sqlite3_free(embedding);
        sqlite_context_result_error(context, SQLITE_TOOBIG, "Input too large for model context: %d tokens > n_ctx %d. Create a context with a n_ctx value higher than %d.", n_tokens, n_ctx, n_tokens);
        return;
    }
    
    // allocate tokens and tokenize
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
    
    // max batch size
    uint32_t n_batch = llama_n_batch(ctx);
    
    size_t pos_base = 0; // running position across chunks
    llama_seq_id sequence_id = 0;
    llama_memory_t memory = llama_get_memory(ctx);
    
    if (memory) {
        // start from a clean slate for this sequence
        llama_memory_seq_rm(memory, sequence_id, 0, -1);
        
        // fresh KV for this prompt (only once!)
        llama_memory_clear(memory, /*clear_kv_cache_only=*/true);
    }
    
    // LLAMA_POOLING_TYPE_NONE is disabled in this version
    const enum llama_pooling_type pooling_type = llama_pooling_type(ctx);
    GGML_ASSERT(pooling_type != LLAMA_POOLING_TYPE_NONE);
    
    // init batch: n_seq_max = 1 (single prompt), embd = 0
    llama_batch batch = llama_batch_init(n_batch, 0, 1);
    while (pos_base < n_tokens) {
        llm_batch_clear(&batch);
        
        size_t to_feed = (n_tokens - pos_base > n_batch) ? n_batch : (n_tokens - pos_base);
        
        // fill the batch with up to n_batch tokens
        for (size_t i = 0; i < to_feed; ++i) {
            const llama_token tk = (llama_token)tokens[pos_base + i];
            const llama_pos   ps = (llama_pos)(pos_base + i);
            const bool want_logits = (i + 1 == to_feed); // last token in this chunk
            llm_batch_add(&batch, tk, ps, &sequence_id, 1, want_logits);
        }
        
        // run model on this chunk
        // from ggerganov: If your application is going to support both models with and without a memory, then you should simply call llama_decode() always
        // https://github.com/ggml-org/llama.cpp/discussions/14454
        int32_t rc = (memory) ? llama_decode(ctx, batch) : llama_encode(ctx, batch);
        if (rc < 0) {
            sqlite3_free(tokens);
            sqlite3_free(embedding);
            llama_batch_free(batch);
            sqlite_context_result_error(context, SQLITE_ERROR, "Model %s failed during embedding generation (%d)", (memory) ? "decode" : "encode", rc);
            return;
        }
        
        pos_base += to_feed;
    }
    
    // retrieve sentence embedding (pooling is enabled)
    const float *result = llama_get_embeddings_seq(ctx, sequence_id);
    if (result == NULL) {
        sqlite3_free(tokens);
        sqlite3_free(embedding);
        llama_batch_free(batch);
        sqlite_context_result_error(context, SQLITE_ERROR, "Failed to retrieve embedding vector from model");
        return;
    }
    
    // llm_embed_debug((const void *)result, EMBEDDING_TYPE_F32, dimension);
    
    // check if normalization is needed (default true)
    if (ai->options.embedding.normalize) {
        llm_embed_normalize(result, embedding, type, dimension);
    } else {
        // copy buffer
        llm_embed_copy(result, embedding, type, dimension, embedding_size);
    }
    
    // IMPORTANT: clear memory for this sequence so the next call starts clean
    if (memory) {
        // remove tokens in this sequence and optionally compact
        llama_memory_seq_rm(memory, sequence_id, 0, -1);
        llama_memory_clear(memory, true);
    }
    
    // llm_embed_debug(embedding, type, dimension);
    
    // check if JSON output is set
    if (ai->options.embedding.json_output) {
        sqlite3_str *s = sqlite3_str_new(sqlite3_context_db_handle(context));
        sqlite3_str_appendchar(s, 1, '[');
        for (int i = 0; i < dimension; i++) {
            if (i) sqlite3_str_appendchar(s, 1, ',');
            float value = 0.0;
            
            switch (type) {
                case EMBEDDING_TYPE_F32:
                    value = ((float *)embedding)[i];
                    break;
                    
                case EMBEDDING_TYPE_F16:
                    value = float16_to_float32(((uint16_t *)embedding)[i]);
                    break;
                
                case EMBEDDING_TYPE_BF16:
                    value = bfloat16_to_float32(((uint16_t *)embedding)[i]);
                    break;
                
                case EMBEDDING_TYPE_U8:
                    value = (float)(((uint8_t *)embedding)[i]);
                    break;
                
                case EMBEDDING_TYPE_I8:
                    value = (float)(((int8_t *)embedding)[i]);
                    break;
            }
            sqlite3_str_appendf(s, "%.6g", value);
        }
        sqlite3_str_appendchar(s, 1, ']');
        
        char *json = sqlite3_str_finish(s);
        (json) ? sqlite3_result_text(context, json, -1, sqlite3_free) : sqlite3_result_null(context);
        sqlite3_free(embedding);
    } else {
        sqlite3_result_blob(context, embedding, embedding_size, sqlite3_free);
    }
    
    sqlite3_free(tokens);
    llama_batch_free(batch);
}

static void llm_embed_generate (sqlite3_context *context, int argc, sqlite3_value **argv) {
    if (llm_check_context(context) == false) return;
    if (llm_common_args_check(context, "llm_embed_generate", argc, argv, true) == false) return;
    
    const char *text = (const char *)sqlite3_value_text(argv[0]);
    int32_t text_len = (int32_t)sqlite3_value_bytes(argv[0]);
    const char *model_options = (argc == 2) ? (const char *)sqlite3_value_text(argv[1]) : NULL;
    
    // handle NULL input
    if (!text || text_len == 0) {
        sqlite3_result_null(context);
        return;
    }
        
    // passing NULL as xdata because context has been already created
    ai_context *ai = (ai_context *)sqlite3_user_data(context);
    if (parse_keyvalue_string(ai, model_options, llm_context_options_callback, NULL) == false) return;
    
    // real processing
    llm_embed_generate_run(context, text, text_len);
}

static void llm_token_count (sqlite3_context *context, int argc, sqlite3_value **argv) {
    // sanity check args and context
    if (llm_check_context(context) == false) return;
    if (llm_common_args_check(context, "llm_token_count", argc, argv, true) == false) return;
    
    const char *text = (const char *)sqlite3_value_text(argv[0]);
    int32_t text_len = (int32_t)sqlite3_value_bytes(argv[0]);
    if (!text || text_len == 0) return;
    
    // sanity check vocab
    ai_context *ai = (ai_context *)sqlite3_user_data(context);
    const struct llama_vocab *vocab = llama_model_get_vocab(ai->model);
    if (!vocab) {
        sqlite_context_result_error(context, SQLITE_ERROR, "Failed to extract vocabulary from the model");
        return;
    }
    
    int32_t n_tokens = -llama_tokenize(vocab, text, text_len, NULL, 0, true, false);
    sqlite3_result_int64(context, n_tokens);
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
    int n_predict = (ai->options.n_predict > 0) ? ai->options.n_predict : NPREDICT_DEFAULT_VALUE;
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
    if (llm_check_context(context) == false) return;
    if (llm_common_args_check(context, "llm_text_generate", argc, argv, true) == false) return;
    
    const char *text = (const char *)sqlite3_value_text(argv[0]);
    int32_t text_len = (int32_t)sqlite3_value_bytes(argv[0]);
    const char *options = (argc == 2) ? (const char *)sqlite3_value_text(argv[1]) : NULL;
    
    // passing NULL as xdata because context has been already created
    ai_context *ai = (ai_context *)sqlite3_user_data(context);
    if (parse_keyvalue_string(ai, options, llm_context_options_callback, NULL) == false) {
        sqlite_context_result_error(context, SQLITE_ERROR, "An error occurred while parsing options (%s)", options);
        return;
    }
        
    if (!text || text_len == 0) return;
    llm_text_run(context, text, text_len);
}

// MARK: - Chat -

static bool llm_chat_check_context (ai_context *ai) {
    if (!ai || !ai->ctx) {
        sqlite_common_set_error(ai ? ai->context : NULL, ai ? ai->vtab : NULL, SQLITE_MISUSE, "No context found. Please call llm_context_create() before llm_chat_create().");
        return false;
    }
    
    // check sampler
    if (!ai->sampler) {
        llm_sampler_check(ai);
        if (ai->sampler == NULL) return false;
        llama_sampler_chain_add(ai->sampler, llama_sampler_init_min_p(0.05, 1));
        llama_sampler_chain_add(ai->sampler, llama_sampler_init_temp(0.8));
        llama_sampler_chain_add(ai->sampler, llama_sampler_init_dist((uint32_t)LLAMA_DEFAULT_SEED));
    }
    
    // initialize the chat struct if already created
    if (ai->chat.uuid[0] != '\0') return true;
    
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
    
    if (!llm_messages_append(messages, ROLE_ASSISTANT, response)) {
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
    if (!llm_messages_append(messages, ROLE_USER, user_prompt)) {
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
    if (llm_check_context(context) == false) return;
    
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
    if (sqlite_sanity_function(context, "llm_chat_restore", argc, argv, 1, types, false, false) == false) return;
    
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
        
        if (!llm_messages_append(messages, role, content)) {
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
    if (llm_check_context(context) == false) return;
    
    int types[] = {SQLITE_TEXT};
    if (sqlite_sanity_function(context, "llm_chat_respond", argc, argv, 1, types, true, false) == false) return;
    
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
        if (sqlite_sanity_function(context, "llm_sampler_init_dist", argc, argv, 1, types, true, false) == false) return;
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
    if (sqlite_sanity_function(context, "llm_sampler_init_top_k", argc, argv, 1, types, true, false) == false) return;
    
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
    if (sqlite_sanity_function(context, "llm_sampler_init_top_p", argc, argv, 2, types, true, false) == false) return;
    
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
    if (sqlite_sanity_function(context, "llm_sampler_init_min_p", argc, argv, 2, types, true, false) == false) return;
    
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
    if (sqlite_sanity_function(context, "llm_sampler_init_typical", argc, argv, 2, types, true, false) == false) return;
    
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
    if (sqlite_sanity_function(context, "llm_sampler_init_temp", argc, argv, 1, types, true, false) == false) return;
    
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
    if (sqlite_sanity_function(context, "llm_sampler_init_temp_ext", argc, argv, 3, types, true, false) == false) return;
    
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
    if (sqlite_sanity_function(context, "llm_sampler_init_xtc", argc, argv, 4, types, true, false) == false) return;
    
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
    if (sqlite_sanity_function(context, "llm_sampler_init_top_n_sigma", argc, argv, 1, types, true, false) == false) return;
    
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
    if (sqlite_sanity_function(context, "llm_sampler_init_mirostat", argc, argv, 4, types, true, false) == false) return;
    
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
    if (sqlite_sanity_function(context, "llm_sampler_init_mirostat_v2", argc, argv, 3, types, true, false) == false) return;
    
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
    if (sqlite_sanity_function(context, "llm_sampler_init_grammar", argc, argv, 2, types, true, false) == false) return;
    
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
    if (sqlite_sanity_function(context, "llm_sampler_init_penalties", argc, argv, 4, types, true, false) == false) return;
    
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

static void llm_lora_free (sqlite3_context *context, int argc, sqlite3_value **argv) {
    ai_context *ai = (ai_context *)sqlite3_user_data(context);
    llama_clear_adapter_lora(ai->ctx);
    for (int i=0; i<MAX_LORAS; ++i) {
        if (ai->lora[i]) {
            llama_adapter_lora_free(ai->lora[i]);
        }
    }
}

static void llm_lora_load (sqlite3_context *context, int argc, sqlite3_value **argv) {
    if (llm_check_context(context) == false) return;
    
    // sanity check arguments
    int types[] = {SQLITE_TEXT, SQLITE_FLOAT};
    if (sqlite_sanity_function(context, "llm_lora_load", argc, argv, 2, types, true, false) == false) return;
    
    const char *lora_path = (const char *)sqlite3_value_text(argv[0]);
    float scale = (float)sqlite3_value_double(argv[1]);
    
    ai_context *ai = (ai_context *)sqlite3_user_data(context);
    struct llama_adapter_lora *lora = llama_adapter_lora_init(ai->model, lora_path);
    if (!lora) {
        sqlite_context_result_error(context, SQLITE_ERROR, "Unable to load LoRA model from file %s", lora_path);
        return;
    }
    
    int index = llm_lora_push(ai, lora, scale);
    if (index == -1) {
        sqlite_context_result_error(context, SQLITE_ERROR, "Unable to save LoRA model (%d maximum allowed models reached)", MAX_LORAS);
        return;
    }
    
    llama_clear_adapter_lora(ai->ctx);
    for (int i=0; i<MAX_LORAS; ++i) {
        if (ai->lora[i] && ai->lora_scale[i] != 0.0) {
            llama_set_adapter_lora(ai->ctx, ai->lora[i], ai->lora_scale[i]);
        }
    }
    
    sqlite3_result_int(context, index);
}

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

static bool llm_context_create_with_options (sqlite3_context *context, ai_context *ai, const char *options1, const char *options2) {
    struct llama_context_params ctx_params = llama_context_default_params();
    if (parse_keyvalue_string(ai, options1, llm_context_options_callback, &ctx_params) == false) {
        sqlite_context_result_error(context, SQLITE_ERROR, "An error occurred while parsing options (%s)", options1);
        return false;
    }
    
    if (options2) {
        if (parse_keyvalue_string(ai, options2, llm_context_options_callback, &ctx_params) == false) {
            sqlite_context_result_error(context, SQLITE_ERROR, "An error occurred while parsing options (%s)", options2);
            return false;
        }
    }
    
    // sanity check embedding_type
    if (ctx_params.embeddings && ai->options.embedding.type == 0) {
        sqlite_context_result_error(context, SQLITE_ERROR, "Embedding type (embedding_type) must be specified in the create context funtion");
        return false;
    }
    
    struct llama_context *ctx = llama_init_from_model(ai->model, ctx_params);
    if (!ctx) {
        sqlite_common_set_error(ai->context, ai->vtab, SQLITE_ERROR, "Unable to create context from model");
        return false;
    }
    
    if (ai->ctx) llm_context_free(context, 0, NULL);
    ai->ctx = ctx;
    
    return true;
}

static void llm_context_create (sqlite3_context *context, int argc, sqlite3_value **argv) {
    // sanity check arguments
    if (llm_common_args_check(context, "llm_context_create", argc, argv, true) == false) return;
    const char *options = (const char *)sqlite3_value_text(argv[0]);
    if ((options == NULL) || (strlen(options) == 0)) {
        sqlite_context_result_error(context, SQLITE_ERROR, "Non empty options must be specified when calling llm_context_create");
        return;
    }
        
    ai_context *ai = (ai_context *)sqlite3_user_data(context);
    llm_context_create_with_options(context, ai, options, NULL);
}

static void llm_context_usage (sqlite3_context *context, int argc, sqlite3_value **argv) {
    ai_context *ai = (ai_context *)sqlite3_user_data(context);
    if (!ai->ctx) {
        sqlite_context_result_error(context, SQLITE_MISUSE, "No context found. Please call llm_context_create() before using this function.");
        return;
    }
    uint32_t n_ctx = llama_n_ctx(ai->ctx);
    int32_t n_ctx_used = llama_memory_seq_pos_max(llama_get_memory(ai->ctx), 0) + 1;
    if (n_ctx_used < 0) n_ctx_used = 0;
    double usage = (n_ctx == 0) ? 0.0 : ((double)(n_ctx_used) / (double)n_ctx);
    char buffer[256];
    int len = snprintf(buffer, sizeof(buffer),
                       "{\"context_size\":%u,\"tokens_used\":%d,\"usage\":%.6f}",
                       n_ctx,
                       n_ctx_used,
                       usage);
    if (len < 0 || len >= (int)sizeof(buffer)) {
        sqlite_context_result_error(context, SQLITE_ERROR, "Failed to format context usage");
        return;
    }
    sqlite3_result_text(context, buffer, len, SQLITE_TRANSIENT);
}

static void llm_context_create_embedding (sqlite3_context *context, int argc, sqlite3_value **argv) {
    const char *options = AI_DEFAULT_CONTEXT_EMBEDDING_OPTIONS;
    const char *options2 = (argc > 0) ? (const char *)sqlite3_value_text(argv[0]) : NULL;
    ai_context *ai = (ai_context *)sqlite3_user_data(context);
    llm_context_create_with_options(context, ai, options, options2);
}

static void llm_context_create_chat (sqlite3_context *context, int argc, sqlite3_value **argv) {
    const char *options = AI_DEFAULT_CONTEXT_CHAT_OPTIONS;
    const char *options2 = (argc > 0) ? (const char *)sqlite3_value_text(argv[0]) : NULL;
    ai_context *ai = (ai_context *)sqlite3_user_data(context);
    llm_context_create_with_options(context, ai, options, options2);
}

static void llm_context_create_textgen (sqlite3_context *context, int argc, sqlite3_value **argv) {
    const char *options = AI_DEFAULT_CONTEXT_TEXTGEN_OPTIONS;
    const char *options2 = (argc > 0) ? (const char *)sqlite3_value_text(argv[0]) : NULL;
    ai_context *ai = (ai_context *)sqlite3_user_data(context);
    llm_context_create_with_options(context, ai, options, options2);
}

static void llm_model_free (sqlite3_context *context, int argc, sqlite3_value **argv) {
    ai_context *ai = (ai_context *)sqlite3_user_data(context);
    ai_cleanup((void *)ai, true, false);
}

static void llm_model_load (sqlite3_context *context, int argc, sqlite3_value **argv) {
    // sanity check arguments
    if (llm_common_args_check(context, "llm_model_load", argc, argv, false) == false) return;
    
    const char *model_path = (const char *)sqlite3_value_text(argv[0]);
    const char *model_options = (argc == 2) ? (const char *)sqlite3_value_text(argv[1]) : NULL;
    if (model_options == NULL) model_options = AI_DEFAULT_MODEL_OPTIONS;
    
    ai_context *ai = (ai_context *)sqlite3_user_data(context);
    struct llama_model_params model_params = llama_model_default_params();
    if (parse_keyvalue_string(ai, model_options, llm_model_options_callback, &model_params) == false) {
        sqlite_context_result_error(context, SQLITE_ERROR, "An error occurred while parsing options (%s)", model_options);
        return;
    }
    
    struct llama_model *model = llama_model_load_from_file(model_path, model_params);
    if (!model) {
        sqlite_context_result_error(context, SQLITE_ERROR, "Unable to load model from file %s", model_path);
        return;
    }
    
    ai_cleanup((void *)ai, true, false);
    ai->model = model;
}

// MARK: - LLM Model -

static void llm_model_get_setting (sqlite3_context *context, int argc, sqlite3_value **argv, ai_model_setting setting) {
    // sanity check model
    if (ai_model_check(context, true, false) == false) {
        sqlite_context_result_error(context, SQLITE_MISUSE, "No model is currently set. Please call llm_model_load() before using this function.");
        return;
    }
    
    ai_context *ai = (ai_context *)sqlite3_user_data(context);
    uint64_t n = 0;
    
    switch (setting) {
        case AI_MODEL_SETTING_N_PARAMS: n = llama_model_n_params(ai->model); break;
        case AI_MODEL_SIZE: n = llama_model_size(ai->model); break;
        case AI_MODEL_N_CTX_TRAIN: n = llama_model_n_ctx_train(ai->model); break;
        case AI_MODEL_N_EMBED: n = llama_model_n_embd(ai->model); break;
        case AI_MODEL_N_LAYER: n = llama_model_n_layer(ai->model); break;
        case AI_MODEL_N_HEAD: n = llama_model_n_head(ai->model); break;
        case AI_MODEL_N_HEAD_KV: n = llama_model_n_head_kv(ai->model); break;
        case AI_MODEL_N_SWA: n = llama_model_n_swa(ai->model); break;
        case AI_MODEL_N_CLS_OUT: n = llama_model_n_cls_out(ai->model); break;
        case AI_MODEL_HAS_ENCODER: n = llama_model_has_encoder(ai->model); break;
        case AI_MODEL_HAS_DECODER: n = llama_model_has_decoder(ai->model); break;
        case AI_MODEL_IS_RECURRENT: n = llama_model_is_recurrent(ai->model); break;
            
        case AI_MODEL_CHAT_TEMPLATE: {
            const char *template = llama_model_chat_template(ai->model, NULL);
            sqlite3_result_text(context, template, -1, SQLITE_STATIC);
            return;
        }
            
        case AI_MODEL_FREQ_SCALE_TRAIN: {
            float n = llama_model_rope_freq_scale_train(ai->model);
            sqlite3_result_double(context, n);
            return;
        }
            
    }
    sqlite3_result_int64(context, (sqlite3_int64)n);
}

static void llm_model_n_params (sqlite3_context *context, int argc, sqlite3_value **argv) {
    // Returns the total number of parameters in the model
    llm_model_get_setting(context, argc, argv, AI_MODEL_SETTING_N_PARAMS);
}

static void llm_model_size (sqlite3_context *context, int argc, sqlite3_value **argv) {
    // Returns the total size of all the tensors in the model in bytes
    llm_model_get_setting(context, argc, argv, AI_MODEL_SIZE);
}

static void llm_model_n_ctx_train (sqlite3_context *context, int argc, sqlite3_value **argv) {
    llm_model_get_setting(context, argc, argv, AI_MODEL_N_CTX_TRAIN);
}

static void llm_model_n_embd (sqlite3_context *context, int argc, sqlite3_value **argv) {
    llm_model_get_setting(context, argc, argv, AI_MODEL_N_EMBED);
}

static void llm_model_n_layer (sqlite3_context *context, int argc, sqlite3_value **argv) {
    // Returns the total size of all the tensors in the model in bytes
    llm_model_get_setting(context, argc, argv, AI_MODEL_N_LAYER);
}

static void llm_model_n_head (sqlite3_context *context, int argc, sqlite3_value **argv) {
    // Returns the total size of all the tensors in the model in bytes
    llm_model_get_setting(context, argc, argv, AI_MODEL_N_HEAD);
}

static void llm_model_n_head_kv (sqlite3_context *context, int argc, sqlite3_value **argv) {
    // Returns the total size of all the tensors in the model in bytes
    llm_model_get_setting(context, argc, argv, AI_MODEL_N_HEAD_KV);
}

static void llm_model_n_swa (sqlite3_context *context, int argc, sqlite3_value **argv) {
    // Returns the total size of all the tensors in the model in bytes
    llm_model_get_setting(context, argc, argv, AI_MODEL_N_SWA);
}

static void llm_model_rope_freq_scale_train (sqlite3_context *context, int argc, sqlite3_value **argv) {
    // Get the model's RoPE frequency scaling factor
    llm_model_get_setting(context, argc, argv, AI_MODEL_FREQ_SCALE_TRAIN);
}

static void llm_model_n_cls_out (sqlite3_context *context, int argc, sqlite3_value **argv) {
    // Returns the number of classifier outputs (only valid for classifier models)
    // Undefined behavior for non-classifier models
    llm_model_get_setting(context, argc, argv, AI_MODEL_N_CLS_OUT);
}

static void llm_model_cls_label (sqlite3_context *context, int argc, sqlite3_value **argv) {
    // Returns label of classifier output by index (<n_cls_out). Returns nullptr if no label provided
    
    int types[] = {SQLITE_INTEGER};
    if (sqlite_sanity_function(context, "llm_model_cls_label", argc, argv, 1, types, true, false) == false) return;
    
    int i = sqlite3_value_int(argv[0]);
    ai_context *ai = (ai_context *)sqlite3_user_data(context);
    const char *label = llama_model_cls_label(ai->model, i);
    
    sqlite3_result_text(context, label, -1, SQLITE_STATIC);
}

static void llm_model_desc (sqlite3_context *context, int argc, sqlite3_value **argv) {
    // Get a string describing the model type
    
    if (ai_model_check(context, true, false) == false) {
        sqlite_context_result_error(context, SQLITE_MISUSE, "No model is currently set. Please call llm_model_load() before using this function.");
        return;
    }
    
    char buffer[4096];
    ai_context *ai = (ai_context *)sqlite3_user_data(context);
    int32_t n = llama_model_desc(ai->model, buffer, sizeof(buffer));
    (n > 0) ? sqlite3_result_text(context, buffer, n, SQLITE_TRANSIENT) : sqlite3_result_null(context);
}

static void llm_model_has_encoder (sqlite3_context *context, int argc, sqlite3_value **argv) {
    // Returns true if the model contains an encoder that requires llama_encode() call
    llm_model_get_setting(context, argc, argv, AI_MODEL_HAS_ENCODER);
}

static void llm_model_has_decoder (sqlite3_context *context, int argc, sqlite3_value **argv) {
    // Returns true if the model contains a decoder that requires llama_decode() call
    llm_model_get_setting(context, argc, argv, AI_MODEL_HAS_DECODER);
}

static void llm_model_is_recurrent (sqlite3_context *context, int argc, sqlite3_value **argv) {
    // Returns true if the model is recurrent (like Mamba, RWKV, etc.)
    llm_model_get_setting(context, argc, argv, AI_MODEL_IS_RECURRENT);
}

static void llm_model_chat_template (sqlite3_context *context, int argc, sqlite3_value **argv) {
    // Get the default chat template. Returns nullptr if not available
    // If name is NULL, returns the default chat template
    llm_model_get_setting(context, argc, argv, AI_MODEL_CHAT_TEMPLATE);
}

// MARK: - Audio -

static bool audio_process_check_arguments (sqlite3_context *context, const char *function_name, int argc, sqlite3_value **argv, bool check_audio_model) {
    // sanity check arguments
    if (argc == 1) {
        int type = sqlite3_value_type(argv[0]);
        if (type != SQLITE_TEXT && type != SQLITE_BLOB) {
            return sqlite_context_result_error(context, SQLITE_ERROR, "Function '%s' expects the first argument to be TEXT or BLOB.", function_name);
        }
        
        int types[] = {type};
        return sqlite_sanity_function(context, function_name, argc, argv, 1, types, false, check_audio_model);
    } else if (argc == 2) {
        int type = sqlite3_value_type(argv[0]);
        if (type != SQLITE_TEXT && type != SQLITE_BLOB) {
            return sqlite_context_result_error(context, SQLITE_ERROR, "Function '%s' expects the first argument to be TEXT or BLOB.", function_name);
        }
        
        int types[] = {type, SQLITE_TEXT};
        return sqlite_sanity_function(context, function_name, argc, argv, 2, types, false, check_audio_model);
    }
    
    return sqlite_context_result_error(context, SQLITE_ERROR, "Function '%s' expects 1 or 2 arguments, but %d were provided.", function_name, argc);
}

static void audio_process_run (sqlite3_context *context, const float *buffer, uint64_t num_samples, uint32_t sample_rate, uint16_t channels, const char *options) {
    struct whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    ai_context *ai = (ai_context *)sqlite3_user_data(context);
    if (parse_keyvalue_string(ai, options, whisper_full_params_options_callback, &params) == false) {
        sqlite_context_result_error(context, SQLITE_ERROR, "An error occurred while parsing options (%s)", options);
        return;
    }
    
    // int rc = whisper_full(ai->whisper, params, buffer, (int)num_samples);
}

static void audio_process_flac (sqlite3_context *context, int argc, sqlite3_value **argv) {
    if (audio_process_check_arguments(context, "audio_process_flac", argc, argv, true) == false) return;
    
    float *buffer = NULL;
    uint64_t num_samples = 0;
    uint32_t sample_rate = 0;
    uint16_t channels = 0;
    
    if (sqlite3_value_type(argv[0]) == SQLITE_TEXT) {
        const char *path = (const char *)sqlite3_value_text(argv[0]);
        buffer = audio_flac_file2pcm(path, &num_samples, &sample_rate, &channels);
        if (!buffer) {
            sqlite_context_result_error(context, SQLITE_ERROR, "Unable to convert FLAC file %s to PCM.", path);
            return;
        }
    } else {
        const void *data = sqlite3_value_blob(argv[0]);
        size_t data_size = (size_t)sqlite3_value_bytes(argv[0]);
        buffer = audio_flac_mem2pcm(data, data_size, &num_samples, &sample_rate, &channels);
        if (!buffer) {
            sqlite_context_result_error(context, SQLITE_ERROR, "Unable to convert FLAC blob to PCM.");
            return;
        }
    }
    
    const char *options = (argc >= 2) ? (const char *)sqlite3_value_text(argv[1]) : NULL;
    audio_process_run(context, buffer, num_samples, sample_rate, channels, options);
    
}

static void audio_process_mp3 (sqlite3_context *context, int argc, sqlite3_value **argv) {
    if (audio_process_check_arguments(context, "audio_process_mp3", argc, argv, true) == false) return;
}

static void audio_process_wav (sqlite3_context *context, int argc, sqlite3_value **argv) {
    if (audio_process_check_arguments(context, "audio_process_wav", argc, argv, true) == false) return;
}

static void audio_process (sqlite3_context *context, int argc, sqlite3_value **argv) {
    if (audio_process_check_arguments(context, "audio_process", argc, argv, true) == false) return;
}

static void audio_model_load (sqlite3_context *context, int argc, sqlite3_value **argv) {
    // sanity check arguments
    if (llm_common_args_check(context, "audio_model_load", argc, argv, false) == false) return;
    
    const char *model_path = (const char *)sqlite3_value_text(argv[0]);
    const char *model_options = (argc == 2) ? (const char *)sqlite3_value_text(argv[1]) : NULL;
    
    ai_context *ai = (ai_context *)sqlite3_user_data(context);
    struct whisper_context_params ctx_params = whisper_context_default_params();
    if (parse_keyvalue_string(ai, model_options, whisper_model_options_callback, &ctx_params) == false) {
        sqlite_context_result_error(context, SQLITE_ERROR, "An error occurred while parsing options (%s)", model_options);
        return;
    }
    
    struct whisper_context *whisper = whisper_init_from_file_with_params(model_path, ctx_params);
    if (!whisper) {
        sqlite_context_result_error(context, SQLITE_ERROR, "Unable to load audio model from file %s", model_path);
        return;
    }
    
    ai_cleanup((void *)ai, false, true);
    ai->whisper = whisper;
}

static void audio_model_free (sqlite3_context *context, int argc, sqlite3_value **argv) {
    ai_context *ai = (ai_context *)sqlite3_user_data(context);
    ai_cleanup((void *)ai, false, true);
}

// MARK: - AI -

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
    rc = sqlite3_create_function_v2(db, "ai_version", 0, SQLITE_UTF8, ctx, ai_version, NULL, NULL, ai_destroy);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function_v2(db, "ai_log_info", 1, SQLITE_UTF8, ctx, ai_log_info, NULL, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    // LLAMA
    rc = sqlite3_create_function(db, "llm_model_load", 1, SQLITE_UTF8, ctx, llm_model_load, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function(db, "llm_model_load", 2, SQLITE_UTF8, ctx, llm_model_load, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function(db, "llm_model_free", 0, SQLITE_UTF8, ctx, llm_model_free, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function(db, "llm_context_create", 1, SQLITE_UTF8, ctx, llm_context_create, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function(db, "llm_context_usage", 0, SQLITE_UTF8, ctx, llm_context_usage, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function(db, "llm_context_create_embedding", 0, SQLITE_UTF8, ctx, llm_context_create_embedding, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function(db, "llm_context_create_embedding", 1, SQLITE_UTF8, ctx, llm_context_create_embedding, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function(db, "llm_context_create_chat", 0, SQLITE_UTF8, ctx, llm_context_create_chat, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function(db, "llm_context_create_chat", 1, SQLITE_UTF8, ctx, llm_context_create_chat, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function(db, "llm_context_create_textgen", 0, SQLITE_UTF8, ctx, llm_context_create_textgen, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function(db, "llm_context_create_textgen", 1, SQLITE_UTF8, ctx, llm_context_create_textgen, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function(db, "llm_context_free", 0, SQLITE_UTF8, ctx, llm_context_free, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function(db, "llm_lora_load", 2, SQLITE_UTF8, ctx, llm_lora_load, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function(db, "llm_lora_free", 0, SQLITE_UTF8, ctx, llm_lora_free, NULL, NULL);
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
    
    rc = sqlite3_create_function(db, "llm_token_count", 1, SQLITE_UTF8, ctx, llm_token_count, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function(db, "llm_text_generate", 1, SQLITE_UTF8, ctx, llm_text_generate, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function(db, "llm_text_generate", 2, SQLITE_UTF8, ctx, llm_text_generate, NULL, NULL);
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
    
    rc = sqlite3_create_module(db, "llm_chat", &llm_chat, ctx);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function(db, "llm_model_n_params", 0, SQLITE_UTF8, ctx, llm_model_n_params, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;

    rc = sqlite3_create_function(db, "llm_model_size", 0, SQLITE_UTF8, ctx, llm_model_size, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;

    rc = sqlite3_create_function(db, "llm_model_n_ctx_train", 0, SQLITE_UTF8, ctx, llm_model_n_ctx_train, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;

    rc = sqlite3_create_function(db, "llm_model_n_embd", 0, SQLITE_UTF8, ctx, llm_model_n_embd, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;

    rc = sqlite3_create_function(db, "llm_model_n_layer", 0, SQLITE_UTF8, ctx, llm_model_n_layer, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;

    rc = sqlite3_create_function(db, "llm_model_n_head", 0, SQLITE_UTF8, ctx, llm_model_n_head, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;

    rc = sqlite3_create_function(db, "llm_model_n_head_kv", 0, SQLITE_UTF8, ctx, llm_model_n_head_kv, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;

    rc = sqlite3_create_function(db, "llm_model_n_swa", 0, SQLITE_UTF8, ctx, llm_model_n_swa, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;

    rc = sqlite3_create_function(db, "llm_model_rope_freq_scale_train", 0, SQLITE_UTF8, ctx, llm_model_rope_freq_scale_train, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;

    rc = sqlite3_create_function(db, "llm_model_n_cls_out", 0, SQLITE_UTF8, ctx, llm_model_n_cls_out, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;

    rc = sqlite3_create_function(db, "llm_model_cls_label", 0, SQLITE_UTF8, ctx, llm_model_cls_label, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;

    rc = sqlite3_create_function(db, "llm_model_desc", 0, SQLITE_UTF8, ctx, llm_model_desc, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;

    rc = sqlite3_create_function(db, "llm_model_has_encoder", 0, SQLITE_UTF8, ctx, llm_model_has_encoder, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;

    rc = sqlite3_create_function(db, "llm_model_has_decoder", 0, SQLITE_UTF8, ctx, llm_model_has_decoder, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;

    rc = sqlite3_create_function(db, "llm_model_is_recurrent", 0, SQLITE_UTF8, ctx, llm_model_is_recurrent, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;

    rc = sqlite3_create_function(db, "llm_model_chat_template", 0, SQLITE_UTF8, ctx, llm_model_chat_template, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    // WHISPER
    /*
    rc = sqlite3_create_function(db, "audio_model_load", 1, SQLITE_UTF8, ctx, audio_model_load, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function(db, "audio_model_load", 2, SQLITE_UTF8, ctx, audio_model_load, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function(db, "audio_model_free", 0, SQLITE_UTF8, ctx, audio_model_free, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    */
     
cleanup:
    return rc;
}
