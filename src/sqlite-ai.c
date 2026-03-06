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

#define MIN_ALLOC_TOKEN                         4096
#define MIN_ALLOC_PROMPT                        4096
#define MIN_ALLOC_RESPONSE                      4096
#define MAX_PATH                                4096
#define MAX_TOKEN_TEXT_LEN                      128     // according to ChatGPT 32 would be safe for all common tokenizers
#define MIN_ALLOC_MESSAGES                      64
#define MAX_LORAS                               64      // max 2 or 3 LoRa adapters are used (usually just one)
#define KEY_MATCHES(k, klen, constant)          ((klen) == (int)strlen(constant) && strncasecmp((k), (constant), (klen)) == 0)

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

// WHISPER OPTIONS
#define OPTION_KEY_LANGUAGE                     "language"
#define OPTION_KEY_TRANSLATE                    "translate"
#define OPTION_KEY_OFFSET_MS                    "offset_ms"
#define OPTION_KEY_DURATION_MS                  "duration_ms"
#define OPTION_KEY_NO_TIMESTAMPS                "no_timestamps"
#define OPTION_KEY_SINGLE_SEGMENT               "single_segment"
#define OPTION_KEY_TOKEN_TIMESTAMPS             "token_timestamps"
#define OPTION_KEY_INITIAL_PROMPT               "initial_prompt"
#define OPTION_KEY_TEMPERATURE_W                "temperature"
#define OPTION_KEY_BEAM_SIZE                    "beam_size"
#define OPTION_KEY_AUDIO_CTX                    "audio_ctx"
#define OPTION_KEY_SUPPRESS_REGEX               "suppress_regex"
#define OPTION_KEY_MAX_LEN                      "max_len"
#define OPTION_KEY_PRINT_TIMESTAMPS             "print_timestamps"

#define AI_COLUMN_REPLY                         0

#define AI_DEFAULT_MODEL_OPTIONS                "gpu_layers=99"
#define AI_DEFAULT_CONTEXT_EMBEDDING_OPTIONS    "generate_embedding=1,normalize_embedding=1,pooling_type=mean"
#define AI_DEFAULT_CONTEXT_CHAT_OPTIONS         ""
#define AI_DEFAULT_CONTEXT_TEXTGEN_OPTIONS      ""

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

const char *ROLE_SYSTEM    = "system";
const char *ROLE_USER       = "user";
const char *ROLE_ASSISTANT  = "assistant";

// normalize a role string to its static pointer (avoids duplicating known roles)
static const char *role_normalize (const char *role) {
    if (!role) return NULL;
    if (role == ROLE_SYSTEM || strcmp(role, ROLE_SYSTEM) == 0) return ROLE_SYSTEM;
    if (role == ROLE_USER || strcmp(role, ROLE_USER) == 0) return ROLE_USER;
    if (role == ROLE_ASSISTANT || strcmp(role, ROLE_ASSISTANT) == 0) return ROLE_ASSISTANT;
    return NULL; // unknown role, caller must duplicate
}

static bool role_is_static (const char *role) {
    return (role == ROLE_SYSTEM || role == ROLE_USER || role == ROLE_ASSISTANT);
}

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
    if (KEY_MATCHES(key, key_len, OPTION_KEY_GPU_LAYERS)) {
        int value = (int)strtol(buffer, NULL, 0);
        options->n_gpu_layers = value;
        return true;
    }
    
    if (KEY_MATCHES(key, key_len, OPTION_KEY_MAIN_GPU)) {
        int value = (int)strtol(buffer, NULL, 0);
        options->main_gpu = value;
        return true;
    }
    
    if (KEY_MATCHES(key, key_len, OPTION_KEY_SPLIT_MODE)) {
        int value = (int)strtol(buffer, NULL, 0);
        if (value >= 0 && value <= 2) options->split_mode = value;
        return true;
    }
    
    if (KEY_MATCHES(key, key_len, OPTION_KEY_VOCAB_ONLY)) {
        int value = (int)strtol(buffer, NULL, 0);
        options->vocab_only = (value != 0);
        return true;
    }
    
    if (KEY_MATCHES(key, key_len, OPTION_KEY_USE_MMAP)) {
        int value = (int)strtol(buffer, NULL, 0);
        options->use_mmap = (value != 0);
        return true;
    }
    
    if (KEY_MATCHES(key, key_len, OPTION_KEY_USE_MLOCK)) {
        int value = (int)strtol(buffer, NULL, 0);
        options->use_mlock = (value != 0);
        return true;
    }
    
    if (KEY_MATCHES(key, key_len, OPTION_KEY_CHECK_TENSORS)) {
        int value = (int)strtol(buffer, NULL, 0);
        options->check_tensors = (value != 0);
        return true;
    }
    
    if (KEY_MATCHES(key, key_len, OPTION_KEY_LOG_INFO)) {
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
    if (KEY_MATCHES(key, key_len, OPTION_KEY_NORMALIZE_EMBEDDING)) {
        int value = (int)strtol(buffer, NULL, 0);
        ai->options.embedding.normalize = (value != 0);
        return true;
    }
    
    if (KEY_MATCHES(key, key_len, OPTION_KEY_JSON_OUTPUT)) {
        int value = (int)strtol(buffer, NULL, 0);
        ai->options.embedding.json_output = (value != 0);
        return true;
    }
    
    if (KEY_MATCHES(key, key_len, OPTION_KEY_MAX_TOKENS)) {
        int value = (int)strtol(buffer, NULL, 0);
        if (value >= 0) ai->options.max_tokens = value;
        return true;
    }
    
    if (KEY_MATCHES(key, key_len, OPTION_KEY_N_PREDICT)) {
        int value = (int)strtol(buffer, NULL, 0);
        if (value >= 0) ai->options.n_predict = value;
        return true;
    }
    
    if (KEY_MATCHES(key, key_len, OPTION_KEY_EMBEDDING_TYPE)) {
        int value = embedding_name_to_type(buffer);
        if (value > 0) ai->options.embedding.type = value;
        return true;
    }
    
    // CONTEXT OPTIONS
    if (options == NULL) {
        char warn_buf[512];
        snprintf(warn_buf, sizeof(warn_buf), "key %.*s ignored because context was already created", key_len, key);
        ai_logger(GGML_LOG_LEVEL_WARN, warn_buf, ai);
        return true;
    }

    if (KEY_MATCHES(key, key_len, OPTION_KEY_GENERATE_EMBEDDING)) {
        // https://github.com/ggml-org/llama.cpp/discussions/15093
        int value = (int)strtol(buffer, NULL, 0);
        options->embeddings = (value != 0);
        options->pooling_type = LLAMA_POOLING_TYPE_MEAN;
        
        // for non-causal models, batch size must be equal to ubatch size
        // when generating embeddings, always tie them together.
        options->n_ubatch = options->n_batch;
        return true;
    }
    
    if (KEY_MATCHES(key, key_len, OPTION_KEY_CONTEXT_SIZE)) {
        int value = (int)strtol(buffer, NULL, 0);
        if (value >= 0) {
            options->n_ctx = value;
            options->n_batch = value;
        }
        return true;
    }
    
    if (KEY_MATCHES(key, key_len, OPTION_KEY_N_CTX)) {
        int value = (int)strtol(buffer, NULL, 0);
        if (value >= 0) options->n_ctx = value;
        return true;
    }
    
    if (KEY_MATCHES(key, key_len, OPTION_KEY_N_BATCH)) {
        int value = (int)strtol(buffer, NULL, 0);
        if (value >= 0) options->n_batch = value;
        return true;
    }
    
    if (KEY_MATCHES(key, key_len, OPTION_KEY_N_UBATCH)) {
        int value = (int)strtol(buffer, NULL, 0);
        if (value >= 0) options->n_ubatch = value;
        return true;
    }
    
    if (KEY_MATCHES(key, key_len, OPTION_KEY_N_SEQ_MAX)) {
        int value = (int)strtol(buffer, NULL, 0);
        if (value >= 0) options->n_seq_max = value;
        return true;
    }
    
    if (KEY_MATCHES(key, key_len, OPTION_KEY_N_THREADS)) {
        int value = (int)strtol(buffer, NULL, 0);
        if (value >= 0) options->n_threads = value;
        return true;
    }
    
    if (KEY_MATCHES(key, key_len, OPTION_KEY_N_THREADS_BATCH)) {
        int value = (int)strtol(buffer, NULL, 0);
        if (value >= 0) options->n_threads_batch = value;
        return true;
    }
    
    if (KEY_MATCHES(key, key_len, OPTION_KEY_POOLING_TYPE)) {
        // pooling_type mean is not supported and so in this version we forced it to be really mean so ONE EMBEDDING will be generated
        if (strcasecmp(buffer, "none") == 0) options->pooling_type = LLAMA_POOLING_TYPE_MEAN;
        else if (strcasecmp(buffer, "unspecified") == 0) options->pooling_type = LLAMA_POOLING_TYPE_MEAN;
        else if (strcasecmp(buffer, "mean") == 0) options->pooling_type = LLAMA_POOLING_TYPE_MEAN;
        else if (strcasecmp(buffer, "cls") == 0) options->pooling_type = LLAMA_POOLING_TYPE_CLS;
        else if (strcasecmp(buffer, "last") == 0) options->pooling_type = LLAMA_POOLING_TYPE_LAST;
        else if (strcasecmp(buffer, "rank") == 0) options->pooling_type = LLAMA_POOLING_TYPE_RANK;
        return true;
    }
    
    if (KEY_MATCHES(key, key_len, OPTION_KEY_ATTENTION_TYPE)) {
        if (strcasecmp(buffer, "unspecified") == 0) options->attention_type = LLAMA_ATTENTION_TYPE_UNSPECIFIED;
        else if (strcasecmp(buffer, "causal") == 0) options->attention_type = LLAMA_ATTENTION_TYPE_CAUSAL;
        else if (strcasecmp(buffer, "non_causal") == 0) options->attention_type = LLAMA_ATTENTION_TYPE_NON_CAUSAL;
        return true;
    }
    
    if (KEY_MATCHES(key, key_len, OPTION_KEY_ROPE_SCALING_TYPE)) {
        if (strcasecmp(buffer, "unspecified") == 0) options->rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED;
        else if (strcasecmp(buffer, "none") == 0) options->rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_NONE;
        else if (strcasecmp(buffer, "linear") == 0) options->rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_LINEAR;
        else if (strcasecmp(buffer, "yarn") == 0) options->rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_YARN;
        else if (strcasecmp(buffer, "longrope") == 0) options->rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_LONGROPE;
        return true;
    }
    
    if (KEY_MATCHES(key, key_len, OPTION_KEY_FLASH_ATTN_TYPE)) {
        if (strcasecmp(buffer, "auto") == 0) options->flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
        else if (strcasecmp(buffer, "disabled") == 0) options->flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        else if (strcasecmp(buffer, "enabled") == 0) options->flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        return true;
    }
    
    if (KEY_MATCHES(key, key_len, OPTION_KEY_ROPE_FREQ_BASE)) {
        float value = strtof(buffer, NULL);
        options->rope_freq_base = value;
        return true;
    }
    
    if (KEY_MATCHES(key, key_len, OPTION_KEY_ROPE_FREQ_SCALE)) {
        float value = strtof(buffer, NULL);
        options->rope_freq_scale = value;
        return true;
    }
    
    if (KEY_MATCHES(key, key_len, OPTION_KEY_YARN_EXT_FACTOR)) {
        float value = strtof(buffer, NULL);
        options->yarn_ext_factor = value;
        return true;
    }
    
    if (KEY_MATCHES(key, key_len, OPTION_KEY_YARN_ATTN_FACTOR)) {
        float value = strtof(buffer, NULL);
        options->yarn_attn_factor = value;
        return true;
    }
    
    if (KEY_MATCHES(key, key_len, OPTION_KEY_YARN_BETA_FAST)) {
        float value = strtof(buffer, NULL);
        options->yarn_beta_fast = value;
        return true;
    }
    
    if (KEY_MATCHES(key, key_len, OPTION_KEY_YARN_BETA_SLOW)) {
        float value = strtof(buffer, NULL);
        options->yarn_beta_slow = value;
        return true;
    }
    
    if (KEY_MATCHES(key, key_len, OPTION_KEY_DEFRAG_THOLD)) {
        float value = strtof(buffer, NULL);
        options->defrag_thold = value;
        return true;
    }
    
    if (KEY_MATCHES(key, key_len, OPTION_KEY_YARN_ORIG_CTX)) {
        int value = (int)strtol(buffer, NULL, 0);
        if (value >= 0) options->yarn_orig_ctx = value;
        return true;
    }
    
    if (KEY_MATCHES(key, key_len, OPTION_KEY_OFFLOAD_KQV)) {
        int value = (int)strtol(buffer, NULL, 0);
        options->offload_kqv = (value != 0);
        return true;
    }
    
    if (KEY_MATCHES(key, key_len, OPTION_KEY_OP_OFFLOAD)) {
        int value = (int)strtol(buffer, NULL, 0);
        options->op_offload = (value != 0);
        return true;
    }
    
    if (KEY_MATCHES(key, key_len, OPTION_KEY_SWA_FULL)) {
        int value = (int)strtol(buffer, NULL, 0);
        options->swa_full = (value != 0);
        return true;
    }
    
    if (KEY_MATCHES(key, key_len, OPTION_KEY_TYPE_K)) {
        int value = (int)strtol(buffer, NULL, 0);
        if (value >= 0) options->type_k = value;
        return true;
    }
    
    if (KEY_MATCHES(key, key_len, OPTION_KEY_TYPE_V)) {
        int value = (int)strtol(buffer, NULL, 0);
        if (value >= 0) options->type_v = value;
        return true;
    }
    
    if (KEY_MATCHES(key, key_len, OPTION_KEY_TYPE_KV_UNIFIED)) {
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
    struct whisper_full_params *params = (struct whisper_full_params *)xdata;

    // sanity check
    if (!key || key_len == 0) return true;
    if (!value || value_len == 0) return true;

    // convert value to c-string
    char buffer[256] = {0};
    size_t len = (value_len > (int)sizeof(buffer)-1) ? (int)sizeof(buffer)-1 : value_len;
    memcpy(buffer, value, len);

    if (KEY_MATCHES(key, key_len, OPTION_KEY_LANGUAGE)) {
        // language is stored inside params as a pointer; must outlive the whisper_full() call
        if (strcasecmp(buffer, "auto") == 0) params->language = NULL;
        else params->language = sqlite_strdup(buffer);
        return true;
    }

    if (KEY_MATCHES(key, key_len, OPTION_KEY_TRANSLATE)) {
        params->translate = ((int)strtol(buffer, NULL, 0) != 0);
        return true;
    }

    if (KEY_MATCHES(key, key_len, OPTION_KEY_N_THREADS)) {
        int v = (int)strtol(buffer, NULL, 0);
        if (v > 0) params->n_threads = v;
        return true;
    }

    if (KEY_MATCHES(key, key_len, OPTION_KEY_OFFSET_MS)) {
        params->offset_ms = (int)strtol(buffer, NULL, 0);
        return true;
    }

    if (KEY_MATCHES(key, key_len, OPTION_KEY_DURATION_MS)) {
        params->duration_ms = (int)strtol(buffer, NULL, 0);
        return true;
    }

    if (KEY_MATCHES(key, key_len, OPTION_KEY_NO_TIMESTAMPS)) {
        params->no_timestamps = ((int)strtol(buffer, NULL, 0) != 0);
        return true;
    }

    if (KEY_MATCHES(key, key_len, OPTION_KEY_SINGLE_SEGMENT)) {
        params->single_segment = ((int)strtol(buffer, NULL, 0) != 0);
        return true;
    }

    if (KEY_MATCHES(key, key_len, OPTION_KEY_TOKEN_TIMESTAMPS)) {
        params->token_timestamps = ((int)strtol(buffer, NULL, 0) != 0);
        return true;
    }

    if (KEY_MATCHES(key, key_len, OPTION_KEY_INITIAL_PROMPT)) {
        params->initial_prompt = sqlite_strdup(buffer);
        return true;
    }

    if (KEY_MATCHES(key, key_len, OPTION_KEY_TEMPERATURE_W)) {
        params->temperature = strtof(buffer, NULL);
        return true;
    }

    if (KEY_MATCHES(key, key_len, OPTION_KEY_BEAM_SIZE)) {
        int v = (int)strtol(buffer, NULL, 0);
        if (v > 0) {
            params->strategy = WHISPER_SAMPLING_BEAM_SEARCH;
            params->beam_search.beam_size = v;
        }
        return true;
    }

    if (KEY_MATCHES(key, key_len, OPTION_KEY_AUDIO_CTX)) {
        params->audio_ctx = (int)strtol(buffer, NULL, 0);
        return true;
    }

    if (KEY_MATCHES(key, key_len, OPTION_KEY_SUPPRESS_REGEX)) {
        params->suppress_regex = sqlite_strdup(buffer);
        return true;
    }

    if (KEY_MATCHES(key, key_len, OPTION_KEY_MAX_LEN)) {
        params->max_len = (int)strtol(buffer, NULL, 0);
        return true;
    }

    if (KEY_MATCHES(key, key_len, OPTION_KEY_PRINT_TIMESTAMPS)) {
        params->print_timestamps = ((int)strtol(buffer, NULL, 0) != 0);
        return true;
    }

    // ignore unknown keys
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
        if (ai->ctx) llama_set_adapters_lora(ai->ctx, NULL, 0, NULL);
        if (ai->ctx) llama_free(ai->ctx);
        if (ai->model) llama_model_free(ai->model);
        // sampler chain is freed explicitly via llm_sampler_free() or llm_sampler_create() SQL functions;
        // freeing it here causes a double-free crash when ai_destroy runs after explicit cleanup
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
    const char *normalized = role_normalize(role);

    if (normalized == ROLE_SYSTEM && list->count > 0) {
        // only one system prompt allowed at the beginning
        return false;
    }

    bool needs_system_message = (list->count == 0 && normalized != ROLE_SYSTEM);
    size_t required = list->count + (needs_system_message ? 2 : 1);
    if (required > list->capacity) {
        size_t new_cap = list->capacity ? list->capacity * 2 : MIN_ALLOC_MESSAGES;
        if (new_cap < required) new_cap = required;
        llama_chat_message *new_items = sqlite3_realloc64(list->items, new_cap * sizeof(llama_chat_message));
        if (!new_items) return false;

        list->items = new_items;
        list->capacity = new_cap;
    }

    if (needs_system_message) {
        // reserve first item for empty system prompt
        list->items[list->count].role = ROLE_SYSTEM;
        list->items[list->count].content = sqlite_strdup("");
        list->count += 1;
    }

    list->items[list->count].role = normalized ? normalized : sqlite_strdup(role);
    list->items[list->count].content = sqlite_strdup(content);
    list->count += 1;
    return true;
}

bool llm_messages_set (ai_messages *list, int pos, const char *role, const char *content) {
    if (pos < 0 || pos >= (int)list->count)
        return false;

    const char *normalized = role_normalize(role);
    llama_chat_message *message = &list->items[pos];

    if (!role_is_static(message->role))
        sqlite3_free((char *)message->role);
    sqlite3_free((char *)message->content);

    message->role = normalized ? normalized : sqlite_strdup(role);
    message->content = sqlite_strdup(content);
    return true;
}

void llm_messages_free (ai_messages *list) {
    for (size_t i = 0; i < list->count; ++i) {
        if (!role_is_static(list->items[i].role))
            sqlite3_free((char *)list->items[i].role);
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

    // sanity check model (encoder-decoder models are not supported for embeddings)
    if (llama_model_has_encoder(model) && llama_model_has_decoder(model)) {
        sqlite_context_result_error(context, SQLITE_ERROR, "Computing embeddings in encoder-decoder models is not supported");
        return;
    }

    // sanity check vocab
    const struct llama_vocab *vocab = llama_model_get_vocab(model);
    if (!vocab) {
        sqlite_context_result_error(context, SQLITE_ERROR, "Failed to extract vocabulary from the model");
        return;
    }

    // determine model architecture
    bool has_encoder = llama_model_has_encoder(model);
    bool has_decoder = llama_model_has_decoder(model);
    bool is_encoder_only = has_encoder && !has_decoder;

    struct llama_context *ctx = ai->ctx;
    llama_set_embeddings(ctx, true);

    // clamp effective context to model's training window to avoid position embedding overflow
    // also clamp to n_batch since we submit all tokens in a single batch
    const int n_ctx_train = llama_model_n_ctx_train(model);
    const int n_ctx_raw = (int)llama_n_ctx(ctx);
    const int n_batch = (int)llama_n_batch(ctx);
    int n_ctx = (n_ctx_raw > n_ctx_train) ? n_ctx_train : n_ctx_raw;
    if (n_ctx > n_batch) n_ctx = n_batch;

    // pooling type sanity check
    const enum llama_pooling_type pooling_type = llama_pooling_type(ctx);
    if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
        sqlite_context_result_error(context, SQLITE_ERROR, "Embedding generation requires pooling (pooling_type must not be NONE)");
        return;
    }

    // allocate embedding buffer
    int dimension = llama_model_n_embd(model);
    embedding_type type = ai->options.embedding.type;
    int embedding_size = (int)embedding_type_to_size(type) * dimension;
    void *embedding = (void *)sqlite3_malloc64(embedding_size);
    if (!embedding) {
        sqlite_context_result_error(context, SQLITE_NOMEM, "Out of memory: failed to allocate embedding buffer of size %d", embedding_size);
        return;
    }

    // allocate token buffer sized to context limit
    llama_token *tokens = (llama_token *)sqlite3_malloc64(n_ctx * sizeof(llama_token));
    if (!tokens) {
        sqlite3_free(embedding);
        sqlite_context_result_error(context, SQLITE_NOMEM, "Out of memory: failed to allocate tokens buffer");
        return;
    }

    // tokenize directly into the buffer
    int32_t n_tokens = llama_tokenize(vocab, text, text_len, tokens, n_ctx, true, true);
    if (n_tokens < 0) {
        // negative return means input needs more tokens than n_ctx — truncate
        int32_t n_needed = -n_tokens;

        // check user-defined max_tokens limit
        if (ai->options.max_tokens > 0 && n_needed > ai->options.max_tokens) {
            sqlite3_free(tokens);
            sqlite3_free(embedding);
            sqlite_context_result_error(context, SQLITE_TOOBIG, "Input too large: %d tokens exceeds max allowed (%d)", n_needed, ai->options.max_tokens);
            return;
        }

        // allocate a temporary buffer large enough for the full tokenization, then truncate
        llama_token *full_tokens = (llama_token *)sqlite3_malloc64(n_needed * sizeof(llama_token));
        if (!full_tokens) {
            sqlite3_free(tokens);
            sqlite3_free(embedding);
            sqlite_context_result_error(context, SQLITE_NOMEM, "Out of memory: failed to allocate tokens buffer");
            return;
        }
        int32_t n_actual = llama_tokenize(vocab, text, text_len, full_tokens, n_needed, true, true);
        if (n_actual < 0 || n_actual != n_needed) {
            sqlite3_free(full_tokens);
            sqlite3_free(tokens);
            sqlite3_free(embedding);
            sqlite_context_result_error(context, SQLITE_ERROR, "Tokenization failed");
            return;
        }
        // truncate to n_ctx
        memcpy(tokens, full_tokens, n_ctx * sizeof(llama_token));
        sqlite3_free(full_tokens);
        n_tokens = n_ctx;
    }

    if (n_tokens == 0) {
        sqlite3_free(tokens);
        sqlite3_free(embedding);
        sqlite_context_result_error(context, SQLITE_ERROR, "Tokenization produced no tokens");
        return;
    }

    // check user-defined max_tokens limit
    if (ai->options.max_tokens > 0 && n_tokens > ai->options.max_tokens) {
        sqlite3_free(tokens);
        sqlite3_free(embedding);
        sqlite_context_result_error(context, SQLITE_TOOBIG, "Input too large: %d tokens exceeds max allowed (%d)", n_tokens, ai->options.max_tokens);
        return;
    }

    // prepare batch
    llama_seq_id sequence_id = 0;
    llama_memory_t memory = llama_get_memory(ctx);

    if (memory) {
        llama_memory_clear(memory, true);
    }

    struct llama_batch batch = {
        .n_tokens   = n_tokens,
        .token      = tokens,
        .embd       = NULL,
        .pos        = NULL,
        .n_seq_id   = NULL,
        .seq_id     = NULL,
        .logits     = NULL,
    };

    // encode or decode based on model architecture
    // encoder-only models (BERT-style) use llama_encode
    // decoder-only models use llama_decode (which also works for models without memory)
    int32_t rc = is_encoder_only ? llama_encode(ctx, batch) : llama_decode(ctx, batch);
    if (rc != 0) {
        sqlite3_free(tokens);
        sqlite3_free(embedding);
        sqlite_context_result_error(context, SQLITE_ERROR, "Model %s failed during embedding generation (%d)", is_encoder_only ? "encode" : "decode", rc);
        return;
    }

    // retrieve sentence embedding
    const float *result = NULL;
    if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
        result = llama_get_embeddings_ith(ctx, n_tokens - 1);
    } else {
        result = llama_get_embeddings_seq(ctx, sequence_id);
    }
    if (result == NULL) {
        sqlite3_free(tokens);
        sqlite3_free(embedding);
        sqlite_context_result_error(context, SQLITE_ERROR, "Failed to retrieve embedding vector from model");
        return;
    }

    // normalize or copy embedding
    if (ai->options.embedding.normalize) {
        llm_embed_normalize(result, embedding, type, dimension);
    } else {
        llm_embed_copy(result, embedding, type, dimension, embedding_size);
    }

    // clear memory so the next call starts clean
    if (memory) {
        llama_memory_seq_rm(memory, sequence_id, 0, -1);
        llama_memory_clear(memory, true);
    }

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
    llama_token *tokens = NULL;
    bool buffer_initialized = false;
    buffer_t buffer = {0};
    char *formatted_prompt = NULL;

    // sanity check vocab
    const struct llama_vocab *vocab = llama_model_get_vocab(ai->model);
    if (!vocab) {
        sqlite_context_result_error(context, SQLITE_ERROR, "Failed to extract vocabulary from the model");
        return;
    }

    struct llama_context *ctx = ai->ctx;
    if (ctx == NULL) {
        sqlite_context_result_error(context, SQLITE_ERROR, "No context found. Please call llm_context_create() before using this function.");
        return;
    }

    // if the model has a chat template, wrap the prompt so the model emits EOG tokens
    const char *chat_template = llama_model_chat_template(ai->model, NULL);
    if (chat_template) {
        llama_chat_message messages[] = {{ ROLE_USER, text }};
        int32_t formatted_len = llama_chat_apply_template(chat_template, messages, 1, true, NULL, 0);
        if (formatted_len > 0) {
            formatted_prompt = (char *)sqlite3_malloc64(formatted_len + 1);
            if (!formatted_prompt) {
                sqlite_context_result_error(context, SQLITE_NOMEM, "Out of memory: failed to allocate formatted prompt");
                return;
            }
            llama_chat_apply_template(chat_template, messages, 1, true, formatted_prompt, formatted_len + 1);
            formatted_prompt[formatted_len] = '\0';
            text = formatted_prompt;
            text_len = formatted_len;
        }
    }

    // clear KV cache so each generation starts clean
    llama_memory_t memory = llama_get_memory(ctx);
    if (memory) llama_memory_clear(memory, true);

    const int n_ctx = (int)llama_n_ctx(ctx);
    const int n_batch = (int)llama_n_batch(ctx);

    // find the number of tokens in the prompt
    int n_prompt = -llama_tokenize(vocab, text, text_len, NULL, 0, true, true);
    if (n_prompt <= 0) {
        sqlite_context_result_error(context, SQLITE_ERROR, "Unable to extract number of tokens from prompt");
        goto error;
    }

    // ensure prompt leaves room for at least one generated token
    int max_prompt = n_ctx - 1;
    if (max_prompt <= 0) {
        sqlite_context_result_error(context, SQLITE_ERROR, "Context size too small for text generation (n_ctx=%d)", n_ctx);
        goto error;
    }
    if (n_prompt > max_prompt) {
        n_prompt = max_prompt;
    }

    // allocate space for the tokens and tokenize the prompt
    tokens = (llama_token *)sqlite3_malloc(n_prompt * sizeof(llama_token));
    if (!tokens) {
        sqlite_context_result_error(context, SQLITE_NOMEM, "Out of memory: failed to allocate %d tokens", n_prompt);
        goto error;
    }

    int n_actual = llama_tokenize(vocab, text, text_len, tokens, n_prompt, true, true);
    if (n_actual < 0) {
        // input needs more tokens than n_prompt — tokenize fully then truncate
        int n_full = -n_actual;
        llama_token *full_tokens = (llama_token *)sqlite3_malloc(n_full * sizeof(llama_token));
        if (!full_tokens) {
            sqlite_context_result_error(context, SQLITE_NOMEM, "Out of memory: failed to allocate %d tokens", n_full);
            goto error;
        }
        int n_got = llama_tokenize(vocab, text, text_len, full_tokens, n_full, true, true);
        if (n_got < 0 || n_got != n_full) {
            sqlite3_free(full_tokens);
            sqlite_context_result_error(context, SQLITE_ERROR, "Tokenization failed");
            goto error;
        }
        memcpy(tokens, full_tokens, n_prompt * sizeof(llama_token));
        sqlite3_free(full_tokens);
    } else {
        n_prompt = n_actual;
    }

    // when n_predict is not set, default to 4096 tokens (capped by remaining context space)
    // and let the model stop naturally via EOG
    int n_predict = (ai->options.n_predict > 0) ? ai->options.n_predict : 4096;
    if (n_predict > n_ctx - n_prompt) n_predict = n_ctx - n_prompt;
    if (n_predict <= 0) {
        sqlite_context_result_error(context, SQLITE_ERROR, "Prompt fills entire context (%d tokens), no room for generation", n_prompt);
        goto error;
    }

    // initialize the sampler
    bool sampler_already_setup = (ai->sampler != NULL);
    struct llama_sampler *sampler = llm_sampler_check(ai);
    if (!sampler) goto error;
    if (!sampler_already_setup) {
        // no sampler was setup, so initialize it with some default values
        llama_sampler_chain_add(sampler, llama_sampler_init_penalties(64, 1.1, 0, 0));
        llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
    }

    // allocate output buffer (starts small, grows dynamically via buffer_append)
    if (!buffer_create(&buffer, 0)) {
        sqlite_context_result_error(context, SQLITE_NOMEM, "Out of memory: failed to allocate buffer");
        goto error_sampler;
    }
    buffer_initialized = true;

    // feed prompt in batches of n_batch tokens
    {
        int prompt_pos = 0;
        while (prompt_pos < n_prompt) {
            int chunk = n_prompt - prompt_pos;
            if (chunk > n_batch) chunk = n_batch;
            struct llama_batch batch = llama_batch_get_one(tokens + prompt_pos, chunk);
            if (llama_decode(ctx, batch)) {
                sqlite_context_result_error(context, SQLITE_ERROR, "Failed to execute the decoding function during prompt processing");
                goto error_sampler;
            }
            prompt_pos += chunk;
        }
    }

    // generate tokens
    {
        llama_token new_token_id;
        for (int i = 0; i < n_predict; i++) {
            new_token_id = llama_sampler_sample(sampler, ctx, -1);

            if (llama_vocab_is_eog(vocab, new_token_id)) {
                break;
            }

            char buf[MAX_TOKEN_TEXT_LEN];
            int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
            if (n < 0) {
                sqlite_context_result_error(context, SQLITE_ERROR, "Failed to convert token to piece (%d)", n);
                goto error_sampler;
            }

            if (buffer_append(&buffer, buf, n, true) == false) {
                sqlite_context_result_error(context, SQLITE_NOMEM, "Out of memory: failed to append to buffer");
                goto error_sampler;
            }

            // decode the sampled token to advance the KV cache
            struct llama_batch batch = llama_batch_get_one(&new_token_id, 1);
            if (llama_decode(ctx, batch)) {
                sqlite_context_result_error(context, SQLITE_ERROR, "Failed to execute the decoding function during generation");
                goto error_sampler;
            }
        }
    }

    // success — transfer buffer ownership to SQLite
    sqlite3_result_text(context, buffer.data, buffer.length, sqlite3_free);
    sqlite3_free(tokens);
    sqlite3_free(formatted_prompt);
    if (!sampler_already_setup) llama_sampler_free(sampler);
    return;

error_sampler:
    if (!sampler_already_setup && ai->sampler) {
        llama_sampler_free(ai->sampler);
        ai->sampler = NULL;
    }
error:
    if (buffer_initialized) buffer_destroy(&buffer);
    sqlite3_free(tokens);
    sqlite3_free(formatted_prompt);
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
    if (!buffer_create(&ai->chat.formatted, n_ctx) || !buffer_create(&ai->chat.response, MIN_ALLOC_RESPONSE)) {
        sqlite_common_set_error(ai->context, ai->vtab, SQLITE_NOMEM, "Out of memory: failed to allocate chat buffers");
        return false;
    }

    ai->chat.prompt = (char *)sqlite3_malloc(MIN_ALLOC_PROMPT);
    ai->chat.tokens = (llama_token *)sqlite3_malloc(sizeof(llama_token) * MIN_ALLOC_TOKEN);
    if (!ai->chat.prompt || !ai->chat.tokens) {
        sqlite_common_set_error(ai->context, ai->vtab, SQLITE_NOMEM, "Out of memory: failed to allocate chat prompt/token buffers");
        return false;
    }
    ai->chat.ntokens = MIN_ALLOC_TOKEN;

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
    
    // skip empty system message if present
    size_t messages_count = messages->count;
    const llama_chat_message *messages_items = messages->items;
    if (messages->count > 0) {
        const llama_chat_message first_message = messages->items[0];
        if (first_message.role == ROLE_SYSTEM && first_message.content[0] == '\0') {
            messages_items = messages->items + 1;
            messages_count = messages->count - 1;
        }
    }
    
    // transform a list of messages (the context) into
    // <|user|>What is AI?<|end|><|assistant|>AI stands for Artificial Intelligence...<|end|><|user|>Can you give an example?<|end|><|assistant|>...
    int32_t new_len = llama_chat_apply_template(template, messages_items, messages_count, true, formatted->data, formatted->capacity);
    if (new_len > formatted->capacity) {
        if (buffer_resize(formatted, new_len * 2) == false) return false;
        new_len = llama_chat_apply_template(template, messages_items, messages_count, true, formatted->data, formatted->capacity);
    }
    if ((new_len < 0) || (new_len > formatted->capacity)) {
        sqlite_common_set_error (ai->context, ai->vtab, SQLITE_ERROR, "failed to apply chat template");
        return false;
    }
    
    // check if there is enough space for the new formatted prompt
    int32_t prompt_len = new_len - ai->chat.prev_len;
    if (prompt_len <= 0) {
        sqlite_common_set_error (ai->context, ai->vtab, SQLITE_ERROR, "Invalid prompt length (template state inconsistency)");
        return false;
    }
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

    // save response before freeing the cursor
    ai_messages *messages = &ai->chat.messages;
    const char *template = ai->chat.template;
    bool saved = llm_chat_save_response(ai, messages, template);

    sqlite3_free(c);

    return saved ? SQLITE_OK : SQLITE_ERROR;
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
    ai->chat.tokens = NULL;
    ai->chat.ntokens = 0;

    if (ai->chat.prompt) sqlite3_free(ai->chat.prompt);
    ai->chat.prompt = NULL;
    ai->chat.prev_len = 0;

    ai->chat.template = NULL;
    ai->chat.vocab = NULL;
    ai->chat.token_count = 0;
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
        // skip empty system message placeholder
        if (role == ROLE_SYSTEM && (!content || content[0] == '\0')) continue;
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

    ai_context *ai = (ai_context *)sqlite3_user_data(context);

    // re-initialize chat state (UUID, buffers, tokens)
    if (llm_chat_check_context(ai) == false) return;

    // UUID
    const char *uuid = (const char *)sqlite3_value_text(argv[0]);
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
            sqlite_context_result_error(context, SQLITE_ERROR, "Failed to append message during restore");
            sqlite3_finalize(vm);
            return;
        }
        ++counter;
    }

    sqlite3_result_int(context, counter);
    if (vm) sqlite3_finalize(vm);
    return;

abort_restore:
    sqlite3_result_error(context, sqlite3_errmsg(db), -1);
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
    buffer_reset(&ai->chat.response);
    llm_chat_run(ai, NULL, user_prompt);
}

static void llm_chat_system_prompt(sqlite3_context *context, int argc, sqlite3_value **argv) {
    if (llm_check_context(context) == false)
        return;

    ai_context *ai = (ai_context *)sqlite3_user_data(context);
    if (llm_chat_check_context(ai) == false)
        return;

    ai_messages *messages = &ai->chat.messages;
    
    // get system role message
    if (argc == 0) {
        if (messages->count == 0) {
            sqlite3_result_null(context);
            return;
        }

        // only the first message is reserved to the system role
        llama_chat_message *system_message = &messages->items[0];
        const char *content = system_message->content;
        if (system_message->role == ROLE_SYSTEM && content && content[0] != '\0') {
            sqlite3_result_text(context, content, -1, SQLITE_TRANSIENT);
        } else {
            sqlite3_result_null(context);
        }

        return;
    }

    bool is_null_prompt = (sqlite3_value_type(argv[0]) == SQLITE_NULL);
    int types[1];
    types[0] = is_null_prompt ? SQLITE_NULL : SQLITE_TEXT;

    if (sqlite_sanity_function(context, "llm_chat_system_prompt", argc, argv, 1, types, true, false) == false)
        return;

    const unsigned char *prompt_text = sqlite3_value_text(argv[0]);
    const char *system_prompt = prompt_text ? (const char *)prompt_text : "";
    if (!llm_messages_set(messages, 0, ROLE_SYSTEM, system_prompt)) {
        if (!llm_messages_append(messages, ROLE_SYSTEM, system_prompt)) {
            sqlite_common_set_error (ai->context, ai->vtab, SQLITE_ERROR, "Failed to set chat system prompt");
            return;
        }
    }
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
    if (ai->ctx) llama_set_adapters_lora(ai->ctx, NULL, 0, NULL);
    for (int i=0; i<MAX_LORAS; ++i) {
        if (ai->lora[i]) {
            llama_adapter_lora_free(ai->lora[i]);
            ai->lora[i] = NULL;
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
    
    {
        struct llama_adapter_lora *adapters[MAX_LORAS];
        float scales[MAX_LORAS];
        size_t n = 0;
        for (int i=0; i<MAX_LORAS; ++i) {
            if (ai->lora[i] && ai->lora_scale[i] != 0.0) {
                adapters[n] = ai->lora[i];
                scales[n] = ai->lora_scale[i];
                n++;
            }
        }
        llama_set_adapters_lora(ai->ctx, adapters, n, scales);
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

    // auto-size n_ctx to the model's training window when no explicit context_size was set
    // llama_context_default_params() sets n_ctx=512; setting n_ctx=0 tells llama.cpp to use n_ctx_train
    struct llama_context_params defaults = llama_context_default_params();
    if (ai->model && ctx_params.n_ctx == defaults.n_ctx) {
        ctx_params.n_ctx = 0;
    }

    // for embedding contexts, clamp n_ctx to n_ctx_train to avoid position overflow
    if (ctx_params.embeddings && ai->model) {
        int n_ctx_train = llama_model_n_ctx_train(ai->model);
        if ((int)ctx_params.n_ctx > n_ctx_train) {
            ctx_params.n_ctx = n_ctx_train;
        }
        if ((int)ctx_params.n_batch > (int)ctx_params.n_ctx && ctx_params.n_ctx > 0) {
            ctx_params.n_batch = ctx_params.n_ctx;
        }
        if ((int)ctx_params.n_ubatch > (int)ctx_params.n_batch) {
            ctx_params.n_ubatch = ctx_params.n_batch;
        }
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

static void llm_context_size (sqlite3_context *context, int argc, sqlite3_value **argv) {
    ai_context *ai = (ai_context *)sqlite3_user_data(context);
    if (!ai->ctx) {
        sqlite_context_result_error(context, SQLITE_MISUSE, "No context found. Please call llm_context_create() before using this function.");
        return;
    }
    uint32_t n_ctx = llama_n_ctx(ai->ctx);
    sqlite3_result_int(context, n_ctx);
}

static void llm_context_used (sqlite3_context *context, int argc, sqlite3_value **argv) {
    ai_context *ai = (ai_context *)sqlite3_user_data(context);
    if (!ai->ctx) {
        sqlite_context_result_error(context, SQLITE_MISUSE, "No context found. Please call llm_context_create() before using this function.");
        return;
    }
    int32_t n_ctx_used = llama_memory_seq_pos_max(llama_get_memory(ai->ctx), 0) + 1;
    if (n_ctx_used < 0) n_ctx_used = 0;
    sqlite3_result_int(context, n_ctx_used);
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

    if (label) sqlite3_result_text(context, label, -1, SQLITE_STATIC);
    else sqlite3_result_null(context);
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

// detect audio format from file extension
static int audio_detect_format_from_path (const char *path) {
    if (!path) return 0;
    const char *dot = strrchr(path, '.');
    if (!dot) return 0;
    if (strcasecmp(dot, ".wav") == 0) return 1;
    if (strcasecmp(dot, ".mp3") == 0) return 2;
    if (strcasecmp(dot, ".flac") == 0) return 3;
    return 0;
}

// detect audio format from blob header (magic bytes)
static int audio_detect_format_from_blob (const void *data, size_t size) {
    if (!data || size < 4) return 0;
    const uint8_t *bytes = (const uint8_t *)data;

    // WAV: starts with "RIFF"
    if (bytes[0] == 'R' && bytes[1] == 'I' && bytes[2] == 'F' && bytes[3] == 'F') return 1;

    // MP3: starts with ID3 tag or MPEG sync word
    if (bytes[0] == 'I' && bytes[1] == 'D' && bytes[2] == '3') return 2;
    if (bytes[0] == 0xFF && (bytes[1] & 0xE0) == 0xE0) return 2;

    // FLAC: starts with "fLaC"
    if (bytes[0] == 'f' && bytes[1] == 'L' && bytes[2] == 'a' && bytes[3] == 'C') return 3;

    return 0;
}

// convert multi-channel audio to mono by averaging channels, then resample to target_rate
static float *audio_convert_to_mono_16khz (const float *src, uint64_t num_samples, uint32_t sample_rate, uint32_t channels, int *out_samples) {
    // step 1: downmix to mono
    uint64_t mono_count = num_samples; // num_samples is per-channel frame count
    float *mono = NULL;

    if (channels == 1) {
        // already mono — just reference (we'll copy during resampling)
        mono = (float *)src;
    } else {
        mono = (float *)sqlite3_malloc64(mono_count * sizeof(float));
        if (!mono) return NULL;
        for (uint64_t i = 0; i < mono_count; i++) {
            float sum = 0.0f;
            for (uint32_t c = 0; c < channels; c++) {
                sum += src[i * channels + c];
            }
            mono[i] = sum / channels;
        }
    }

    // step 2: resample to WHISPER_SAMPLE_RATE (16000 Hz) if needed
    if (sample_rate == WHISPER_SAMPLE_RATE) {
        if (mono == src) {
            // need to copy so caller can free
            float *copy = (float *)sqlite3_malloc64(mono_count * sizeof(float));
            if (!copy) return NULL;
            memcpy(copy, src, mono_count * sizeof(float));
            mono = copy;
        }
        *out_samples = (int)mono_count;
        return mono;
    }

    // linear interpolation resampling
    double ratio = (double)WHISPER_SAMPLE_RATE / (double)sample_rate;
    int64_t out_count = (int64_t)(mono_count * ratio) + 1;
    float *resampled = (float *)sqlite3_malloc64(out_count * sizeof(float));
    if (!resampled) {
        if (mono != src) sqlite3_free(mono);
        return NULL;
    }

    for (int64_t i = 0; i < out_count; i++) {
        double src_idx = i / ratio;
        int64_t idx0 = (int64_t)src_idx;
        double frac = src_idx - idx0;
        if (idx0 >= (int64_t)mono_count - 1) {
            resampled[i] = mono[mono_count - 1];
        } else {
            resampled[i] = (float)(mono[idx0] * (1.0 - frac) + mono[idx0 + 1] * frac);
        }
    }

    if (mono != src) sqlite3_free(mono);
    *out_samples = (int)out_count;
    return resampled;
}

static void audio_model_transcribe (sqlite3_context *context, int argc, sqlite3_value **argv) {
    if (audio_process_check_arguments(context, "audio_model_transcribe", argc, argv, true) == false) return;

    ai_context *ai = (ai_context *)sqlite3_user_data(context);
    float *pcm_buffer = NULL;
    uint64_t num_samples = 0;
    uint32_t sample_rate = 0;
    uint32_t channels = 0;

    if (sqlite3_value_type(argv[0]) == SQLITE_TEXT) {
        const char *path = (const char *)sqlite3_value_text(argv[0]);
        int format = audio_detect_format_from_path(path);
        switch (format) {
            case 1: pcm_buffer = audio_wav_file2pcm(path, &num_samples, &sample_rate, &channels); break;
            case 2: pcm_buffer = audio_mp3_file2pcm(path, &num_samples, &sample_rate, &channels); break;
            case 3: pcm_buffer = audio_flac_file2pcm(path, &num_samples, &sample_rate, &channels); break;
            default:
                sqlite_context_result_error(context, SQLITE_ERROR, "Unsupported audio format for file '%s'. Supported: .wav, .mp3, .flac", path);
                return;
        }
        if (!pcm_buffer) {
            sqlite_context_result_error(context, SQLITE_ERROR, "Unable to decode audio file '%s'", path);
            return;
        }
    } else {
        const void *data = sqlite3_value_blob(argv[0]);
        size_t data_size = (size_t)sqlite3_value_bytes(argv[0]);
        int format = audio_detect_format_from_blob(data, data_size);
        switch (format) {
            case 1: pcm_buffer = audio_wav_mem2pcm(data, data_size, &num_samples, &sample_rate, &channels); break;
            case 2: pcm_buffer = audio_mp3_mem2pcm(data, data_size, &num_samples, &sample_rate, &channels); break;
            case 3: pcm_buffer = audio_flac_mem2pcm(data, data_size, &num_samples, &sample_rate, &channels); break;
            default:
                sqlite_context_result_error(context, SQLITE_ERROR, "Unsupported audio format in BLOB. Supported: WAV, MP3, FLAC");
                return;
        }
        if (!pcm_buffer) {
            sqlite_context_result_error(context, SQLITE_ERROR, "Unable to decode audio BLOB");
            return;
        }
    }

    // convert to mono 16kHz as required by whisper
    int whisper_samples = 0;
    float *whisper_pcm = audio_convert_to_mono_16khz(pcm_buffer, num_samples, sample_rate, channels, &whisper_samples);
    sqlite3_free(pcm_buffer); // allocated via miniaudio's sqlite3_malloc wrapper

    if (!whisper_pcm || whisper_samples <= 0) {
        sqlite_context_result_error(context, SQLITE_ERROR, "Failed to convert audio to mono 16kHz PCM");
        return;
    }

    // parse transcription options
    struct whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    params.print_special = false;
    params.print_progress = false;
    params.print_realtime = false;
    params.print_timestamps = false;

    // save default pointers so we can tell which ones we allocated
    const char *default_language = params.language;
    const char *default_initial_prompt = params.initial_prompt;
    const char *default_suppress_regex = params.suppress_regex;

    const char *options = (argc >= 2) ? (const char *)sqlite3_value_text(argv[1]) : NULL;
    if (parse_keyvalue_string(ai, options, whisper_full_params_options_callback, &params) == false) {
        sqlite3_free(whisper_pcm);
        if (params.language != default_language) sqlite3_free((void *)params.language);
        if (params.initial_prompt != default_initial_prompt) sqlite3_free((void *)params.initial_prompt);
        if (params.suppress_regex != default_suppress_regex) sqlite3_free((void *)params.suppress_regex);
        sqlite_context_result_error(context, SQLITE_ERROR, "An error occurred while parsing options (%s)", options);
        return;
    }

    // run whisper inference
    int rc = whisper_full(ai->whisper, params, whisper_pcm, whisper_samples);
    sqlite3_free(whisper_pcm);

    // free allocated option strings (only those we allocated via sqlite_strdup)
    if (params.language != default_language) sqlite3_free((void *)params.language);
    if (params.initial_prompt != default_initial_prompt) sqlite3_free((void *)params.initial_prompt);
    if (params.suppress_regex != default_suppress_regex) sqlite3_free((void *)params.suppress_regex);

    if (rc != 0) {
        sqlite_context_result_error(context, SQLITE_ERROR, "Whisper transcription failed (error code %d)", rc);
        return;
    }

    // collect all segments into a single result
    int n_segments = whisper_full_n_segments(ai->whisper);
    if (n_segments == 0) {
        sqlite3_result_text(context, "", 0, SQLITE_STATIC);
        return;
    }

    if (n_segments == 1) {
        const char *text = whisper_full_get_segment_text(ai->whisper, 0);
        sqlite3_result_text(context, text, -1, SQLITE_TRANSIENT);
        return;
    }

    // multiple segments — concatenate
    buffer_t result = {0};
    if (!buffer_create(&result, 4096)) {
        sqlite_context_result_error(context, SQLITE_NOMEM, "Out of memory");
        return;
    }

    for (int i = 0; i < n_segments; i++) {
        const char *seg_text = whisper_full_get_segment_text(ai->whisper, i);
        if (seg_text) {
            buffer_append(&result, seg_text, (uint32_t)strlen(seg_text), false);
        }
    }

    sqlite3_result_text(context, result.data, result.length, sqlite3_free);
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
    
    rc = sqlite3_create_function(db, "llm_context_size", 0, SQLITE_UTF8, ctx, llm_context_size, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function(db, "llm_context_used", 0, SQLITE_UTF8, ctx, llm_context_used, NULL, NULL);
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

    rc = sqlite3_create_function(db, "llm_chat_system_prompt", 0, SQLITE_UTF8, ctx, llm_chat_system_prompt, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;

    rc = sqlite3_create_function(db, "llm_chat_system_prompt", 1, SQLITE_UTF8, ctx, llm_chat_system_prompt, NULL, NULL);
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
    rc = sqlite3_create_function(db, "audio_model_load", 1, SQLITE_UTF8, ctx, audio_model_load, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;

    rc = sqlite3_create_function(db, "audio_model_load", 2, SQLITE_UTF8, ctx, audio_model_load, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;

    rc = sqlite3_create_function(db, "audio_model_free", 0, SQLITE_UTF8, ctx, audio_model_free, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;

    rc = sqlite3_create_function(db, "audio_model_transcribe", 1, SQLITE_UTF8, ctx, audio_model_transcribe, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;

    rc = sqlite3_create_function(db, "audio_model_transcribe", 2, SQLITE_UTF8, ctx, audio_model_transcribe, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
     
cleanup:
    return rc;
}
