#include "sqlite3.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef SQLITEAI_LOAD_FROM_SOURCES
#include "sqlite-ai.h"
#endif

// Just a lightweight model to use for testing
#define DEFAULT_MODEL_PATH "tests/models/unsloth/gemma-3-270m-it-GGUF/gemma-3-270m-it-UD-IQ2_M.gguf"

typedef struct {
    const char *extension_path;
    const char *model_path;
    const char *whisper_model_path;
    const char *audio_path;
    bool verbose;
} test_env;

typedef int (*test_fn)(const test_env *env);

typedef struct {
    const char *name;
    test_fn fn;
} test_case;

static void usage(const char *prog) {
    fprintf(stderr, "Usage: %s [--extension /path/to/ai] [--model /path/to/model] [--whisper-model /path/to/whisper] [--audio /path/to/audio.wav] [--verbose]\n", prog);
}

static int expect_error_contains(const char *err_msg, const char *needle) {
    if (!err_msg) {
        fprintf(stderr, "Expected SQLite error message but got NULL\n");
        return 1;
    }
    if (!strstr(err_msg, needle)) {
        fprintf(stderr, "Expected error to contain \"%s\", got: %s\n", needle, err_msg);
        return 1;
    }
    return 0;
}

static int open_db_and_load(const test_env *env, sqlite3 **out_db) {
    sqlite3 *db = NULL;
    int rc = sqlite3_open(":memory:", &db);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "sqlite3_open failed: %s\n", db ? sqlite3_errmsg(db) : "unknown error");
        if (db) sqlite3_close(db);
        return rc;
    }
    sqlite3_enable_load_extension(db, 1);
    char *errmsg = NULL;
#ifdef SQLITEAI_LOAD_FROM_SOURCES
    rc = sqlite3_ai_init(db, NULL, NULL);
#else
    rc = sqlite3_load_extension(db, env->extension_path, NULL, &errmsg);
#endif
    if (rc != SQLITE_OK) {
        fprintf(stderr, "sqlite3_load_extension failed: %s\n", errmsg ? errmsg : sqlite3_errmsg(db));
        if (errmsg) sqlite3_free(errmsg);
        sqlite3_close(db);
        return rc;
    }
    if (errmsg) sqlite3_free(errmsg);
    *out_db = db;
    return SQLITE_OK;
}

// ---------------------------------------------------------------------
// Helper utilities
// ---------------------------------------------------------------------

typedef struct {
    const test_env *env;
} exec_userdata;

static int select_single_int(const test_env *env, sqlite3 *db, const char *sql, int *value_out) {
    if (env->verbose) {
        printf("[SQL] %s\n", sql);
    }
    sqlite3_stmt *stmt = NULL;
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "sqlite3_prepare_v2 failed (%d): %s\n", rc, sqlite3_errmsg(db));
        if (stmt) sqlite3_finalize(stmt);
        return 1;
    }
    rc = sqlite3_step(stmt);
    if (rc != SQLITE_ROW) {
        fprintf(stderr, "Expected a row for query: %s\n", sql);
        sqlite3_finalize(stmt);
        return 1;
    }
    if (value_out) *value_out = sqlite3_column_int(stmt, 0);
    sqlite3_finalize(stmt);
    return 0;
}

static int verbose_callback(void *udata, int columns, char **values, char **names) {
    exec_userdata *ud = (exec_userdata *)udata;
    if (!ud || !ud->env || !ud->env->verbose) {
        return 0;
    }
    printf("[SQL] row:\n");
    for (int i = 0; i < columns; ++i) {
        printf("  %s = %s\n", names[i] ? names[i] : "(null)", values[i] ? values[i] : "NULL");
    }
    return 0;
}

static int exec_expect_error(const test_env *env, sqlite3 *db, const char *sql, const char *needle) {
    if (env->verbose) {
        printf("[SQL] %s\n", sql);
    }
    char *errmsg = NULL;
    exec_userdata udata = {.env = env};
    int rc = sqlite3_exec(db, sql, env->verbose ? verbose_callback : NULL, env->verbose ? &udata : NULL, &errmsg);
    if (rc == SQLITE_OK) {
        fprintf(stderr, "Expected failure executing SQL: %s\n", sql);
        return 1;
    }
    if (env->verbose && errmsg) {
        printf("[SQL][ERROR] %s\n", errmsg);
    }
    int status = expect_error_contains(errmsg, needle);
    sqlite3_free(errmsg);
    return status;
}

static int exec_expect_ok(const test_env *env, sqlite3 *db, const char *sql) {
    if (env->verbose) {
        printf("[SQL] %s\n", sql);
    }
    char *errmsg = NULL;
    exec_userdata udata = {.env = env};
    int rc = sqlite3_exec(db, sql, env->verbose ? verbose_callback : NULL, env->verbose ? &udata : NULL, &errmsg);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "SQL execution failed (%d): %s\n", rc, errmsg ? errmsg : sqlite3_errmsg(db));
        if (errmsg) sqlite3_free(errmsg);
        return 1;
    }
    if (errmsg) sqlite3_free(errmsg);
    return 0;
}

static int exec_select_rows(const test_env *env, sqlite3 *db, const char *sql, int *rows_out) {
    if (env->verbose) {
        printf("[SQL] %s\n", sql);
    }
    sqlite3_stmt *stmt = NULL;
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "sqlite3_prepare_v2 failed (%d): %s\n", rc, sqlite3_errmsg(db));
        if (stmt) sqlite3_finalize(stmt);
        return 1;
    }
    int row_count = 0;
    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
        row_count++;
        if (env->verbose) {
            printf("[SQL][ROW] ");
            int cols = sqlite3_column_count(stmt);
            for (int i = 0; i < cols; ++i) {
                const char *name = sqlite3_column_name(stmt, i);
                const unsigned char *val = sqlite3_column_text(stmt, i);
                printf("%s=%s%s", name ? name : "(null)", val ? (const char *)val : "NULL", (i + 1 < cols) ? ", " : "");
            }
            printf("\n");
        }
    }
    if (rc != SQLITE_DONE) {
        fprintf(stderr, "sqlite3_step failed (%d): %s\n", rc, sqlite3_errmsg(db));
        sqlite3_finalize(stmt);
        return 1;
    }
    sqlite3_finalize(stmt);
    if (rows_out) *rows_out = row_count;
    return 0;
}

static int assert_sqlite_memory_clean(const char *label, const test_env *env) {
    sqlite3_int64 current = 0;
    sqlite3_int64 highwater = 0;
    if (sqlite3_status64(SQLITE_STATUS_MEMORY_USED, &current, &highwater, 0) != SQLITE_OK) {
        fprintf(stderr, "[%s] sqlite3_status64 failed\n", label);
        return 1;
    }
    if (env->verbose) {
        printf("[STATUS][%s] memory current=%lld highwater=%lld\n",
               label, (long long)current, (long long)highwater);
    }
    if (current != 0) {
        fprintf(stderr, "[%s] sqlite3 memory leak detected: current=%lld highwater=%lld\n",
                label, (long long)current, (long long)highwater);
        return 1;
    }
    return 0;
}

static int exec_query_text(const test_env *env, sqlite3 *db, const char *sql, char *text_out, size_t text_out_len) {
    if (env->verbose) {
        printf("[SQL] %s\n", sql);
    }
    sqlite3_stmt *stmt = NULL;
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "sqlite3_prepare_v2 failed (%d): %s\n", rc, sqlite3_errmsg(db));
        if (stmt) sqlite3_finalize(stmt);
        return 1;
    }
    rc = sqlite3_step(stmt);
    if (rc != SQLITE_ROW) {
        fprintf(stderr, "Expected a row for query: %s (rc=%d)\n", sql, rc);
        sqlite3_finalize(stmt);
        return 1;
    }
    const unsigned char *text = sqlite3_column_text(stmt, 0);
    if (text_out && text_out_len > 0) {
        if (text) {
            snprintf(text_out, text_out_len, "%s", (const char *)text);
        } else {
            text_out[0] = '\0';
        }
    }
    rc = sqlite3_step(stmt);
    if (rc != SQLITE_DONE) {
        fprintf(stderr, "Unexpected extra rows for query: %s\n", sql);
        sqlite3_finalize(stmt);
        return 1;
    }
    sqlite3_finalize(stmt);
    return 0;
}

static int query_system_prompt(const test_env *env, sqlite3 *db, char *buffer, size_t buffer_len, bool *is_null) {
    const char *sql = "SELECT llm_chat_system_prompt();";
    if (env->verbose) {
        printf("[SQL] %s\n", sql);
    }
    sqlite3_stmt *stmt = NULL;
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "sqlite3_prepare_v2 failed (%d): %s\n", rc, sqlite3_errmsg(db));
        if (stmt) sqlite3_finalize(stmt);
        return 1;
    }
    rc = sqlite3_step(stmt);
    if (rc != SQLITE_ROW) {
        fprintf(stderr, "Expected a row for query: %s (rc=%d)\n", sql, rc);
        sqlite3_finalize(stmt);
        return 1;
    }
    if (sqlite3_column_type(stmt, 0) == SQLITE_NULL) {
        if (is_null) *is_null = true;
        if (buffer && buffer_len) buffer[0] = '\0';
    } else {
        if (is_null) *is_null = false;
        const unsigned char *text = sqlite3_column_text(stmt, 0);
        if (buffer && buffer_len) {
            snprintf(buffer, buffer_len, "%s", text ? (const char *)text : "");
        }
    }
    sqlite3_finalize(stmt);
    return 0;
}

typedef struct {
    int id;
    int chat_id;
    char role[32];
    char content[4096];
} ai_chat_message_row;

static int fetch_ai_chat_messages(const test_env *env, sqlite3 *db, ai_chat_message_row *rows, size_t max_rows, int *count_out) {
    const char *sql = "SELECT * FROM ai_chat_messages ORDER BY id ASC;";
    if (env->verbose) {
        printf("[SQL] %s\n", sql);
    }

    sqlite3_stmt *stmt = NULL;
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "sqlite3_prepare_v2 failed (%d): %s\n", rc, sqlite3_errmsg(db));
        if (stmt) sqlite3_finalize(stmt);
        return 1;
    }

    int count = 0;
    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
        if (rows && (size_t)count < max_rows) {
            rows[count].id = sqlite3_column_int(stmt, 0);
            rows[count].chat_id = sqlite3_column_int(stmt, 1);
            const unsigned char *role = sqlite3_column_text(stmt, 2);
            const unsigned char *content = sqlite3_column_text(stmt, 3);
            snprintf(rows[count].role, sizeof(rows[count].role), "%s", role ? (const char *)role : "");
            snprintf(rows[count].content, sizeof(rows[count].content), "%s", content ? (const char *)content : "");
        }
        count++;
    }
    if (rc != SQLITE_DONE) {
        fprintf(stderr, "sqlite3_step failed (%d): %s\n", rc, sqlite3_errmsg(db));
        sqlite3_finalize(stmt);
        return 1;
    }
    sqlite3_finalize(stmt);

    if (rows && (size_t)count > max_rows) {
        fprintf(stderr, "Expected at most %zu messages but found %d\n", max_rows, count);
        return 1;
    }
    if (count_out) *count_out = count;
    return 0;
}

// ---------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------

static int test_issue15_chat_without_context(const test_env *env) {
    sqlite3 *db = NULL;
    if (open_db_and_load(env, &db) != SQLITE_OK) {
        return 1;
    }

    int rc = exec_expect_error(env, db, "SELECT llm_chat_create();", "Please call llm_context_create()");
    sqlite3_close(db);
    if (rc == 0) {
        return assert_sqlite_memory_clean("issue15", env);
    }
    return rc;
}

static int test_llm_chat_respond_repeated(const test_env *env) {
    sqlite3 *db = NULL;
    if (open_db_and_load(env, &db) != SQLITE_OK) {
        return 1;
    }

    char sqlbuf[512];
    const char *model = env->model_path ? env->model_path : DEFAULT_MODEL_PATH;
    snprintf(sqlbuf, sizeof(sqlbuf), "SELECT llm_model_load('%s');", model);
    if (exec_expect_ok(env, db, sqlbuf) != 0) {
        sqlite3_close(db);
        return 1;
    }
    if (exec_expect_ok(env, db, "SELECT llm_context_create('context_size=1000');") != 0) {
        sqlite3_close(db);
        return 1;
    }
    if (exec_expect_ok(env, db, "SELECT llm_chat_create();") != 0) {
        sqlite3_close(db);
        return 1;
    }

    const int iterations = 3;
    char *prompts[] = {
        "SELECT llm_chat_respond('Hi');", 
        "SELECT llm_chat_respond('How are you?');",
        "SELECT llm_chat_respond('Again');"
    };
    for (int i = 0; i < iterations; ++i) {
        if (exec_expect_ok(env, db, prompts[i]) != 0) {
            sqlite3_close(db);
            return 1;
        }
        
        if (exec_expect_ok(env, db, "SELECT llm_context_used() AS context_used, llm_context_size() AS context_size, CAST(llm_context_used() AS FLOAT)/CAST(llm_context_size() AS FLOAT) || '%' AS 'context_usage_percentage';") != 0) {
            sqlite3_close(db);
            return 1;
        }
    }

    if (exec_expect_ok(env, db, "SELECT llm_chat_free();") != 0) {
        sqlite3_close(db);
        return 1;
    }
    if (exec_expect_ok(env, db, "SELECT llm_context_free();") != 0) {
        sqlite3_close(db);
        return 1;
    }
    if (exec_expect_ok(env, db, "SELECT llm_model_free();") != 0) {
        sqlite3_close(db);
        return 1;
    }

    sqlite3_close(db);

    return assert_sqlite_memory_clean("chat_respond_repeated", env);
}

static int test_llm_chat_vtab(const test_env *env) {
    sqlite3 *db = NULL;
    if (open_db_and_load(env, &db) != SQLITE_OK) {
        return 1;
    }

    const char *model = env->model_path ? env->model_path : DEFAULT_MODEL_PATH;
    char sqlbuf[512];
    snprintf(sqlbuf, sizeof(sqlbuf), "SELECT llm_model_load('%s');", model);
    if (exec_expect_ok(env, db, sqlbuf) != 0) goto fail;
    if (exec_expect_ok(env, db, "SELECT llm_context_create('context_size=1000');") != 0) goto fail;
    if (exec_expect_ok(env, db, "SELECT llm_chat_create();") != 0) goto fail;
    int rows = 0;
    if (exec_select_rows(env, db, "SELECT * FROM llm_chat('Hi');", &rows) != 0) goto fail;
    if (rows <= 0) {
        fprintf(stderr, "[chat_vtab] expected rows but got %d\n", rows);
        goto fail;
    }
    rows = 0;
    if (exec_select_rows(env, db, "SELECT * FROM llm_chat('How are you');", &rows) != 0) goto fail;
    if (rows <= 0) {
        fprintf(stderr, "[chat_vtab] expected rows but got %d\n", rows);
        goto fail;
    }
    if (exec_expect_ok(env, db, "SELECT llm_chat_free();") != 0) goto fail;
    if (exec_expect_ok(env, db, "SELECT llm_context_free();") != 0) goto fail;
    if (exec_expect_ok(env, db, "SELECT llm_model_free();") != 0) goto fail;
    sqlite3_close(db);

    return assert_sqlite_memory_clean("chat_vtab", env);

fail:
    if (db) sqlite3_close(db);
    return 1;
}

static int test_llm_embed_generate(const test_env *env) {
    sqlite3 *db = NULL;
    if (open_db_and_load(env, &db) != SQLITE_OK) {
        return 1;
    }

    const char *model = env->model_path ? env->model_path : DEFAULT_MODEL_PATH;
    char sqlbuf[512];
    const char *options = "log_info=1";
    snprintf(sqlbuf, sizeof(sqlbuf), "SELECT llm_model_load('%s','%s');", model, options);

    if (exec_expect_ok(env, db, sqlbuf) != 0) goto fail;
    if (exec_expect_ok(env, db, "SELECT llm_context_create_embedding('context_size=1000,embedding_type=UINT8');") != 0)
        goto fail;
    if (exec_expect_ok(env, db, "SELECT llm_embed_generate('embedding test text');") != 0)
        goto fail;

    // Intentionally skip llm_context_free/llm_model_free to mimic how Python GC drops
    // connections without calling the cleanup helpers (see GH issue #14).
    sqlite3_close(db);
    db = NULL;

    // Reopening another connection will reinitialize the extension; on unfixed builds
    // this often hits the crash because the global logger still points to the freed ctx.
    if (open_db_and_load(env, &db) != SQLITE_OK) {
        return 1;
    }
    sqlite3_close(db);
    
    return assert_sqlite_memory_clean("llm_embed_generate", env);

fail:
    if (db) sqlite3_close(db);
    return 1;
}

static int test_llm_embed_generate_basic(const test_env *env) {
    sqlite3 *db = NULL;
    sqlite3_stmt *stmt = NULL;
    int status = 1;

    if (open_db_and_load(env, &db) != SQLITE_OK) {
        return 1;
    }

    const char *model = env->model_path ? env->model_path : DEFAULT_MODEL_PATH;
    char sqlbuf[512];
    snprintf(sqlbuf, sizeof(sqlbuf), "SELECT llm_model_load('%s');", model);
    if (exec_expect_ok(env, db, sqlbuf) != 0) goto cleanup;
    if (exec_expect_ok(env, db, "SELECT llm_context_create_embedding('embedding_type=UINT8');") != 0) goto cleanup;

    const char *embed_sql = "SELECT llm_embed_generate('hello world') AS embedding;";
    if (env->verbose) {
        printf("[SQL] %s\n", embed_sql);
    }
    if (sqlite3_prepare_v2(db, embed_sql, -1, &stmt, NULL) != SQLITE_OK) {
        fprintf(stderr, "sqlite3_prepare_v2 failed: %s\n", sqlite3_errmsg(db));
        goto cleanup;
    }
    int rc = sqlite3_step(stmt);
    if (rc != SQLITE_ROW) {
        fprintf(stderr, "Expected one row from llm_embed_generate, got rc=%d\n", rc);
        goto cleanup;
    }
    const void *blob = sqlite3_column_blob(stmt, 0);
    int blob_bytes = sqlite3_column_bytes(stmt, 0);
    if (blob == NULL || blob_bytes <= 0) {
        fprintf(stderr, "Embedding blob is empty (bytes=%d)\n", blob_bytes);
        goto cleanup;
    }

    status = 0;

cleanup:
    if (stmt) sqlite3_finalize(stmt);
    exec_expect_ok(env, db, "SELECT llm_context_free();");
    exec_expect_ok(env, db, "SELECT llm_model_free();");
    if (db) sqlite3_close(db);
    if (status == 0) {
        if (assert_sqlite_memory_clean("llm_embed_generate_basic", env) != 0) {
            return 1;
        }
    }
    return status;
}

static int test_llm_embedding_then_chat(const test_env *env) {
    sqlite3 *db = NULL;
    if (open_db_and_load(env, &db) != SQLITE_OK) {
        return 1;
    }

    const char *model = env->model_path ? env->model_path : DEFAULT_MODEL_PATH;
    char sqlbuf[512];
    snprintf(sqlbuf, sizeof(sqlbuf), "SELECT llm_model_load('%s');", model);
    if (exec_expect_ok(env, db, sqlbuf) != 0) goto fail;

    if (exec_expect_ok(env, db, "SELECT llm_context_create_embedding('embedding_type=UINT8');") != 0) goto fail;
    if (exec_expect_ok(env, db, "SELECT llm_embed_generate('document text for embeddings');") != 0) goto fail;
    if (exec_expect_ok(env, db, "SELECT llm_context_free();") != 0) goto fail;

    if (exec_expect_ok(env, db, "SELECT llm_context_create_chat('context_size=512');") != 0) goto fail;
    if (exec_expect_ok(env, db, "SELECT llm_chat_create();") != 0) goto fail;
    if (exec_expect_ok(env, db, "SELECT llm_chat_respond('Summarize the previous document.');") != 0) goto fail;
    if (exec_expect_ok(env, db, "SELECT llm_chat_free();") != 0) goto fail;
    if (exec_expect_ok(env, db, "SELECT llm_context_free();") != 0) goto fail;
    if (exec_expect_ok(env, db, "SELECT llm_model_free();") != 0) goto fail;

    sqlite3_close(db);
    return assert_sqlite_memory_clean("llm_embedding_then_chat", env);

fail:
    if (db) sqlite3_close(db);
    return 1;
}

static int test_llm_context_size_errors(const test_env *env) {
    sqlite3 *db = NULL;
    if (open_db_and_load(env, &db) != SQLITE_OK) {
        return 1;
    }

    if (exec_expect_error(env, db, "SELECT llm_context_size();", "No context found") != 0) {
        sqlite3_close(db);
        return 1;
    }

    const char *model = env->model_path ? env->model_path : DEFAULT_MODEL_PATH;
    char sqlbuf[512];
    snprintf(sqlbuf, sizeof(sqlbuf), "SELECT llm_model_load('%s');", model);
    if (exec_expect_ok(env, db, sqlbuf) != 0) goto fail;
    if (exec_expect_ok(env, db, "SELECT llm_context_create('context_size=256');") != 0) goto fail;
    if (exec_expect_ok(env, db, "SELECT llm_context_size();") != 0) goto fail;
    if (exec_expect_ok(env, db, "SELECT llm_context_free();") != 0) goto fail;
    if (exec_expect_ok(env, db, "SELECT llm_model_free();") != 0) goto fail;

    sqlite3_close(db);
    return assert_sqlite_memory_clean("llm_context_size_errors", env);

fail:
    if (db) sqlite3_close(db);
    return 1;
}

static int test_document_ingestion_flow(const test_env *env) {
    sqlite3 *db = NULL;
    if (open_db_and_load(env, &db) != SQLITE_OK) {
        return 1;
    }

    const char *model = env->model_path ? env->model_path : DEFAULT_MODEL_PATH;
    char sqlbuf[512];
    snprintf(sqlbuf, sizeof(sqlbuf), "SELECT llm_model_load('%s');", model);
    if (exec_expect_ok(env, db, sqlbuf) != 0) goto fail;

    if (exec_expect_ok(env, db, "SELECT llm_context_create_embedding('context_size=768,embedding_type=UINT8');") != 0) goto fail;
    if (exec_expect_ok(env, db, "SELECT llm_embed_generate('Document chunk content.');") != 0) goto fail;
    if (exec_expect_ok(env, db, "SELECT llm_embed_generate('Sentence level content.');") != 0) goto fail;
    if (exec_expect_ok(env, db, "SELECT llm_context_free();") != 0) goto fail;

    if (exec_expect_ok(env, db, "SELECT llm_context_create_chat('context_size=768');") != 0) goto fail;
    if (exec_expect_ok(env, db, "SELECT llm_chat_create();") != 0) goto fail;
    if (exec_expect_ok(env, db, "SELECT llm_chat_respond('Return a concise answer');") != 0) goto fail;
    if (exec_expect_ok(env, db, "SELECT llm_chat_free();") != 0) goto fail;
    if (exec_expect_ok(env, db, "SELECT llm_context_free();") != 0) goto fail;
    if (exec_expect_ok(env, db, "SELECT llm_model_free();") != 0) goto fail;

    sqlite3_close(db);
    return assert_sqlite_memory_clean("document_ingestion_flow", env);

fail:
    if (db) sqlite3_close(db);
    return 1;
}

static int test_llm_sampler_roundtrip(const test_env *env) {
    sqlite3 *db = NULL;
    if (open_db_and_load(env, &db) != SQLITE_OK) {
        return 1;
    }

    const char *model = env->model_path ? env->model_path : DEFAULT_MODEL_PATH;
    char sqlbuf[512];
    snprintf(sqlbuf, sizeof(sqlbuf), "SELECT llm_model_load('%s');", model);
    if (exec_expect_ok(env, db, sqlbuf) != 0) goto fail;
    if (exec_expect_ok(env, db, "SELECT llm_context_create_textgen('context_size=1024');") != 0) goto fail;
    if (exec_expect_ok(env, db, "SELECT llm_sampler_create();") != 0) goto fail;
    if (exec_expect_ok(env, db, "SELECT llm_sampler_init_top_k(20);") != 0) goto fail;
    if (exec_expect_ok(env, db, "SELECT llm_sampler_init_temp(0.7);") != 0) goto fail;
    // dist or greedy step must be added at the end of the sampler chain
    // otherwise the llm_chat_respond function will crash
    if (exec_expect_ok(env, db, "SELECT llm_sampler_init_dist();") != 0) goto fail;
    if (exec_expect_ok(env, db, "SELECT llm_chat_create();") != 0) goto fail;
    if (exec_expect_ok(env, db, "SELECT llm_chat_respond('Say hello');") != 0) goto fail;
    if (exec_expect_ok(env, db, "SELECT llm_chat_free();") != 0) goto fail;
    if (exec_expect_ok(env, db, "SELECT llm_sampler_free();") != 0) goto fail;
    if (exec_expect_ok(env, db, "SELECT llm_context_free();") != 0) goto fail;
    if (exec_expect_ok(env, db, "SELECT llm_model_free();") != 0) goto fail;

    sqlite3_close(db);
    return assert_sqlite_memory_clean("llm_sampler_roundtrip", env);

fail:
    if (db) sqlite3_close(db);
    return 1;
}

static int test_dual_connection_roles(const test_env *env) {
    sqlite3 *db_embed = NULL;
    sqlite3 *db_text = NULL;
    const char *model = env->model_path ? env->model_path : DEFAULT_MODEL_PATH;
    char sqlbuf[512];

    if (open_db_and_load(env, &db_embed) != SQLITE_OK) goto fail;
    if (open_db_and_load(env, &db_text) != SQLITE_OK) goto fail;

    snprintf(sqlbuf, sizeof(sqlbuf), "SELECT llm_model_load('%s');", model);
    if (exec_expect_ok(env, db_embed, sqlbuf) != 0) goto fail;
    if (exec_expect_ok(env, db_embed, "SELECT llm_context_create_embedding('context_size=512,embedding_type=UINT8');") != 0) goto fail;
    if (exec_expect_ok(env, db_embed, "SELECT llm_embed_generate('dual connection embedding text');") != 0) goto fail;
    if (exec_expect_ok(env, db_embed, "SELECT llm_context_free();") != 0) goto fail;
    if (exec_expect_ok(env, db_embed, "SELECT llm_model_free();") != 0) goto fail;

    if (exec_expect_ok(env, db_text, sqlbuf) != 0) goto fail;
    if (exec_expect_ok(env, db_text, "SELECT llm_context_create_chat('context_size=512');") != 0) goto fail;
    if (exec_expect_ok(env, db_text, "SELECT llm_chat_create();") != 0) goto fail;
    if (exec_expect_ok(env, db_text, "SELECT llm_chat_respond('Hello from text connection');") != 0) goto fail;
    if (exec_expect_ok(env, db_text, "SELECT llm_chat_free();") != 0) goto fail;
    if (exec_expect_ok(env, db_text, "SELECT llm_context_free();") != 0) goto fail;
    if (exec_expect_ok(env, db_text, "SELECT llm_model_free();") != 0) goto fail;

    sqlite3_close(db_embed);
    sqlite3_close(db_text);
    return assert_sqlite_memory_clean("dual_connection_roles", env);

fail:
    if (db_embed) sqlite3_close(db_embed);
    if (db_text) sqlite3_close(db_text);
    return 1;
}

static int test_concurrent_connections_independent(const test_env *env) {
    sqlite3 *db_one = NULL;
    sqlite3 *db_two = NULL;
    const char *model = env->model_path ? env->model_path : DEFAULT_MODEL_PATH;
    char sqlbuf[512];

    if (open_db_and_load(env, &db_one) != SQLITE_OK) goto fail;
    if (open_db_and_load(env, &db_two) != SQLITE_OK) goto fail;

    snprintf(sqlbuf, sizeof(sqlbuf), "SELECT llm_model_load('%s');", model);
    if (exec_expect_ok(env, db_one, sqlbuf) != 0) goto fail;
    if (exec_expect_ok(env, db_one, "SELECT llm_context_create_embedding('context_size=384,embedding_type=UINT8');") != 0) goto fail;
    if (exec_expect_ok(env, db_one, "SELECT llm_embed_generate('first connection payload');") != 0) goto fail;

    if (exec_expect_ok(env, db_two, sqlbuf) != 0) goto fail;
    if (exec_expect_ok(env, db_two, "SELECT llm_context_create_embedding('context_size=384,embedding_type=UINT8');") != 0) goto fail;
    if (exec_expect_ok(env, db_two, "SELECT llm_embed_generate('second connection payload');") != 0) goto fail;

    if (exec_expect_ok(env, db_one, "SELECT llm_context_free();") != 0) goto fail;
    if (exec_expect_ok(env, db_one, "SELECT llm_model_free();") != 0) goto fail;
    sqlite3_close(db_one);
    db_one = NULL;

    if (exec_expect_ok(env, db_two, "SELECT llm_embed_generate('still active after peer closed');") != 0) goto fail;
    if (exec_expect_ok(env, db_two, "SELECT llm_context_free();") != 0) goto fail;
    if (exec_expect_ok(env, db_two, "SELECT llm_model_free();") != 0) goto fail;
    sqlite3_close(db_two);

    return assert_sqlite_memory_clean("concurrent_connections_independent", env);

fail:
    if (db_one) sqlite3_close(db_one);
    if (db_two) sqlite3_close(db_two);
    return 1;
}

static int test_llm_model_load_error_recovery(const test_env *env) {
    sqlite3 *db = NULL;
    if (open_db_and_load(env, &db) != SQLITE_OK) {
        return 1;
    }

    if (exec_expect_error(env, db, "SELECT llm_model_load('/path/that/does/not/exist.gguf');", "Unable to load model") != 0) {
        sqlite3_close(db);
        return 1;
    }

    const char *model = env->model_path ? env->model_path : DEFAULT_MODEL_PATH;
    char sqlbuf[512];
    snprintf(sqlbuf, sizeof(sqlbuf), "SELECT llm_model_load('%s');", model);
    if (exec_expect_ok(env, db, sqlbuf) != 0) goto fail;
    if (exec_expect_ok(env, db, "SELECT llm_context_create('context_size=256');") != 0) goto fail;
    if (exec_expect_ok(env, db, "SELECT llm_context_free();") != 0) goto fail;
    if (exec_expect_ok(env, db, "SELECT llm_model_free();") != 0) goto fail;

    sqlite3_close(db);
    return assert_sqlite_memory_clean("llm_model_load_error_recovery", env);

fail:
    if (db) sqlite3_close(db);
    return 1;
}

static int test_ai_logging_table(const test_env *env) {
    sqlite3 *db = NULL;
    if (open_db_and_load(env, &db) != SQLITE_OK) {
        return 1;
    }

    const char *model = env->model_path ? env->model_path : DEFAULT_MODEL_PATH;
    char sqlbuf[512];
    // context_size option is ignored during model load, which triggers a warning log entry
    snprintf(sqlbuf, sizeof(sqlbuf), "SELECT llm_model_load('%s','context_size=256');", model);
    if (exec_expect_ok(env, db, sqlbuf) != 0) goto fail;

    int rows = 0;
    if (select_single_int(env, db, "SELECT COUNT(*) FROM ai_log;", &rows) != 0) goto fail;
    if (rows <= 0) {
        fprintf(stderr, "[ai_logging_table] expected ai_log entries but found %d\n", rows);
        goto fail;
    }

    if (exec_expect_ok(env, db, "SELECT llm_model_free();") != 0) goto fail;
    sqlite3_close(db);
    return assert_sqlite_memory_clean("ai_logging_table", env);

fail:
    if (db) sqlite3_close(db);
    return 1;
}

// Test that input larger than n_ctx is gracefully truncated (not an error)
static int test_llm_embed_input_too_large(const test_env *env) {
    sqlite3 *db = NULL;
    sqlite3_stmt *stmt = NULL;
    if (open_db_and_load(env, &db) != SQLITE_OK) {
        return 1;
    }

    const char *model = env->model_path ? env->model_path : DEFAULT_MODEL_PATH;
    char sqlbuf[512];
    snprintf(sqlbuf, sizeof(sqlbuf), "SELECT llm_model_load('%s');", model);
    if (exec_expect_ok(env, db, sqlbuf) != 0) goto fail;
    if (exec_expect_ok(env, db, "SELECT llm_context_create_embedding('context_size=16,embedding_type=UINT8');") != 0) goto fail;

    // generate a payload that will tokenize to more than 16 tokens
    size_t payload_len = 4096;
    char *payload = (char *)malloc(payload_len + 1);
    if (!payload) goto fail;
    memset(payload, 'A', payload_len);
    payload[payload_len] = '\0';

    // should succeed (truncated), not error
    char *sql = sqlite3_mprintf("SELECT llm_embed_generate('%q');", payload);
    free(payload);
    if (!sql) goto fail;

    if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) != SQLITE_OK) {
        fprintf(stderr, "prepare failed: %s\n", sqlite3_errmsg(db));
        sqlite3_free(sql);
        goto fail;
    }
    sqlite3_free(sql);

    int rc = sqlite3_step(stmt);
    if (rc != SQLITE_ROW) {
        fprintf(stderr, "Expected a row from truncated embed, got rc=%d: %s\n", rc, sqlite3_errmsg(db));
        sqlite3_finalize(stmt);
        goto fail;
    }
    const void *blob = sqlite3_column_blob(stmt, 0);
    int blob_bytes = sqlite3_column_bytes(stmt, 0);
    if (blob == NULL || blob_bytes <= 0) {
        fprintf(stderr, "Truncated embedding blob is empty (bytes=%d)\n", blob_bytes);
        sqlite3_finalize(stmt);
        goto fail;
    }
    sqlite3_finalize(stmt);

    if (exec_expect_ok(env, db, "SELECT llm_context_free();") != 0) goto fail;
    if (exec_expect_ok(env, db, "SELECT llm_model_free();") != 0) goto fail;
    sqlite3_close(db);
    return assert_sqlite_memory_clean("llm_embed_input_too_large", env);

fail:
    if (db) sqlite3_close(db);
    return 1;
}

// Test that n_ctx > n_ctx_train does not crash (position embedding overflow protection)
static int test_llm_embed_nctx_exceeds_train(const test_env *env) {
    sqlite3 *db = NULL;
    sqlite3_stmt *stmt = NULL;
    if (open_db_and_load(env, &db) != SQLITE_OK) return 1;

    const char *model = env->model_path ? env->model_path : DEFAULT_MODEL_PATH;
    char sqlbuf[512];
    snprintf(sqlbuf, sizeof(sqlbuf), "SELECT llm_model_load('%s');", model);
    if (exec_expect_ok(env, db, sqlbuf) != 0) goto fail;

    // deliberately set n_ctx much larger than model's n_ctx_train
    if (exec_expect_ok(env, db, "SELECT llm_context_create_embedding('context_size=999999,embedding_type=UINT8');") != 0) goto fail;

    // this should NOT crash — n_ctx is clamped to n_ctx_train internally
    const char *embed_sql = "SELECT llm_embed_generate('test text for oversized context');";
    if (sqlite3_prepare_v2(db, embed_sql, -1, &stmt, NULL) != SQLITE_OK) {
        fprintf(stderr, "prepare failed: %s\n", sqlite3_errmsg(db));
        goto fail;
    }
    int rc = sqlite3_step(stmt);
    if (rc != SQLITE_ROW) {
        fprintf(stderr, "Expected a row, got rc=%d: %s\n", rc, sqlite3_errmsg(db));
        sqlite3_finalize(stmt);
        goto fail;
    }
    const void *blob = sqlite3_column_blob(stmt, 0);
    int blob_bytes = sqlite3_column_bytes(stmt, 0);
    if (blob == NULL || blob_bytes <= 0) {
        fprintf(stderr, "Embedding blob is empty (bytes=%d)\n", blob_bytes);
        sqlite3_finalize(stmt);
        goto fail;
    }
    sqlite3_finalize(stmt);

    if (exec_expect_ok(env, db, "SELECT llm_context_free();") != 0) goto fail;
    if (exec_expect_ok(env, db, "SELECT llm_model_free();") != 0) goto fail;
    sqlite3_close(db);
    return assert_sqlite_memory_clean("llm_embed_nctx_exceeds_train", env);

fail:
    if (db) sqlite3_close(db);
    return 1;
}

// Test that max_tokens limit is enforced even when the input would fit in context
static int test_llm_embed_max_tokens_limit(const test_env *env) {
    sqlite3 *db = NULL;
    if (open_db_and_load(env, &db) != SQLITE_OK) return 1;

    const char *model = env->model_path ? env->model_path : DEFAULT_MODEL_PATH;
    char sqlbuf[512];
    snprintf(sqlbuf, sizeof(sqlbuf), "SELECT llm_model_load('%s');", model);
    if (exec_expect_ok(env, db, sqlbuf) != 0) goto fail;
    // set max_tokens=2 so even short text exceeds it
    if (exec_expect_ok(env, db, "SELECT llm_context_create_embedding('embedding_type=UINT8,max_tokens=2');") != 0) goto fail;

    // "hello world" tokenizes to more than 2 tokens
    if (exec_expect_error(env, db, "SELECT llm_embed_generate('hello world this is a test');", "exceeds max allowed") != 0) goto fail;

    if (exec_expect_ok(env, db, "SELECT llm_context_free();") != 0) goto fail;
    if (exec_expect_ok(env, db, "SELECT llm_model_free();") != 0) goto fail;
    sqlite3_close(db);
    return assert_sqlite_memory_clean("llm_embed_max_tokens_limit", env);

fail:
    if (db) sqlite3_close(db);
    return 1;
}

// Test that calling llm_embed_generate multiple times in a row produces valid results each time
static int test_llm_embed_repeated_calls(const test_env *env) {
    sqlite3 *db = NULL;
    sqlite3_stmt *stmt = NULL;
    if (open_db_and_load(env, &db) != SQLITE_OK) return 1;

    const char *model = env->model_path ? env->model_path : DEFAULT_MODEL_PATH;
    char sqlbuf[512];
    snprintf(sqlbuf, sizeof(sqlbuf), "SELECT llm_model_load('%s');", model);
    if (exec_expect_ok(env, db, sqlbuf) != 0) goto fail;
    if (exec_expect_ok(env, db, "SELECT llm_context_create_embedding('embedding_type=UINT8');") != 0) goto fail;

    const char *texts[] = {"first call", "second call", "third call"};
    for (int t = 0; t < 3; t++) {
        char *sql = sqlite3_mprintf("SELECT llm_embed_generate('%q');", texts[t]);
        if (!sql) goto fail;
        if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) != SQLITE_OK) {
            fprintf(stderr, "prepare failed on call %d: %s\n", t, sqlite3_errmsg(db));
            sqlite3_free(sql);
            goto fail;
        }
        sqlite3_free(sql);
        int rc = sqlite3_step(stmt);
        if (rc != SQLITE_ROW) {
            fprintf(stderr, "Expected row on call %d, got rc=%d\n", t, rc);
            sqlite3_finalize(stmt);
            goto fail;
        }
        const void *blob = sqlite3_column_blob(stmt, 0);
        int blob_bytes = sqlite3_column_bytes(stmt, 0);
        if (blob == NULL || blob_bytes <= 0) {
            fprintf(stderr, "Embedding blob empty on call %d (bytes=%d)\n", t, blob_bytes);
            sqlite3_finalize(stmt);
            goto fail;
        }
        sqlite3_finalize(stmt);
        stmt = NULL;
    }

    if (exec_expect_ok(env, db, "SELECT llm_context_free();") != 0) goto fail;
    if (exec_expect_ok(env, db, "SELECT llm_model_free();") != 0) goto fail;
    sqlite3_close(db);
    return assert_sqlite_memory_clean("llm_embed_repeated_calls", env);

fail:
    if (stmt) sqlite3_finalize(stmt);
    if (db) sqlite3_close(db);
    return 1;
}

// Test that empty input returns NULL without crashing
static int test_llm_embed_empty_input(const test_env *env) {
    sqlite3 *db = NULL;
    sqlite3_stmt *stmt = NULL;
    if (open_db_and_load(env, &db) != SQLITE_OK) return 1;

    const char *model = env->model_path ? env->model_path : DEFAULT_MODEL_PATH;
    char sqlbuf[512];
    snprintf(sqlbuf, sizeof(sqlbuf), "SELECT llm_model_load('%s');", model);
    if (exec_expect_ok(env, db, sqlbuf) != 0) goto fail;
    if (exec_expect_ok(env, db, "SELECT llm_context_create_embedding('embedding_type=UINT8');") != 0) goto fail;

    // empty string should return NULL, not crash
    if (sqlite3_prepare_v2(db, "SELECT llm_embed_generate('');", -1, &stmt, NULL) != SQLITE_OK) {
        fprintf(stderr, "prepare failed: %s\n", sqlite3_errmsg(db));
        goto fail;
    }
    if (sqlite3_step(stmt) != SQLITE_ROW) {
        fprintf(stderr, "Expected a row\n");
        sqlite3_finalize(stmt);
        goto fail;
    }
    if (sqlite3_column_type(stmt, 0) != SQLITE_NULL) {
        fprintf(stderr, "Expected NULL for empty input, got type=%d\n", sqlite3_column_type(stmt, 0));
        sqlite3_finalize(stmt);
        goto fail;
    }
    sqlite3_finalize(stmt);

    if (exec_expect_ok(env, db, "SELECT llm_context_free();") != 0) goto fail;
    if (exec_expect_ok(env, db, "SELECT llm_model_free();") != 0) goto fail;
    sqlite3_close(db);
    return assert_sqlite_memory_clean("llm_embed_empty_input", env);

fail:
    if (db) sqlite3_close(db);
    return 1;
}

static int query_chat_response(const test_env *env, sqlite3 *db, const char *question, char *response, size_t response_len) {
    char sqlbuf[512];
    snprintf(sqlbuf, sizeof(sqlbuf), "SELECT llm_chat_respond('%s');", question);
    return exec_query_text(env, db, sqlbuf, response, response_len);
}

static int test_chat_system_prompt_new_chat(const test_env *env) {
    sqlite3 *db = NULL;
    bool model_loaded = false;
    bool context_created = false;
    bool chat_created = false;
    int status = 1;

    if (open_db_and_load(env, &db) != SQLITE_OK) {
        goto done;
    }

    const char *model = env->model_path ? env->model_path : DEFAULT_MODEL_PATH;
    char sqlbuf[512];
    snprintf(sqlbuf, sizeof(sqlbuf), "SELECT llm_model_load('%s');", model);
    if (exec_expect_ok(env, db, sqlbuf) != 0) goto done;
    model_loaded = true;

    if (exec_expect_ok(env, db, "SELECT llm_context_create('context_size=1000');") != 0) goto done;
    context_created = true;

    if (exec_expect_ok(env, db, "SELECT llm_chat_create();") != 0) goto done;
    chat_created = true;

    const char *system_prompt = "Always reply with lowercase answers.";
    snprintf(sqlbuf, sizeof(sqlbuf), "SELECT llm_chat_system_prompt('%s');", system_prompt);
    if (exec_expect_ok(env, db, sqlbuf) != 0) goto done;

    bool is_null = false;
    char buffer[4096];
    if (query_system_prompt(env, db, buffer, sizeof(buffer), &is_null) != 0) goto done;
    if (is_null || strcmp(buffer, system_prompt) != 0) {
        fprintf(stderr, "[test_chat_system_prompt_new_chat] expected '%s' but got: %s\n", system_prompt, buffer);
        goto done;
    }

    if (exec_expect_ok(env, db, "SELECT llm_chat_save();") != 0) goto done;

    ai_chat_message_row rows[4];
    int count = 0;
    if (fetch_ai_chat_messages(env, db, rows, 4, &count) != 0) goto done;
    if (count != 1) {
        fprintf(stderr, "[test_chat_system_prompt_new_chat] expected 1 message row, got %d\n", count);
        goto done;
    }
    if (strcmp(rows[0].role, "system") != 0) {
        fprintf(stderr, "[test_chat_system_prompt_new_chat] expected system role, got %s\n", rows[0].role);
        goto done;
    }
    if (strcmp(rows[0].content, system_prompt) != 0) {
        fprintf(stderr, "[test_chat_system_prompt_new_chat] expected content '%s' but got '%s'\n", system_prompt, rows[0].content);
        goto done;
    }

    status = 0;

done:
    if (chat_created) exec_expect_ok(env, db, "SELECT llm_chat_free();");
    if (context_created) exec_expect_ok(env, db, "SELECT llm_context_free();");
    if (model_loaded) exec_expect_ok(env, db, "SELECT llm_model_free();");
    if (db) sqlite3_close(db);
    if (status == 0) status = assert_sqlite_memory_clean("llm_context_size_errors", env);
    return status;
}

static int test_chat_system_prompt_replace_previous_prompt(const test_env *env) {
    sqlite3 *db = NULL;
    bool model_loaded = false;
    bool context_created = false;
    bool chat_created = false;
    int status = 1;

    if (open_db_and_load(env, &db) != SQLITE_OK) {
        goto done;
    }

    const char *model = env->model_path ? env->model_path : DEFAULT_MODEL_PATH;
    char sqlbuf[512];
    snprintf(sqlbuf, sizeof(sqlbuf), "SELECT llm_model_load('%s');", model);
    if (exec_expect_ok(env, db, sqlbuf) != 0) goto done;
    model_loaded = true;

    if (exec_expect_ok(env, db, "SELECT llm_context_create('context_size=1000');") != 0) goto done;
    context_created = true;

    if (exec_expect_ok(env, db, "SELECT llm_chat_create();") != 0) goto done;
    chat_created = true;

    const char *first_prompt = "Always confirm questions.";
    snprintf(sqlbuf, sizeof(sqlbuf), "SELECT llm_chat_system_prompt('%s');", first_prompt);
    if (exec_expect_ok(env, db, sqlbuf) != 0) goto done;

    const char *replacement_prompt = "Always decline questions.";
    snprintf(sqlbuf, sizeof(sqlbuf), "SELECT llm_chat_system_prompt('%s');", replacement_prompt);
    if (exec_expect_ok(env, db, sqlbuf) != 0) goto done;

    bool is_null = false;
    char buffer[4096];
    if (query_system_prompt(env, db, buffer, sizeof(buffer), &is_null) != 0) goto done;
    if (is_null || strcmp(buffer, replacement_prompt) != 0) {
        fprintf(stderr, "[test_chat_system_prompt_replace_previous_prompt] expected '%s' but got: %s\n", replacement_prompt, buffer);
        goto done;
    }

    if (exec_expect_ok(env, db, "SELECT llm_chat_save();") != 0) goto done;

    ai_chat_message_row rows[4];
    int count = 0;
    if (fetch_ai_chat_messages(env, db, rows, 4, &count) != 0) goto done;
    if (count != 1) {
        fprintf(stderr, "[test_chat_system_prompt_replace_previous_prompt] expected 1 message row, got %d\n", count);
        goto done;
    }
    if (strcmp(rows[0].content, replacement_prompt) != 0) {
        fprintf(stderr, "[test_chat_system_prompt_replace_previous_prompt] expected '%s' but got '%s'\n", replacement_prompt, rows[0].content);
        goto done;
    }

    status = 0;

done:
    if (chat_created) exec_expect_ok(env, db, "SELECT llm_chat_free();");
    if (context_created) exec_expect_ok(env, db, "SELECT llm_context_free();");
    if (model_loaded) exec_expect_ok(env, db, "SELECT llm_model_free();");
    if (db) sqlite3_close(db);
    if (status == 0) status = assert_sqlite_memory_clean("llm_context_size_errors", env);
    return status;
}

static int test_chat_system_prompt_after_first_response(const test_env *env) {
    sqlite3 *db = NULL;
    bool model_loaded = false;
    bool context_created = false;
    bool chat_created = false;
    int status = 1;

    if (open_db_and_load(env, &db) != SQLITE_OK) {
        goto done;
    }

    const char *model = env->model_path ? env->model_path : DEFAULT_MODEL_PATH;
    char sqlbuf[512];
    snprintf(sqlbuf, sizeof(sqlbuf), "SELECT llm_model_load('%s');", model);
    if (exec_expect_ok(env, db, sqlbuf) != 0) goto done;
    model_loaded = true;

    if (exec_expect_ok(env, db, "SELECT llm_context_create('context_size=1000');") != 0) goto done;
    context_created = true;

    if (exec_expect_ok(env, db, "SELECT llm_chat_create();") != 0) goto done;
    chat_created = true;

    const char *user_question = "Reply to this ping.";
    char response[4096];
    if (query_chat_response(env, db, user_question, response, sizeof(response)) != 0) goto done;
    if (response[0] == '\0') {
        fprintf(stderr, "[test_chat_system_prompt_after_first_response] expected model response for '%s'\n", user_question);
        goto done;
    }

    const char *system_prompt = "Only answer with short confirmations.";
    snprintf(sqlbuf, sizeof(sqlbuf), "SELECT llm_chat_system_prompt('%s');", system_prompt);
    if (exec_expect_ok(env, db, sqlbuf) != 0) goto done;

    bool is_null = false;
    char buffer[4096];
    if (query_system_prompt(env, db, buffer, sizeof(buffer), &is_null) != 0) goto done;
    if (is_null || strcmp(buffer, system_prompt) != 0) {
        fprintf(stderr, "[test_chat_system_prompt_after_first_response] expected '%s' but got: %s\n", system_prompt, buffer);
        goto done;
    }

    if (exec_expect_ok(env, db, "SELECT llm_chat_save();") != 0) goto done;

    ai_chat_message_row rows[8];
    int count = 0;
    if (fetch_ai_chat_messages(env, db, rows, 8, &count) != 0) goto done;
    if (count < 3) {
        fprintf(stderr, "[test_chat_system_prompt_after_first_response] expected at least 3 rows, got %d\n", count);
        goto done;
    }
    if (!(rows[0].id < rows[1].id && rows[1].id < rows[2].id)) {
        fprintf(stderr, "[test_chat_system_prompt_after_first_response] expected ascending ids but found %d, %d, %d\n",
                rows[0].id, rows[1].id, rows[2].id);
        goto done;
    }
    if (strcmp(rows[0].role, "system") != 0 || strcmp(rows[0].content, system_prompt) != 0) {
        fprintf(stderr, "[test_chat_system_prompt_after_first_response] system row mismatch (%s, %s)\n", rows[0].role, rows[0].content);
        goto done;
    }
    if (strcmp(rows[1].role, "user") != 0 || strcmp(rows[1].content, user_question) != 0) {
        fprintf(stderr, "[test_chat_system_prompt_after_first_response] user row mismatch (%s, %s)\n", rows[1].role, rows[1].content);
        goto done;
    }
    if (strcmp(rows[2].role, "assistant") != 0 || rows[2].content[0] == '\0') {
        fprintf(stderr, "[test_chat_system_prompt_after_first_response] assistant row mismatch (%s, %s)\n", rows[2].role, rows[2].content);
        goto done;
    }

    status = 0;

done:
    if (chat_created) exec_expect_ok(env, db, "SELECT llm_chat_free();");
    if (context_created) exec_expect_ok(env, db, "SELECT llm_context_free();");
    if (model_loaded) exec_expect_ok(env, db, "SELECT llm_model_free();");
    if (db) sqlite3_close(db);
    if (status == 0) status = assert_sqlite_memory_clean("llm_context_size_errors", env);
    return status;
}

// Test chat create/free can be called multiple times without crash (dangling pointer fix)
static int test_chat_create_free_cycle(const test_env *env) {
    sqlite3 *db = NULL;
    if (open_db_and_load(env, &db) != SQLITE_OK) return 1;

    const char *model = env->model_path ? env->model_path : DEFAULT_MODEL_PATH;
    char sqlbuf[512];
    snprintf(sqlbuf, sizeof(sqlbuf), "SELECT llm_model_load('%s');", model);
    if (exec_expect_ok(env, db, sqlbuf) != 0) goto fail;
    if (exec_expect_ok(env, db, "SELECT llm_context_create('context_size=512');") != 0) goto fail;

    // create and free chat multiple times to test for dangling pointers
    for (int i = 0; i < 3; i++) {
        if (exec_expect_ok(env, db, "SELECT llm_chat_create();") != 0) goto fail;
        if (exec_expect_ok(env, db, "SELECT llm_chat_free();") != 0) goto fail;
    }

    // after cycles, should still be able to create and use chat
    if (exec_expect_ok(env, db, "SELECT llm_chat_create();") != 0) goto fail;
    if (exec_expect_ok(env, db, "SELECT llm_chat_respond('Hello');") != 0) goto fail;
    if (exec_expect_ok(env, db, "SELECT llm_chat_free();") != 0) goto fail;
    if (exec_expect_ok(env, db, "SELECT llm_context_free();") != 0) goto fail;
    if (exec_expect_ok(env, db, "SELECT llm_model_free();") != 0) goto fail;

    sqlite3_close(db);
    return assert_sqlite_memory_clean("chat_create_free_cycle", env);

fail:
    if (db) sqlite3_close(db);
    return 1;
}

// Test that chat_create re-creates a fresh session after previous chat had messages
static int test_chat_recreate_after_conversation(const test_env *env) {
    sqlite3 *db = NULL;
    if (open_db_and_load(env, &db) != SQLITE_OK) return 1;

    const char *model = env->model_path ? env->model_path : DEFAULT_MODEL_PATH;
    char sqlbuf[512];
    snprintf(sqlbuf, sizeof(sqlbuf), "SELECT llm_model_load('%s');", model);
    if (exec_expect_ok(env, db, sqlbuf) != 0) goto fail;
    if (exec_expect_ok(env, db, "SELECT llm_context_create('context_size=1000');") != 0) goto fail;

    // first chat session
    if (exec_expect_ok(env, db, "SELECT llm_chat_create();") != 0) goto fail;
    if (exec_expect_ok(env, db, "SELECT llm_chat_respond('First session message');") != 0) goto fail;
    if (exec_expect_ok(env, db, "SELECT llm_chat_free();") != 0) goto fail;

    // second chat session — should be a clean slate
    if (exec_expect_ok(env, db, "SELECT llm_chat_create();") != 0) goto fail;

    // system prompt should be NULL for fresh chat
    bool is_null = false;
    char buffer[256];
    if (query_system_prompt(env, db, buffer, sizeof(buffer), &is_null) != 0) goto fail;
    if (!is_null) {
        fprintf(stderr, "[chat_recreate] expected NULL system prompt for fresh chat, got '%s'\n", buffer);
        goto fail;
    }

    if (exec_expect_ok(env, db, "SELECT llm_chat_respond('Second session message');") != 0) goto fail;
    if (exec_expect_ok(env, db, "SELECT llm_chat_free();") != 0) goto fail;
    if (exec_expect_ok(env, db, "SELECT llm_context_free();") != 0) goto fail;
    if (exec_expect_ok(env, db, "SELECT llm_model_free();") != 0) goto fail;

    sqlite3_close(db);
    return assert_sqlite_memory_clean("chat_recreate_after_conversation", env);

fail:
    if (db) sqlite3_close(db);
    return 1;
}

// Test chat vtab streaming produces tokens and saves response correctly across turns
static int test_chat_vtab_multi_turn(const test_env *env) {
    sqlite3 *db = NULL;
    if (open_db_and_load(env, &db) != SQLITE_OK) return 1;

    const char *model = env->model_path ? env->model_path : DEFAULT_MODEL_PATH;
    char sqlbuf[512];
    snprintf(sqlbuf, sizeof(sqlbuf), "SELECT llm_model_load('%s');", model);
    if (exec_expect_ok(env, db, sqlbuf) != 0) goto fail;
    if (exec_expect_ok(env, db, "SELECT llm_context_create('context_size=1000');") != 0) goto fail;
    if (exec_expect_ok(env, db, "SELECT llm_chat_create();") != 0) goto fail;

    // first turn via vtab
    int rows = 0;
    if (exec_select_rows(env, db, "SELECT * FROM llm_chat('What is 1+1?');", &rows) != 0) goto fail;
    if (rows <= 0) {
        fprintf(stderr, "[chat_vtab_multi_turn] first turn: expected rows but got %d\n", rows);
        goto fail;
    }

    // second turn via vtab — tests that prev_len is correctly maintained
    rows = 0;
    if (exec_select_rows(env, db, "SELECT * FROM llm_chat('And 2+2?');", &rows) != 0) goto fail;
    if (rows <= 0) {
        fprintf(stderr, "[chat_vtab_multi_turn] second turn: expected rows but got %d\n", rows);
        goto fail;
    }

    // third turn via respond (mixing vtab and respond)
    if (exec_expect_ok(env, db, "SELECT llm_chat_respond('Thanks');") != 0) goto fail;

    if (exec_expect_ok(env, db, "SELECT llm_chat_free();") != 0) goto fail;
    if (exec_expect_ok(env, db, "SELECT llm_context_free();") != 0) goto fail;
    if (exec_expect_ok(env, db, "SELECT llm_model_free();") != 0) goto fail;

    sqlite3_close(db);
    return assert_sqlite_memory_clean("chat_vtab_multi_turn", env);

fail:
    if (db) sqlite3_close(db);
    return 1;
}

// Test chat save and restore round-trip
static int test_chat_save_restore_roundtrip(const test_env *env) {
    sqlite3 *db = NULL;
    if (open_db_and_load(env, &db) != SQLITE_OK) return 1;

    const char *model = env->model_path ? env->model_path : DEFAULT_MODEL_PATH;
    char sqlbuf[512];
    snprintf(sqlbuf, sizeof(sqlbuf), "SELECT llm_model_load('%s');", model);
    if (exec_expect_ok(env, db, sqlbuf) != 0) goto fail;
    if (exec_expect_ok(env, db, "SELECT llm_context_create('context_size=1000');") != 0) goto fail;

    // create chat, set system prompt, send a message, save
    if (exec_expect_ok(env, db, "SELECT llm_chat_create();") != 0) goto fail;
    if (exec_expect_ok(env, db, "SELECT llm_chat_system_prompt('Be concise.');") != 0) goto fail;
    if (exec_expect_ok(env, db, "SELECT llm_chat_respond('Hi');") != 0) goto fail;

    // save the chat and capture UUID
    char uuid[128] = {0};
    if (exec_query_text(env, db, "SELECT llm_chat_save();", uuid, sizeof(uuid)) != 0) goto fail;
    if (uuid[0] == '\0') {
        fprintf(stderr, "[chat_save_restore] save returned empty UUID\n");
        goto fail;
    }

    // verify messages were saved
    int msg_count = 0;
    if (fetch_ai_chat_messages(env, db, NULL, 0, &msg_count) != 0) goto fail;
    if (msg_count < 3) { // system + user + assistant
        fprintf(stderr, "[chat_save_restore] expected at least 3 saved messages, got %d\n", msg_count);
        goto fail;
    }

    // free chat and restore it
    if (exec_expect_ok(env, db, "SELECT llm_chat_free();") != 0) goto fail;

    snprintf(sqlbuf, sizeof(sqlbuf), "SELECT llm_chat_restore('%s');", uuid);
    if (exec_expect_ok(env, db, sqlbuf) != 0) goto fail;

    // verify system prompt was restored
    bool is_null = false;
    char buffer[256];
    if (query_system_prompt(env, db, buffer, sizeof(buffer), &is_null) != 0) goto fail;
    if (is_null || strcmp(buffer, "Be concise.") != 0) {
        fprintf(stderr, "[chat_save_restore] system prompt mismatch after restore: '%s' (null=%d)\n", buffer, is_null);
        goto fail;
    }

    if (exec_expect_ok(env, db, "SELECT llm_chat_free();") != 0) goto fail;
    if (exec_expect_ok(env, db, "SELECT llm_context_free();") != 0) goto fail;
    if (exec_expect_ok(env, db, "SELECT llm_model_free();") != 0) goto fail;

    sqlite3_close(db);
    return assert_sqlite_memory_clean("chat_save_restore_roundtrip", env);

fail:
    if (db) sqlite3_close(db);
    return 1;
}

// Test that llm_chat_system_prompt(NULL) clears the system prompt
static int test_chat_system_prompt_clear(const test_env *env) {
    sqlite3 *db = NULL;
    if (open_db_and_load(env, &db) != SQLITE_OK) return 1;

    const char *model = env->model_path ? env->model_path : DEFAULT_MODEL_PATH;
    char sqlbuf[512];
    snprintf(sqlbuf, sizeof(sqlbuf), "SELECT llm_model_load('%s');", model);
    if (exec_expect_ok(env, db, sqlbuf) != 0) goto fail;
    if (exec_expect_ok(env, db, "SELECT llm_context_create('context_size=512');") != 0) goto fail;
    if (exec_expect_ok(env, db, "SELECT llm_chat_create();") != 0) goto fail;

    // set a system prompt
    if (exec_expect_ok(env, db, "SELECT llm_chat_system_prompt('Be helpful.');") != 0) goto fail;

    // verify it was set
    bool is_null = false;
    char buffer[256];
    if (query_system_prompt(env, db, buffer, sizeof(buffer), &is_null) != 0) goto fail;
    if (is_null) {
        fprintf(stderr, "[chat_system_prompt_clear] expected system prompt to be set\n");
        goto fail;
    }

    // clear it with NULL
    if (exec_expect_ok(env, db, "SELECT llm_chat_system_prompt(NULL);") != 0) goto fail;

    // verify it reads back as NULL (empty content treated as NULL)
    if (query_system_prompt(env, db, buffer, sizeof(buffer), &is_null) != 0) goto fail;
    if (!is_null) {
        fprintf(stderr, "[chat_system_prompt_clear] expected NULL after clearing, got '%s'\n", buffer);
        goto fail;
    }

    if (exec_expect_ok(env, db, "SELECT llm_chat_free();") != 0) goto fail;
    if (exec_expect_ok(env, db, "SELECT llm_context_free();") != 0) goto fail;
    if (exec_expect_ok(env, db, "SELECT llm_model_free();") != 0) goto fail;

    sqlite3_close(db);
    return assert_sqlite_memory_clean("chat_system_prompt_clear", env);

fail:
    if (db) sqlite3_close(db);
    return 1;
}

// Test text generation with an instruct model (chat template auto-wrap)
static int test_text_generate_with_eog(const test_env *env) {
    sqlite3 *db = NULL;
    if (open_db_and_load(env, &db) != SQLITE_OK) return 1;

    const char *model = env->model_path ? env->model_path : DEFAULT_MODEL_PATH;
    char sqlbuf[512];
    snprintf(sqlbuf, sizeof(sqlbuf), "SELECT llm_model_load('%s');", model);
    if (exec_expect_ok(env, db, sqlbuf) != 0) goto fail;
    if (exec_expect_ok(env, db, "SELECT llm_context_create_textgen('context_size=1024');") != 0) goto fail;

    // text generation should produce output and stop at EOG (not exhaust context)
    char result[4096] = {0};
    if (exec_query_text(env, db, "SELECT llm_text_generate('Say hello in one word.');", result, sizeof(result)) != 0) goto fail;
    if (result[0] == '\0') {
        fprintf(stderr, "[text_generate_eog] expected non-empty output\n");
        goto fail;
    }
    if (env->verbose) {
        printf("[text_generate_eog] output: %s\n", result);
    }

    if (exec_expect_ok(env, db, "SELECT llm_context_free();") != 0) goto fail;
    if (exec_expect_ok(env, db, "SELECT llm_model_free();") != 0) goto fail;

    sqlite3_close(db);
    return assert_sqlite_memory_clean("text_generate_with_eog", env);

fail:
    if (db) sqlite3_close(db);
    return 1;
}

// Test that double llm_chat_free doesn't crash
static int test_chat_double_free(const test_env *env) {
    sqlite3 *db = NULL;
    if (open_db_and_load(env, &db) != SQLITE_OK) return 1;

    const char *model = env->model_path ? env->model_path : DEFAULT_MODEL_PATH;
    char sqlbuf[512];
    snprintf(sqlbuf, sizeof(sqlbuf), "SELECT llm_model_load('%s');", model);
    if (exec_expect_ok(env, db, sqlbuf) != 0) goto fail;
    if (exec_expect_ok(env, db, "SELECT llm_context_create('context_size=512');") != 0) goto fail;

    if (exec_expect_ok(env, db, "SELECT llm_chat_create();") != 0) goto fail;
    if (exec_expect_ok(env, db, "SELECT llm_chat_respond('Hi');") != 0) goto fail;

    // double free should not crash
    if (exec_expect_ok(env, db, "SELECT llm_chat_free();") != 0) goto fail;
    if (exec_expect_ok(env, db, "SELECT llm_chat_free();") != 0) goto fail;

    if (exec_expect_ok(env, db, "SELECT llm_context_free();") != 0) goto fail;
    if (exec_expect_ok(env, db, "SELECT llm_model_free();") != 0) goto fail;

    sqlite3_close(db);
    return assert_sqlite_memory_clean("chat_double_free", env);

fail:
    if (db) sqlite3_close(db);
    return 1;
}

// Test chat_respond without explicit chat_create (auto-init via llm_chat_check_context)
static int test_chat_respond_auto_init(const test_env *env) {
    sqlite3 *db = NULL;
    if (open_db_and_load(env, &db) != SQLITE_OK) return 1;

    const char *model = env->model_path ? env->model_path : DEFAULT_MODEL_PATH;
    char sqlbuf[512];
    snprintf(sqlbuf, sizeof(sqlbuf), "SELECT llm_model_load('%s');", model);
    if (exec_expect_ok(env, db, sqlbuf) != 0) goto fail;
    if (exec_expect_ok(env, db, "SELECT llm_context_create('context_size=1000');") != 0) goto fail;

    // skip llm_chat_create — llm_chat_respond should auto-initialize via check_context
    if (exec_expect_ok(env, db, "SELECT llm_chat_respond('Auto init test');") != 0) goto fail;

    // second message should also work
    if (exec_expect_ok(env, db, "SELECT llm_chat_respond('Follow up');") != 0) goto fail;

    if (exec_expect_ok(env, db, "SELECT llm_chat_free();") != 0) goto fail;
    if (exec_expect_ok(env, db, "SELECT llm_context_free();") != 0) goto fail;
    if (exec_expect_ok(env, db, "SELECT llm_model_free();") != 0) goto fail;

    sqlite3_close(db);
    return assert_sqlite_memory_clean("chat_respond_auto_init", env);

fail:
    if (db) sqlite3_close(db);
    return 1;
}

// Test that chat save with title and metadata stores correctly
static int test_chat_save_with_metadata(const test_env *env) {
    sqlite3 *db = NULL;
    if (open_db_and_load(env, &db) != SQLITE_OK) return 1;

    const char *model = env->model_path ? env->model_path : DEFAULT_MODEL_PATH;
    char sqlbuf[512];
    snprintf(sqlbuf, sizeof(sqlbuf), "SELECT llm_model_load('%s');", model);
    if (exec_expect_ok(env, db, sqlbuf) != 0) goto fail;
    if (exec_expect_ok(env, db, "SELECT llm_context_create('context_size=1000');") != 0) goto fail;
    if (exec_expect_ok(env, db, "SELECT llm_chat_create();") != 0) goto fail;
    if (exec_expect_ok(env, db, "SELECT llm_chat_respond('Hello');") != 0) goto fail;

    // save with title and metadata
    if (exec_expect_ok(env, db, "SELECT llm_chat_save('Test Title', '{\"key\":\"value\"}');") != 0) goto fail;

    // verify in database
    int val = 0;
    if (select_single_int(env, db, "SELECT COUNT(*) FROM ai_chat_history WHERE title='Test Title';", &val) != 0) goto fail;
    if (val != 1) {
        fprintf(stderr, "[chat_save_metadata] expected 1 history row with title, got %d\n", val);
        goto fail;
    }

    if (exec_expect_ok(env, db, "SELECT llm_chat_free();") != 0) goto fail;
    if (exec_expect_ok(env, db, "SELECT llm_context_free();") != 0) goto fail;
    if (exec_expect_ok(env, db, "SELECT llm_model_free();") != 0) goto fail;

    sqlite3_close(db);
    return assert_sqlite_memory_clean("chat_save_with_metadata", env);

fail:
    if (db) sqlite3_close(db);
    return 1;
}

// Test text generation default n_predict cap (should not exhaust memory)
static int test_text_generate_default_limit(const test_env *env) {
    sqlite3 *db = NULL;
    if (open_db_and_load(env, &db) != SQLITE_OK) return 1;

    const char *model = env->model_path ? env->model_path : DEFAULT_MODEL_PATH;
    char sqlbuf[512];
    snprintf(sqlbuf, sizeof(sqlbuf), "SELECT llm_model_load('%s');", model);
    if (exec_expect_ok(env, db, sqlbuf) != 0) goto fail;

    // create context without explicit n_predict — should use default cap of 4096
    if (exec_expect_ok(env, db, "SELECT llm_context_create_textgen('context_size=2048');") != 0) goto fail;

    char result[4096] = {0};
    if (exec_query_text(env, db, "SELECT llm_text_generate('Say hi.');", result, sizeof(result)) != 0) goto fail;
    if (result[0] == '\0') {
        fprintf(stderr, "[text_generate_default_limit] expected non-empty output\n");
        goto fail;
    }

    if (exec_expect_ok(env, db, "SELECT llm_context_free();") != 0) goto fail;
    if (exec_expect_ok(env, db, "SELECT llm_model_free();") != 0) goto fail;

    sqlite3_close(db);
    return assert_sqlite_memory_clean("text_generate_default_limit", env);

fail:
    if (db) sqlite3_close(db);
    return 1;
}

// ---------------------------------------------------------------------
// Audio / Whisper tests
// ---------------------------------------------------------------------

static int test_audio_transcribe_no_model(const test_env *env) {
    // audio_model_transcribe should fail when no whisper model is loaded
    sqlite3 *db = NULL;
    if (open_db_and_load(env, &db) != SQLITE_OK) return 1;

    int rc = exec_expect_error(env, db, "SELECT audio_model_transcribe('/tmp/test.wav');", "No model");
    sqlite3_close(db);
    if (rc != 0) return rc;
    return assert_sqlite_memory_clean("audio_transcribe_no_model", env);
}

static int test_audio_model_load_invalid_path(const test_env *env) {
    // audio_model_load should fail with a non-existent file
    sqlite3 *db = NULL;
    if (open_db_and_load(env, &db) != SQLITE_OK) return 1;

    int rc = exec_expect_error(env, db, "SELECT audio_model_load('/nonexistent/model.bin');", "Unable to load audio model");
    sqlite3_close(db);
    if (rc != 0) return rc;
    return assert_sqlite_memory_clean("audio_model_load_invalid_path", env);
}

static int test_audio_model_load_free(const test_env *env) {
    // load and free a whisper model (requires whisper model path)
    if (!env->whisper_model_path) {
        printf("  [SKIP] no --whisper-model provided\n");
        return 0;
    }

    sqlite3 *db = NULL;
    if (open_db_and_load(env, &db) != SQLITE_OK) return 1;

    char sql[1024];
    snprintf(sql, sizeof(sql), "SELECT audio_model_load('%s');", env->whisper_model_path);
    if (exec_expect_ok(env, db, sql) != 0) { sqlite3_close(db); return 1; }

    if (exec_expect_ok(env, db, "SELECT audio_model_free();") != 0) { sqlite3_close(db); return 1; }

    sqlite3_close(db);
    return assert_sqlite_memory_clean("audio_model_load_free", env);
}

static int test_audio_transcribe_file(const test_env *env) {
    // transcribe a WAV file from a file path
    if (!env->whisper_model_path || !env->audio_path) {
        printf("  [SKIP] no --whisper-model or --audio provided\n");
        return 0;
    }

    sqlite3 *db = NULL;
    if (open_db_and_load(env, &db) != SQLITE_OK) return 1;

    char sql[1024];
    snprintf(sql, sizeof(sql), "SELECT audio_model_load('%s');", env->whisper_model_path);
    if (exec_expect_ok(env, db, sql) != 0) goto fail;

    // transcribe the audio file
    char result[4096] = {0};
    snprintf(sql, sizeof(sql), "SELECT audio_model_transcribe('%s');", env->audio_path);
    if (exec_query_text(env, db, sql, result, sizeof(result)) != 0) goto fail;

    // the result should be non-empty
    if (strlen(result) == 0) {
        fprintf(stderr, "Expected non-empty transcription result\n");
        goto fail;
    }
    if (env->verbose) printf("  Transcription: %s\n", result);

    if (exec_expect_ok(env, db, "SELECT audio_model_free();") != 0) goto fail;

    sqlite3_close(db);
    return assert_sqlite_memory_clean("audio_transcribe_file", env);
fail:
    if (db) sqlite3_close(db);
    return 1;
}

static int test_audio_transcribe_blob(const test_env *env) {
    // transcribe audio from a BLOB (read file into a table, then transcribe)
    if (!env->whisper_model_path || !env->audio_path) {
        printf("  [SKIP] no --whisper-model or --audio provided\n");
        return 0;
    }

    sqlite3 *db = NULL;
    if (open_db_and_load(env, &db) != SQLITE_OK) return 1;

    char sql[1024];
    snprintf(sql, sizeof(sql), "SELECT audio_model_load('%s');", env->whisper_model_path);
    if (exec_expect_ok(env, db, sql) != 0) goto fail;

    // read audio file into a BLOB using readfile() — use sqlite's blob approach
    // Instead, load the file manually and bind as blob
    {
        FILE *f = fopen(env->audio_path, "rb");
        if (!f) {
            fprintf(stderr, "Cannot open audio file: %s\n", env->audio_path);
            goto fail;
        }
        fseek(f, 0, SEEK_END);
        long file_size = ftell(f);
        fseek(f, 0, SEEK_SET);
        void *blob_data = malloc(file_size);
        if (!blob_data) { fclose(f); goto fail; }
        fread(blob_data, 1, file_size, f);
        fclose(f);

        // create a table, insert the blob, then transcribe
        if (exec_expect_ok(env, db, "CREATE TABLE audio_test(id INTEGER PRIMARY KEY, data BLOB);") != 0) {
            free(blob_data);
            goto fail;
        }

        sqlite3_stmt *stmt = NULL;
        int rc = sqlite3_prepare_v2(db, "INSERT INTO audio_test(data) VALUES(?);", -1, &stmt, NULL);
        if (rc != SQLITE_OK) {
            fprintf(stderr, "prepare failed: %s\n", sqlite3_errmsg(db));
            free(blob_data);
            goto fail;
        }
        sqlite3_bind_blob(stmt, 1, blob_data, (int)file_size, SQLITE_TRANSIENT);
        rc = sqlite3_step(stmt);
        sqlite3_finalize(stmt);
        free(blob_data);
        if (rc != SQLITE_DONE) {
            fprintf(stderr, "insert blob failed: %s\n", sqlite3_errmsg(db));
            goto fail;
        }

        // transcribe from the blob column
        char result[4096] = {0};
        if (exec_query_text(env, db, "SELECT audio_model_transcribe(data) FROM audio_test WHERE id=1;", result, sizeof(result)) != 0) goto fail;

        if (strlen(result) == 0) {
            fprintf(stderr, "Expected non-empty transcription from BLOB\n");
            goto fail;
        }
        if (env->verbose) printf("  Transcription from BLOB: %s\n", result);
    }

    if (exec_expect_ok(env, db, "SELECT audio_model_free();") != 0) goto fail;

    sqlite3_close(db);
    return assert_sqlite_memory_clean("audio_transcribe_blob", env);
fail:
    if (db) sqlite3_close(db);
    return 1;
}

static int test_audio_transcribe_with_options(const test_env *env) {
    // transcribe with options (language, translate, etc.)
    if (!env->whisper_model_path || !env->audio_path) {
        printf("  [SKIP] no --whisper-model or --audio provided\n");
        return 0;
    }

    sqlite3 *db = NULL;
    if (open_db_and_load(env, &db) != SQLITE_OK) return 1;

    char sql[1024];
    snprintf(sql, sizeof(sql), "SELECT audio_model_load('%s');", env->whisper_model_path);
    if (exec_expect_ok(env, db, sql) != 0) goto fail;

    // transcribe with language=en and single_segment=1
    char result[4096] = {0};
    snprintf(sql, sizeof(sql), "SELECT audio_model_transcribe('%s', 'language=en,single_segment=1');", env->audio_path);
    if (exec_query_text(env, db, sql, result, sizeof(result)) != 0) goto fail;

    if (strlen(result) == 0) {
        fprintf(stderr, "Expected non-empty transcription with options\n");
        goto fail;
    }
    if (env->verbose) printf("  Transcription (with options): %s\n", result);

    if (exec_expect_ok(env, db, "SELECT audio_model_free();") != 0) goto fail;

    sqlite3_close(db);
    return assert_sqlite_memory_clean("audio_transcribe_with_options", env);
fail:
    if (db) sqlite3_close(db);
    return 1;
}

static int test_audio_transcribe_unsupported_format(const test_env *env) {
    // transcribing an unsupported file format should fail
    if (!env->whisper_model_path) {
        printf("  [SKIP] no --whisper-model provided\n");
        return 0;
    }

    sqlite3 *db = NULL;
    if (open_db_and_load(env, &db) != SQLITE_OK) return 1;

    char sql[1024];
    snprintf(sql, sizeof(sql), "SELECT audio_model_load('%s');", env->whisper_model_path);
    if (exec_expect_ok(env, db, sql) != 0) goto fail;

    // try to transcribe a .txt file
    int rc = exec_expect_error(env, db, "SELECT audio_model_transcribe('/tmp/test.txt');", "Unsupported audio format");
    if (rc != 0) goto fail;

    // try to transcribe a random blob
    rc = exec_expect_error(env, db, "SELECT audio_model_transcribe(X'DEADBEEF');", "Unsupported audio format");
    if (rc != 0) goto fail;

    if (exec_expect_ok(env, db, "SELECT audio_model_free();") != 0) goto fail;

    sqlite3_close(db);
    return assert_sqlite_memory_clean("audio_transcribe_unsupported_format", env);
fail:
    if (db) sqlite3_close(db);
    return 1;
}

static int test_audio_model_load_free_cycle(const test_env *env) {
    // load, free, reload should work without leaks
    if (!env->whisper_model_path) {
        printf("  [SKIP] no --whisper-model provided\n");
        return 0;
    }

    sqlite3 *db = NULL;
    if (open_db_and_load(env, &db) != SQLITE_OK) return 1;

    char sql[1024];
    for (int i = 0; i < 3; i++) {
        snprintf(sql, sizeof(sql), "SELECT audio_model_load('%s');", env->whisper_model_path);
        if (exec_expect_ok(env, db, sql) != 0) goto fail;
        if (exec_expect_ok(env, db, "SELECT audio_model_free();") != 0) goto fail;
    }

    sqlite3_close(db);
    return assert_sqlite_memory_clean("audio_model_load_free_cycle", env);
fail:
    if (db) sqlite3_close(db);
    return 1;
}

static int test_llm_chat_double_save(const test_env *env) {
    sqlite3 *db = NULL;
    bool model_loaded = false;
    bool context_created = false;
    bool chat_created = false;
    int status = 1;
    
    if (open_db_and_load(env, &db) != SQLITE_OK) {
        goto done;
    }
    
    const char *model = env->model_path ? env->model_path : DEFAULT_MODEL_PATH;
    char sqlbuf[512];
    snprintf(sqlbuf, sizeof(sqlbuf), "SELECT llm_model_load('%s');", model);
    if (exec_expect_ok(env, db, sqlbuf) != 0)
        goto done;
    model_loaded = true;
    
    if (exec_expect_ok(env, db,
                       "SELECT llm_context_create('context_size=1000');") != 0)
        goto done;
    context_created = true;
    
    if (exec_expect_ok(env, db, "SELECT llm_chat_create();") != 0)
        goto done;
    chat_created = true;
    
    // First prompt
    const char *prompt1 = "First prompt";
    if (exec_expect_ok(env, db, "SELECT llm_chat_respond('First prompt');") != 0)
        goto done;
    
    // First save
    if (exec_expect_ok(env, db, "SELECT llm_chat_save();") != 0)
        goto done;
    
    // Second prompt
    const char *prompt2 = "Second prompt";
    if (exec_expect_ok(env, db, "SELECT llm_chat_respond('Second prompt');") != 0)
        goto done;
    
    // Second save
    if (exec_expect_ok(env, db, "SELECT llm_chat_save();") != 0)
        goto done;
    
    ai_chat_message_row rows[8];
    int count = 0;
    // We expect 4 messages: User1, Assistant1, User2, Assistant2
    if (fetch_ai_chat_messages(env, db, rows, 8, &count) != 0)
        goto done;
    
    if (count != 5) {
        fprintf(stderr,
                "[test_llm_chat_double_save] expected 4 message rows, got %d\n",
                count);
        goto done;
    }
    
    // Verify order and roles
    if (strcmp(rows[0].role, "system") != 0 ||
        strcmp(rows[0].content, "") != 0) {
        fprintf(stderr,
                "[test_llm_chat_double_save] row 0 mismatch (expected system/'%s', "
                "got %s/'%s')\n",
                "", rows[0].role, rows[0].content);
        goto done;
    }
    if (strcmp(rows[1].role, "user") != 0 ||
        strcmp(rows[1].content, prompt1) != 0) {
        fprintf(stderr,
                "[test_llm_chat_double_save] row 0 mismatch (expected user/'%s', "
                "got %s/'%s')\n",
                prompt1, rows[1].role, rows[1].content);
        goto done;
    }
    if (strcmp(rows[2].role, "assistant") != 0) {
        fprintf(stderr,
                "[test_llm_chat_double_save] row 1 mismatch (expected assistant, "
                "got %s)\n",
                rows[2].role);
        goto done;
    }
    if (strcmp(rows[3].role, "user") != 0 ||
        strcmp(rows[3].content, prompt2) != 0) {
        fprintf(stderr,
                "[test_llm_chat_double_save] row 2 mismatch (expected user/'%s', "
                "got %s/'%s')\n",
                prompt2, rows[3].role, rows[3].content);
        goto done;
    }
    if (strcmp(rows[4].role, "assistant") != 0) {
        fprintf(stderr,
                "[test_llm_chat_double_save] row 3 mismatch (expected assistant, "
                "got %s)\n",
                rows[4].role);
        goto done;
    }
    
    status = 0;
    
done:
    if (chat_created)
        exec_expect_ok(env, db, "SELECT llm_chat_free();");
    if (context_created)
        exec_expect_ok(env, db, "SELECT llm_context_free();");
    if (model_loaded)
        exec_expect_ok(env, db, "SELECT llm_model_free();");
    if (db)
        sqlite3_close(db);
    if (status == 0)
        status = assert_sqlite_memory_clean("llm_chat_double_save", env);
    return status;
}

static const test_case TESTS[] = {
    {"issue15_llm_chat_without_context", test_issue15_chat_without_context},
    {"llm_chat_respond_repeated", test_llm_chat_respond_repeated},
    {"llm_chat_vtab", test_llm_chat_vtab},
    {"test_llm_embed_generate", test_llm_embed_generate},
    {"llm_embed_generate_basic", test_llm_embed_generate_basic},
    {"llm_embedding_then_chat", test_llm_embedding_then_chat},
    {"llm_context_size_errors", test_llm_context_size_errors},
    {"document_ingestion_flow", test_document_ingestion_flow},
    {"llm_sampler_roundtrip", test_llm_sampler_roundtrip},
    {"dual_connection_roles", test_dual_connection_roles},
    {"concurrent_connections_independent", test_concurrent_connections_independent},
    {"llm_model_load_error_recovery", test_llm_model_load_error_recovery},
    {"ai_logging_table", test_ai_logging_table},
    {"llm_embed_input_too_large", test_llm_embed_input_too_large},
    {"llm_embed_nctx_exceeds_train", test_llm_embed_nctx_exceeds_train},
    {"llm_embed_max_tokens_limit", test_llm_embed_max_tokens_limit},
    {"llm_embed_repeated_calls", test_llm_embed_repeated_calls},
    {"llm_embed_empty_input", test_llm_embed_empty_input},
    {"chat_system_prompt_new_chat", test_chat_system_prompt_new_chat},
    {"chat_system_prompt_replace_previous_prompt", test_chat_system_prompt_replace_previous_prompt},
    {"chat_system_prompt_after_first_response", test_chat_system_prompt_after_first_response},
    {"chat_create_free_cycle", test_chat_create_free_cycle},
    {"chat_recreate_after_conversation", test_chat_recreate_after_conversation},
    {"chat_vtab_multi_turn", test_chat_vtab_multi_turn},
    {"chat_save_restore_roundtrip", test_chat_save_restore_roundtrip},
    {"chat_system_prompt_clear", test_chat_system_prompt_clear},
    {"text_generate_with_eog", test_text_generate_with_eog},
    {"chat_double_free", test_chat_double_free},
    {"chat_respond_auto_init", test_chat_respond_auto_init},
    {"chat_save_with_metadata", test_chat_save_with_metadata},
    {"text_generate_default_limit", test_text_generate_default_limit},
    {"llm_chat_double_save", test_llm_chat_double_save},
    // Audio / Whisper tests
    {"audio_transcribe_no_model", test_audio_transcribe_no_model},
    {"audio_model_load_invalid_path", test_audio_model_load_invalid_path},
    {"audio_model_load_free", test_audio_model_load_free},
    {"audio_transcribe_file", test_audio_transcribe_file},
    {"audio_transcribe_blob", test_audio_transcribe_blob},
    {"audio_transcribe_with_options", test_audio_transcribe_with_options},
    {"audio_transcribe_unsupported_format", test_audio_transcribe_unsupported_format},
    {"audio_model_load_free_cycle", test_audio_model_load_free_cycle},
};

int main(int argc, char **argv) {
    test_env env = {
        .extension_path = "./dist/ai",
        .model_path = NULL,
        .whisper_model_path = NULL,
        .audio_path = NULL,
        .verbose = false,
    };
    const char *selected_test = NULL;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--extension") == 0) {
            if (++i >= argc) {
                usage(argv[0]);
                return EXIT_FAILURE;
            }
            env.extension_path = argv[i];
        } else if (strcmp(argv[i], "--model") == 0) {
            if (++i >= argc) {
                usage(argv[0]);
                return EXIT_FAILURE;
            }
            env.model_path = argv[i];
        } else if (strcmp(argv[i], "--whisper-model") == 0) {
            if (++i >= argc) {
                usage(argv[0]);
                return EXIT_FAILURE;
            }
            env.whisper_model_path = argv[i];
        } else if (strcmp(argv[i], "--audio") == 0) {
            if (++i >= argc) {
                usage(argv[0]);
                return EXIT_FAILURE;
            }
            env.audio_path = argv[i];
        } else if (strcmp(argv[i], "--verbose") == 0) {
            env.verbose = true;
        } else if (strcmp(argv[i], "--test") == 0) {
            if (++i >= argc) {
                usage(argv[0]);
                return EXIT_FAILURE;
            }
            selected_test = argv[i];
        } else if (strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            return EXIT_SUCCESS;
        } else {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            usage(argv[0]);
            return EXIT_FAILURE;
        }
    }

    size_t total = sizeof(TESTS) / sizeof(TESTS[0]);
    int failures = 0;

    if (selected_test) printf("Running 1 C test\n\n");
    else printf("Running %zu C test(s)\n\n", total);
    for (size_t i = 0; i < total; ++i) {
        const test_case *tc = &TESTS[i];
        if (selected_test && strcmp(tc->name, selected_test) != 0) {
            continue;
        }
        int rc = tc->fn(&env);
        printf("- %s ... %s\n", tc->name, rc == 0 ? "PASS" : "FAIL");
        if (rc != 0) failures += 1;
    }
    if (selected_test && failures == 0) {
        bool found = false;
        for (size_t i = 0; i < total; ++i) {
            if (strcmp(TESTS[i].name, selected_test) == 0) {
                found = true;
                break;
            }
        }
        if (!found) {
            fprintf(stderr, "Unknown test '%s'\n", selected_test);
            return EXIT_FAILURE;
        }
    }

    if (failures) {
        fprintf(stderr, "\n%d C test(s) failed.\n", failures);
        return EXIT_FAILURE;
    }

    printf("\nAll C tests passed.\n");
    return EXIT_SUCCESS;
}
