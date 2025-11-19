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
    bool verbose;
} test_env;

typedef int (*test_fn)(const test_env *env);

typedef struct {
    const char *name;
    test_fn fn;
} test_case;

static void usage(const char *prog) {
    fprintf(stderr, "Usage: %s [--extension /path/to/ai] [--model /path/to/model] [--verbose]\n", prog);
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

    sqlite3_int64 current = 0;
    sqlite3_int64 highwater = 0;
    if (sqlite3_status64(SQLITE_STATUS_MEMORY_USED, &current, &highwater, 0) != SQLITE_OK) {
        fprintf(stderr, "[chat_respond_repeated] sqlite3_status64 failed\n");
        return 1;
    }
    if (env->verbose) {
        printf("[STATUS] memory current=%lld highwater=%lld\n", (long long)current, (long long)highwater);
    }
    if (current > 0 || highwater <= 0) {
        fprintf(stderr, "[chat_respond_repeated] invalid memory stats: current=%lld highwater=%lld\n",
                (long long)current, (long long)highwater);
        return 1;
    }

    return 0;
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

    sqlite3_int64 current = 0;
    sqlite3_int64 highwater = 0;
    if (sqlite3_status64(SQLITE_STATUS_MEMORY_USED, &current, &highwater, 0) != SQLITE_OK) {
        fprintf(stderr, "[chat_vtab] sqlite3_status64 failed\n");
        return 1;
    }
    if (current > 0 || highwater <= 0) {
        fprintf(stderr, "[chat_vtab] invalid memory stats: current=%lld highwater=%lld\n",
                (long long)current, (long long)highwater);
        return 1;
    }
    return 0;

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
    return status;
}

static const test_case TESTS[] = {
    {"issue15_llm_chat_without_context", test_issue15_chat_without_context},
    {"llm_chat_respond_repeated", test_llm_chat_respond_repeated},
    {"llm_chat_vtab", test_llm_chat_vtab},
    {"chat_system_prompt_new_chat", test_chat_system_prompt_new_chat},
    {"chat_system_prompt_replace_previous_prompt", test_chat_system_prompt_replace_previous_prompt},
    {"chat_system_prompt_after_first_response", test_chat_system_prompt_after_first_response},
};

int main(int argc, char **argv) {
    test_env env = {
        .extension_path = "./dist/ai",
        .model_path = NULL,
        .verbose = false,
    };

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
        } else if (strcmp(argv[i], "--verbose") == 0) {
            env.verbose = true;
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

    printf("Running %zu C test(s)\n\n", total);
    for (size_t i = 0; i < total; ++i) {
        const test_case *tc = &TESTS[i];
        int rc = tc->fn(&env);
        printf("- %s ... %s\n", tc->name, rc == 0 ? "PASS" : "FAIL");
        if (rc != 0) failures += 1;
    }

    if (failures) {
        fprintf(stderr, "\n%d C test(s) failed.\n", failures);
        return EXIT_FAILURE;
    }

    printf("\nAll C tests passed.\n");
    return EXIT_SUCCESS;
}
