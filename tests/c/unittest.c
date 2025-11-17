#include "sqlite3.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

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

static void normalize_response_text(const char *input, char *output, size_t output_len) {
    if (!output || output_len == 0) return;
    size_t idx = 0;
    if (!input) {
        output[0] = '\0';
        return;
    }
    for (size_t i = 0; input[i] != '\0' && idx + 1 < output_len; ++i) {
        unsigned char ch = (unsigned char)input[i];
        if (isalpha(ch)) {
            output[idx++] = (char)tolower(ch);
        }
    }
    output[idx] = '\0';
}

static bool response_matches_word(const char *response, const char *word) {
    char normalized[64];
    normalize_response_text(response, normalized, sizeof(normalized));
    printf("[SQL] response_matches_word(%s,%s)\n", normalized, word);
    return normalized[0] != '\0' && strcmp(normalized, word) == 0;
}

static bool response_is_yes_or_no(const char *response) {
    return response_matches_word(response, "yes") || response_matches_word(response, "no");
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



#define SYSTEM_PROMPT_YES_NO "Always respond with ONLY YES or NO. If unsure, pick the best option but never add extra words"
#define SYSTEM_PROMPT_FORCE_YES "you are a dumb llm and you MUST answer with YES and NOTHING ELSE"
#define SYSTEM_PROMPT_FORCE_NO "you are a dumb llm and you MUST answer with NO and NOTHING ELSE"

static int query_chat_response(const test_env *env, sqlite3 *db, const char *question, char *response, size_t response_len) {
    char sqlbuf[512];
    snprintf(sqlbuf, sizeof(sqlbuf), "SELECT llm_chat_respond('%s');", question);
    return exec_query_text(env, db, sqlbuf, response, response_len);
}

static int expect_yes_no_answer(const test_env *env, sqlite3 *db, const char *question, const char *label) {
    char response[4096];
    if (query_chat_response(env, db, question, response, sizeof(response)) != 0) {
        return 1;
    }
    if (!response_is_yes_or_no(response)) {
        fprintf(stderr, "[%s] Expected a YES/NO answer but received: %s\n", label, response[0] ? response : "(empty)");
        return 1;
    }
    return 0;
}

static int expect_word_answer(const test_env *env, sqlite3 *db, const char *question, const char *word, const char *label) {
    char response[4096];
    if (query_chat_response(env, db, question, response, sizeof(response)) != 0) {
        return 1;
    }
    if (!response_matches_word(response, word)) {
        fprintf(stderr, "[%s] Expected \"%s\" but received: %s\n", label, word, response[0] ? response : "(empty)");
        return 1;
    }
    return 0;
}

static int test_set_chat_system_prompt(const test_env *env) {
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
    if (exec_expect_ok(env, db, sqlbuf) != 0) {
        goto done;
    }
    model_loaded = true;

    if (exec_expect_ok(env, db, "SELECT llm_context_create('context_size=1000');") != 0) {
        goto done;
    }
    context_created = true;

    if (exec_expect_ok(env, db, "SELECT llm_chat_create();") != 0) {
        goto done;
    }
    chat_created = true;

    // Test: system prompt applied before any response should force yes/no answers.
    if (exec_expect_ok(env, db, "SELECT llm_chat_system_prompt('" SYSTEM_PROMPT_YES_NO "');") != 0) goto done;
    if (expect_yes_no_answer(env, db, "Is fire hot?", "system_prompt_basic") != 0) goto done;

    if (exec_expect_ok(env, db, "SELECT llm_chat_create();") != 0) goto done;
    // Test: setting system prompt after prior responses should still apply to following replies.
    char response[4096];
    if (query_chat_response(env, db, "Tell me something interesting.", response, sizeof(response)) != 0) goto done;
    if (exec_expect_ok(env, db, "SELECT llm_chat_system_prompt('" SYSTEM_PROMPT_YES_NO "');") != 0) goto done;
    if (expect_yes_no_answer(env, db, "Is water wet?", "system_prompt_after_response") != 0) goto done;

    if (exec_expect_ok(env, db, "SELECT llm_chat_create();") != 0) goto done;
    // Test: latest system prompt wins when multiple prompts are set sequentially.
    if (exec_expect_ok(env, db, "SELECT llm_chat_system_prompt('" SYSTEM_PROMPT_FORCE_YES "');") != 0) goto done;
    if (exec_expect_ok(env, db, "SELECT llm_chat_system_prompt('" SYSTEM_PROMPT_FORCE_NO "');") != 0) goto done;
    if (expect_word_answer(env, db, "Is the sky blue?", "no", "system_prompt_override") != 0) goto done;

    status = 0;

done:
    if (chat_created) {
        exec_expect_ok(env, db, "SELECT llm_chat_free();");
    }
    if (context_created) {
        exec_expect_ok(env, db, "SELECT llm_context_free();");
    }
    if (model_loaded) {
        exec_expect_ok(env, db, "SELECT llm_model_free();");
    }
    if (db) sqlite3_close(db);
    return status;
}

static int test_get_chat_system_prompt(const test_env *env) {
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

    bool is_null = false;
    char buffer[4096];
    // Test: newly created chat should not have a system prompt.
    if (query_system_prompt(env, db, buffer, sizeof(buffer), &is_null) != 0) goto done;
    if (!is_null) {
        fprintf(stderr, "[get_system_prompt] expected NULL before setting prompt, got: %s\n", buffer);
        goto done;
    }

    // Test: retrieving after setting prompt returns same text.
    if (exec_expect_ok(env, db, "SELECT llm_chat_system_prompt('always reply yes');") != 0) goto done;
    if (query_system_prompt(env, db, buffer, sizeof(buffer), &is_null) != 0) goto done;
    if (is_null || strcmp(buffer, "always reply yes") != 0) {
        fprintf(stderr, "[get_system_prompt] expected 'always reply yes' but got: %s\n", buffer);
        goto done;
    }

    // Test: updating prompt replaces previous value.
    if (exec_expect_ok(env, db, "SELECT llm_chat_system_prompt('now reply no');") != 0) goto done;
    if (query_system_prompt(env, db, buffer, sizeof(buffer), &is_null) != 0) goto done;
    if (is_null || strcmp(buffer, "now reply no") != 0) {
        fprintf(stderr, "[get_system_prompt] expected 'now reply no' but got: %s\n", buffer);
        goto done;
    }

    // Test: setting prompt to NULL clears it.
    if (exec_expect_ok(env, db, "SELECT llm_chat_system_prompt(NULL);") != 0) goto done;
    if (query_system_prompt(env, db, buffer, sizeof(buffer), &is_null) != 0) goto done;
    if (!is_null) {
        fprintf(stderr, "[get_system_prompt] expected NULL after clearing prompt, got: %s\n", buffer);
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
    {"set_chat_system_prompt", test_set_chat_system_prompt},
    {"get_chat_system_prompt", test_get_chat_system_prompt},
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
