//
//  utils.c
//  sqliteai
//
//  Created by Marco Bambini on 27/06/25.
//

#include "utils.h"
#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#include <objbase.h>
#include <bcrypt.h>
#include <ntstatus.h> //for STATUS_SUCCESS
#else
#include <unistd.h>
#if defined(__APPLE__)
#include <Security/Security.h>
#elif !defined(__ANDROID__)
#include <sys/random.h>
#endif
#endif

#ifndef SQLITE_CORE
SQLITE_EXTENSION_INIT3
#endif

#define SKIP_SPACES(_p)                         while (*(_p) && isspace((unsigned char)*(_p))) (_p)++
#define TRIM_TRAILING(_start, _len)             while ((_len) > 0 && isspace((unsigned char)(_start)[(_len) - 1])) (_len)--
#define MIN_BUFFER_SIZE                         4096
#define UUID_LEN                                16

// defined in sqlite-ai.c
bool ai_model_check (sqlite3_context *context);

// MARK: -

bool buffer_create (buffer_t *b, uint32_t size) {
    if (size < MIN_BUFFER_SIZE) size = MIN_BUFFER_SIZE;
    char *mem = sqlite3_malloc(size);
    if (!mem) return false;
    
    b->data = mem;
    b->capacity = size;
    b->length = 0;
    
    return true;
}

bool buffer_resize (buffer_t *b, uint32_t new_capacity) {
    char *clone = sqlite3_realloc(b->data, new_capacity);
    if (!clone) return false;
    
    b->data = clone;
    b->capacity = new_capacity;
    
    return true;
}

void buffer_reset (buffer_t *b) {
    if (b->data) b->data[0] = 0;
    b->length = 0;
}

bool buffer_append (buffer_t *b, const char *data, uint32_t len, bool zero_terminate) {
    if (b->length + len + 1> b->capacity) {
        uint32_t new_capacity = b->length + len + 1 + MIN_BUFFER_SIZE;
        if (buffer_resize(b, new_capacity) == false) return false;
    }
    
    memcpy(b->data + b->length, data, len);
    b->length += len;
    if (zero_terminate) b->data[b->length] = 0;
    
    return true;
}

void buffer_destroy (buffer_t *b) {
    if (b->data) {
        sqlite3_free(b->data);
        b->data = NULL;
    }
    b->data = NULL;
    b->capacity = 0;
    b->length = 0;
}

/*
#define HASH_KEY_AI                             "SQLITE-AI"

// linked list used to share context with virtual tables
typedef struct kv_node {
    void            *key;
    void            *value;
    struct kv_node  *next;
} kv_node_t;        // node structure for the linked list

typedef struct kv_list {
    kv_node_t       *head;
    size_t          count;
    sqlite3_mutex   *mutex;
} kv_list_t;        // list structure

static kv_list_t llist;

// MARK:

int llist_set (kv_list_t *list, void *key, void *value) {
    int rc = 0;
    
    sqlite3_mutex_enter(list->mutex);
    
    // check if key already exists
    kv_node_t *current = list->head;
    while (current) {
        if (current->key == key) {
            // key exists, update value
            current->value = value;
            goto cleanup;
        }
        current = current->next;
    }
    
    // key doesn't exist, create new node
    kv_node_t *new_node = sqlite3_malloc(sizeof(kv_node_t));
    if (!new_node) {rc = -1; goto cleanup;}
    
    new_node->key = key;
    new_node->value = value;
    new_node->next = list->head;
    list->head = new_node;
    list->count++;
    
cleanup:
    sqlite3_mutex_leave(list->mutex);
    return rc;
}

void *llist_get (kv_list_t *list, void *key) {
    void *ptr = NULL;
    sqlite3_mutex_enter(list->mutex);
    
    kv_node_t *current = list->head;
    while (current) {
        if (current->key == key) {
            ptr = current->value;
            goto cleanup;
        }
        current = current->next;
    }
    
cleanup:
    sqlite3_mutex_leave(list->mutex);
    return ptr;
}

int llist_remove (kv_list_t *list, void *key) {
    kv_node_t *current = list->head;
    kv_node_t *prev = NULL;
    int rc = -1;
    
    sqlite3_mutex_enter(list->mutex);
    while (current) {
        if (current->key == key) {
            // found the node to remove
            if (prev) {
                prev->next = current->next;
            } else {
                list->head = current->next;
            }
            
            sqlite3_free(current);
            list->count--;
            rc = 0;
            goto cleanup;
        }
        prev = current;
        current = current->next;
    }
    
cleanup:
    sqlite3_mutex_leave(list->mutex);
    return rc;
}
 */

// MARK: -

static const char *sqlite_type_name (int type) {
    switch (type) {
        case SQLITE_TEXT: return "TEXT";
        case SQLITE_INTEGER: return "INTEGER";
        case SQLITE_FLOAT: return "REAL";
        case SQLITE_BLOB: return "BLOB";
    }
    return "N/A";
}

int sqlite_vtab_set_error (sqlite3_vtab *vtab, const char *format, ...) {
    va_list arg;
    va_start (arg, format);
    char *err = sqlite3_vmprintf(format, arg);
    va_end (arg);
    
    vtab->zErrMsg = err;
    return SQLITE_ERROR;
}

bool sqlite_context_result_error (sqlite3_context *context, int rc, const char *format, ...) {
    char buffer[4096];
    
    va_list arg;
    va_start (arg, format);
    vsnprintf(buffer, sizeof(buffer), format, arg);
    va_end (arg);
    
    if (context) {
        sqlite3_result_error(context, buffer, -1);
        sqlite3_result_error_code(context, rc);
    }
    
    return false;
}

void sqlite_common_set_error (sqlite3_context *context, sqlite3_vtab *vtab, int rc, const char *format, ...) {
    char buffer[4096];
    char *err = NULL;
    
    va_list arg;
    va_start (arg, format);
    if (vtab) err = sqlite3_vmprintf(format, arg);
    else if (context) vsnprintf(buffer, sizeof(buffer), format, arg);
    va_end (arg);
        
    if (vtab) {
        vtab->zErrMsg = err;
    } else if (context) {
        sqlite3_result_error(context, buffer, -1);
        sqlite3_result_error_code(context, rc);
    }
}

bool sqlite_sanity_function (sqlite3_context *context, const char *func_name, int argc, sqlite3_value **argv, int ntypes, int *types, bool check_model) {
    if (argc != ntypes) {
        sqlite_context_result_error(context, SQLITE_ERROR, "Function '%s' expects %d arguments, but %d were provided.", func_name, ntypes, argc);
        return false;
    }
    
    for (int i=0; i<argc; ++i) {
        int actual_type = sqlite3_value_type(argv[i]);
        if (actual_type != types[i]) {
            sqlite_context_result_error(context, SQLITE_ERROR, "Function '%s': argument %d must be of type %s (got %s).", func_name, (i+1), sqlite_type_name(types[i]), sqlite_type_name(actual_type));
            return false;
        }
    }
    
    if (check_model) {
        if (ai_model_check(context) == false) {
            sqlite_context_result_error(context, SQLITE_MISUSE, "No model is currently set. Please call ai_model_load() before using this function.");
            return false;
        }
    }
    
    return true;
}

int sqlite_db_write (sqlite3_context *context, sqlite3 *db, const char *sql, const char **values, int types[], int lens[], int count) {
    sqlite3_stmt *pstmt = NULL;
    
    // compile sql
    int rc = sqlite3_prepare_v2(db, sql, -1, &pstmt, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    // check bindings
    for (int i=0; i<count; ++i) {
        // sanity check input
        if ((types[i] != SQLITE_NULL) && (values[i] == NULL)) {
            rc = sqlite3_bind_null(pstmt, i+1);
            continue;
        }
        
        switch (types[i]) {
            case SQLITE_NULL:
                rc = sqlite3_bind_null(pstmt, i+1);
                break;
            case SQLITE_TEXT:
                rc = sqlite3_bind_text(pstmt, i+1, values[i], lens[i], SQLITE_STATIC);
                break;
            case SQLITE_BLOB:
                rc = sqlite3_bind_blob(pstmt, i+1, values[i], lens[i], SQLITE_STATIC);
                break;
            case SQLITE_INTEGER: {
                sqlite3_int64 value = strtoll(values[i], NULL, 0);
                rc = sqlite3_bind_int64(pstmt, i+1, value);
            }   break;
            case SQLITE_FLOAT: {
                double value = strtod(values[i], NULL);
                rc = sqlite3_bind_double(pstmt, i+1, value);
            }   break;
        }
        if (rc != SQLITE_OK) goto cleanup;
    }
        
    // execute statement
    rc = sqlite3_step(pstmt);
    if (rc == SQLITE_DONE) rc = SQLITE_OK;
    
cleanup:
    if (rc != SQLITE_OK) {
        if (context) sqlite3_result_error(context, sqlite3_errmsg(db), -1);
        else printf("Error executing %s in db_write (%s).", sql, sqlite3_errmsg(db));
    }
    if (pstmt) sqlite3_finalize(pstmt);
    return rc;
}

int sqlite_db_write_simple (sqlite3_context *context, sqlite3 *db, const char *sql) {
    return sqlite_db_write(context, db, sql, NULL, NULL, NULL, 0);
}

char *sqlite_strdup (const char *str) {
    if (!str) return NULL;
    
    size_t len = strlen(str) + 1;
    char *result = (char*)sqlite3_malloc((int)len);
    if (result) memcpy(result, str, len);
    
    return result;
}

// MARK: -

int ai_uuid_v7_generate (uint8_t value[UUID_LEN]) {
    // fill the buffer with high-quality random data
    #ifdef _WIN32
    if (BCryptGenRandom(NULL, (BYTE*)value, UUID_LEN, BCRYPT_USE_SYSTEM_PREFERRED_RNG) != STATUS_SUCCESS) return -1;
    #elif defined(__APPLE__)
    // Use SecRandomCopyBytes for macOS/iOS
    if (SecRandomCopyBytes(kSecRandomDefault, UUID_LEN, value) != errSecSuccess) return -1;
    #elif defined(__ANDROID__)
    //arc4random_buf doesn't have a return value to check for success
    arc4random_buf(value, UUID_LEN);
    #else
    if (getentropy(value, UUID_LEN) != 0) return -1;
    #endif
    
    // get current timestamp in ms
    struct timespec ts;
    #ifdef __ANDROID__
    if (clock_gettime(CLOCK_REALTIME, &ts) != 0) return -1;
    #else
    if (timespec_get(&ts, TIME_UTC) == 0) return -1;
    #endif
    
    // add timestamp part to UUID
    uint64_t timestamp = (uint64_t)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
    value[0] = (timestamp >> 40) & 0xFF;
    value[1] = (timestamp >> 32) & 0xFF;
    value[2] = (timestamp >> 24) & 0xFF;
    value[3] = (timestamp >> 16) & 0xFF;
    value[4] = (timestamp >> 8) & 0xFF;
    value[5] = timestamp & 0xFF;
    
    // version and variant
    value[6] = (value[6] & 0x0F) | 0x70; // UUID version 7
    value[8] = (value[8] & 0x3F) | 0x80; // RFC 4122 variant
    
    return 0;
}

char *ai_uuid_v7_stringify (uint8_t uuid[UUID_LEN], char value[UUID_STR_MAXLEN], bool dash_format) {
    if (dash_format) {
        snprintf(value, UUID_STR_MAXLEN, "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
            uuid[0], uuid[1], uuid[2], uuid[3], uuid[4], uuid[5], uuid[6], uuid[7],
            uuid[8], uuid[9], uuid[10], uuid[11], uuid[12], uuid[13], uuid[14], uuid[15]
        );
    } else {
        snprintf(value, UUID_STR_MAXLEN, "%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x",
            uuid[0], uuid[1], uuid[2], uuid[3], uuid[4], uuid[5], uuid[6], uuid[7],
            uuid[8], uuid[9], uuid[10], uuid[11], uuid[12], uuid[13], uuid[14], uuid[15]
        );
    }
    
    return (char *)value;
}

char *ai_uuid_v7_string (char value[UUID_STR_MAXLEN], bool dash_format) {
    uint8_t uuid[UUID_LEN];
    if (ai_uuid_v7_generate(uuid) != 0) return NULL;
    return ai_uuid_v7_stringify(uuid, value, dash_format);
}

// MARK: -

bool parse_keyvalue_string (const char *str, keyvalue_callback callback, void *xdata) {
    if (!str) return true;
    
    const char *p = str;
    while (*p) {
        SKIP_SPACES(p);
        
        const char *key_start = p;
        while (*p && *p != '=' && *p != ',') p++;
        
        int key_len = (int)(p - key_start);
        TRIM_TRAILING(key_start, key_len);
        
        if (*p != '=') {
            // Skip malformed pair
            while (*p && *p != ',') p++;
            if (*p == ',') p++;
            continue;
        }
        
        p++; // skip '='
        SKIP_SPACES(p);
        
        const char *val_start = p;
        while (*p && *p != ',') p++;
        
        int val_len = (int)(p - val_start);
        TRIM_TRAILING(val_start, val_len);
        
        bool rc = callback(xdata, key_start, key_len, val_start, val_len);
        if (!rc) return rc;
        
        if (*p == ',') p++;
    }
    
    return true;
}
