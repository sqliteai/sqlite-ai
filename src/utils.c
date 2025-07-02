//
//  utils.c
//  sqliteai
//
//  Created by Marco Bambini on 27/06/25.
//

#include "utils.h"
#include <stdio.h>
#include <ctype.h>

#define SKIP_SPACES(_p)                         while (*(_p) && isspace((unsigned char)*(_p))) (_p)++
#define TRIM_TRAILING(_start, _len)             while ((_len) > 0 && isspace((unsigned char)(_start)[(_len) - 1])) (_len)--
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

// defined in sqlite-ai.c
bool ai_check_model (sqlite3_context *context);
bool is_sqlite_344_or_higher;

// MARK: -

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

void *sqlite_common_set_error (sqlite3_context *context, sqlite3_vtab *vtab, int rc, const char *format, ...) {
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
    
    return NULL;
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
        if (ai_check_model(context) == false) {
            sqlite_context_result_error(context, SQLITE_MISUSE, "");
            return false;
        }
    }
    
    return true;
}

// MARK: -

bool sqlite_set_ptr (sqlite3 *db, void *ptr) {
    if (is_sqlite_344_or_higher) {
        return (sqlite3_set_clientdata(db, HASH_KEY_AI, ptr, NULL) == SQLITE_OK);
    }
    return (llist_set(&llist, db, ptr) == 0);
}

void *sqlite_get_ptr (sqlite3 *db) {
    if (is_sqlite_344_or_higher) {
        return sqlite3_get_clientdata(db, HASH_KEY_AI);
    }
    return llist_get(&llist, db);
}

void sqlite_clear_ptr (sqlite3 *db) {
    llist_remove(&llist, db);
}

bool sqlite_utils_init (void) {
    is_sqlite_344_or_higher = (sqlite3_libversion_number() >= 3044000);
    
    llist.mutex = sqlite3_mutex_alloc(SQLITE_MUTEX_STATIC_APP1);
    if (llist.mutex == NULL) return false;
    
    return true;
}

// MARK: -

bool parse_keyvalue_string (sqlite3_context *context, const char *str, keyvalue_callback callback, void *xdata) {
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
        
        bool rc = callback(context, xdata, key_start, key_len, val_start, val_len);
        if (!rc) return rc;
        
        if (*p == ',') p++;
    }
    
    return true;
}
