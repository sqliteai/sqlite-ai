//
//  utils.h
//  sqliteai
//
//  Created by Marco Bambini on 27/06/25.
//

#ifndef __SQLITEAI_UTILS__
#define __SQLITEAI_UTILS__

#include <stdint.h>
#include <stdbool.h>

#ifndef SQLITE_CORE
#include "sqlite3ext.h"
#else
#include "sqlite3.h"
#endif

typedef struct {
    char                *data;                  // raw buffer
    uint32_t            capacity;               // size of the buffer
    uint32_t            length;                 // currently used size
} buffer_t;

typedef bool (*keyvalue_callback)(sqlite3_context *context, void *xdata, const char *key, int key_len, const char *value, int value_len);

//bool sqlite_utils_init (void);
bool sqlite_sanity_function (sqlite3_context *context, const char *func_name, int argc, sqlite3_value **argv, int ntypes, int *types, bool check_model);
int sqlite_db_write (sqlite3_context *context, sqlite3 *db, const char *sql, const char **values, int types[], int lens[], int count);
bool sqlite_context_result_error (sqlite3_context *context, int rc, const char *format, ...);
int sqlite_vtab_set_error (sqlite3_vtab *vtab, const char *format, ...);
bool parse_keyvalue_string (sqlite3_context *context, const char *str, keyvalue_callback callback, void *xdata);
char *sqlite_strdup (const char *str);

bool buffer_create (buffer_t *b, uint32_t size);
bool buffer_append (buffer_t *b, const char *data, uint32_t len);
void buffer_destroy (buffer_t *b);

//bool sqlite_set_ptr (sqlite3 *db, void *ptr);
//void *sqlite_get_ptr (sqlite3 *db);
//void sqlite_clear_ptr (sqlite3 *db);

#endif
