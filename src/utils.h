//
//  utils.h
//  sqliteai
//
//  Created by Marco Bambini on 27/06/25.
//

#ifndef __SQLITEAI_UTILS__
#define __SQLITEAI_UTILS__

#include <stdbool.h>
#ifndef SQLITE_CORE
#include "sqlite3ext.h"
#else
#include "sqlite3.h"
#endif

typedef bool (*keyvalue_callback)(sqlite3_context *context, void *xdata, const char *key, int key_len, const char *value, int value_len);

bool sqlite_utils_init (void);
bool sqlite_sanity_function (sqlite3_context *context, const char *func_name, int argc, sqlite3_value **argv, int ntypes, int *types, bool check_model);
bool sqlite_context_result_error (sqlite3_context *context, int rc, const char *format, ...);
bool parse_keyvalue_string (sqlite3_context *context, const char *str, keyvalue_callback callback, void *xdata);

bool sqlite_set_ptr (sqlite3 *db, void *ptr);
void *sqlite_get_ptr (sqlite3 *db);
void sqlite_clear_ptr (sqlite3 *db);

#endif
