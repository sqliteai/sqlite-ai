//
//  utils.h
//  sqliteai
//
//  Created by Marco Bambini on 27/06/25.
//

#ifndef __SQLITEAI_UTILS__
#define __SQLITEAI_UTILS__

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifndef SQLITE_CORE
#include "sqlite3ext.h"
#else
#include "sqlite3.h"
#endif

#define UUID_STR_MAXLEN                         37

typedef struct {
    char                *data;                  // raw buffer
    uint32_t            capacity;               // size of the buffer
    uint32_t            length;                 // currently used size
} buffer_t;

// callbacks
typedef bool (*keyvalue_callback)(void *xdata, const char *key, int key_len, const char *value, int value_len);
typedef void (*audio_list_devices_callback)(uint32_t count, uint32_t index, const char *name, bool is_default, void *xdata);

bool parse_keyvalue_string (const char *str, keyvalue_callback callback, void *xdata);

bool sqlite_sanity_function (sqlite3_context *context, const char *func_name, int argc, sqlite3_value **argv, int ntypes, int *types, bool check_llm_model, bool check_audio_model);
int  sqlite_db_write (sqlite3_context *context, sqlite3 *db, const char *sql, const char **values, int types[], int lens[], int count);
int  sqlite_db_write_simple (sqlite3_context *context, sqlite3 *db, const char *sql);
bool sqlite_context_result_error (sqlite3_context *context, int rc, const char *format, ...);
int  sqlite_vtab_set_error (sqlite3_vtab *vtab, const char *format, ...);
void sqlite_common_set_error (sqlite3_context *context, sqlite3_vtab *vtab, int rc, const char *format, ...);
char *sqlite_strdup (const char *str);

bool buffer_create (buffer_t *b, uint32_t size);
bool buffer_append (buffer_t *b, const char *data, uint32_t len, bool zero_terminate);
bool buffer_resize (buffer_t *b, uint32_t new_capacity);
void buffer_reset (buffer_t *b);
void buffer_destroy (buffer_t *b);

char *ai_uuid_v7_string (char value[UUID_STR_MAXLEN], bool dash_format);

float *audio_wav_file2pcm (const char *wav_path, uint64_t *num_samples, uint32_t *sample_rate, uint16_t *channels);
float *audio_wav_mem2pcm (const void *data, size_t data_size, uint64_t *num_samples, uint32_t *sample_rate, uint16_t *channels);
float *audio_flac_file2pcm (const char *flac_path, uint64_t *num_samples, uint32_t *sample_rate, uint16_t *channels);
float *audio_flac_mem2pcm (const void *data, size_t data_size, uint64_t *num_samples, uint32_t *sample_rate, uint16_t *channels);
float *audio_mp3_file2pcm (const char *mp3_path, uint64_t *num_samples, uint32_t *sample_rate, uint16_t *channels);
float *audio_mp3_mem2pcm (const void *data, size_t data_size, uint64_t *num_samples, uint32_t *sample_rate, uint32_t *channels);
int    audio_list_devices (void *xdata, audio_list_devices_callback input_devices_cb, audio_list_devices_callback output_devices_cb);

#endif
