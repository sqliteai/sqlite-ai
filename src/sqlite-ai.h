//
//  sqlite-ai.h
//  sqliteai
//
//  Created by Marco Bambini on 26/06/25.
//

#ifndef __SQLITE_AI__
#define __SQLITE_AI__

#ifndef SQLITE_CORE
#include "sqlite3ext.h"
#else
#include "sqlite3.h"
#endif

#ifdef _WIN32
  #define SQLITE_AI_API __declspec(dllexport)
#else
  #define SQLITE_AI_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define SQLITE_AI_VERSION "0.6.1"

SQLITE_AI_API int sqlite3_ai_init (sqlite3 *db, char **pzErrMsg, const sqlite3_api_routines *pApi);

#ifdef __cplusplus
}
#endif


#endif
