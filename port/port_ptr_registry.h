#ifndef PORT_PTR_REGISTRY_H
#define PORT_PTR_REGISTRY_H

/* Named heap blocks game state may point into (asset files, area tables,
 * room property arrays, map definitions...). A save state records a pointer
 * into one as (key, offset) and resolves it again in another process, where
 * the block lives at a different address or hasn't been built yet. Keys are
 * stable: they name the data ("file:<path>", "props:<area>:<room>"), not the
 * allocation. */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Registering an existing key replaces it. */
void Port_PtrBlock_Register(const char* key, const void* base, size_t size);
/* Drops every block whose key starts with prefix (a rebuilt area). */
void Port_PtrBlock_UnregisterPrefix(const char* prefix);
/* The block containing p: its key and p's offset in it. Returns 0 if none. */
int Port_PtrBlock_Lookup(const void* p, const char** key, size_t* offset);
/* key + offset in this process, building the block through the loader if it
 * isn't there yet. NULL if it can't be resolved. */
void* Port_PtrBlock_Resolve(const char* key, size_t offset);
/* Builds the block for key (returns nonzero if it then exists). */
typedef int (*PortPtrBlockLoader)(const char* key);
void Port_PtrBlock_SetLoader(PortPtrBlockLoader loader);

#ifdef __cplusplus
}
#endif

#endif
