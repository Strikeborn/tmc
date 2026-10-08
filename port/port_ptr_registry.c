#include "port_ptr_registry.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char* key;
    const unsigned char* base;
    size_t size;
} PtrBlock;

static PtrBlock* sBlocks;
static size_t sCount, sCap;
static PortPtrBlockLoader sLoader;

void Port_PtrBlock_Register(const char* key, const void* base, size_t size) {
    if (!key || !base || !size)
        return;
    for (size_t i = 0; i < sCount; ++i)
        if (strcmp(sBlocks[i].key, key) == 0) {
            sBlocks[i].base = (const unsigned char*)base;
            sBlocks[i].size = size;
            return;
        }
    if (sCount == sCap) {
        const size_t cap = sCap ? sCap * 2 : 256;
        PtrBlock* grown = (PtrBlock*)realloc(sBlocks, cap * sizeof(PtrBlock));
        if (!grown)
            return;
        sBlocks = grown, sCap = cap;
    }
    const size_t len = strlen(key) + 1;
    char* copy = (char*)malloc(len);
    if (!copy)
        return;
    memcpy(copy, key, len);
    sBlocks[sCount++] = (PtrBlock){ copy, (const unsigned char*)base, size };
}

void Port_PtrBlock_UnregisterPrefix(const char* prefix) {
    const size_t n = strlen(prefix);
    size_t out = 0;
    for (size_t i = 0; i < sCount; ++i) {
        if (strncmp(sBlocks[i].key, prefix, n) == 0) {
            free(sBlocks[i].key);
            continue;
        }
        sBlocks[out++] = sBlocks[i];
    }
    sCount = out;
}

int Port_PtrBlock_Lookup(const void* p, const char** key, size_t* offset) {
    const unsigned char* at = (const unsigned char*)p;
    for (size_t i = 0; i < sCount; ++i)
        if (at >= sBlocks[i].base && (size_t)(at - sBlocks[i].base) < sBlocks[i].size) {
            *key = sBlocks[i].key;
            *offset = (size_t)(at - sBlocks[i].base);
            return 1;
        }
    return 0;
}

static void* Find(const char* key, size_t offset) {
    for (size_t i = 0; i < sCount; ++i)
        if (strcmp(sBlocks[i].key, key) == 0)
            return offset < sBlocks[i].size ? (void*)(sBlocks[i].base + offset) : NULL;
    return NULL;
}

void* Port_PtrBlock_Resolve(const char* key, size_t offset) {
    void* p = Find(key, offset);
    if (!p && sLoader && sLoader(key))
        p = Find(key, offset);
    return p;
}

void Port_PtrBlock_SetLoader(PortPtrBlockLoader loader) {
    sLoader = loader;
}
