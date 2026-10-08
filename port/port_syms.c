#include "port_syms.h"

#include <SDL3/SDL.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint64_t addr; /* link-time */
    char* name;
} Sym;

static Sym* sByAddr;  /* file order: ascending address */
static Sym** sByName; /* sorted by name */
static size_t sCount;
static int sState;     /* 0 not loaded, 1 ok, -1 missing or stale */
static int64_t sSlide; /* runtime address - link-time address */

static int CmpName(const void* a, const void* b) {
    return strcmp((*(Sym* const*)a)->name, (*(Sym* const*)b)->name);
}

static const Sym* FindName(const char* name) {
    size_t lo = 0, hi = sCount;
    while (lo < hi) {
        const size_t mid = (lo + hi) / 2;
        const int c = strcmp(sByName[mid]->name, name);
        if (c == 0)
            return sByName[mid];
        if (c < 0)
            lo = mid + 1;
        else
            hi = mid;
    }
    return NULL;
}

static void Load(void) {
    sState = -1;
    const char* base = SDL_GetBasePath();
    char path[1024];
    snprintf(path, sizeof(path), "%stmc_pc.syms", base ? base : "");
    FILE* f = fopen(path, "rb");
    if (!f)
        return;
    size_t cap = 0;
    char line[1024];
    while (fgets(line, sizeof(line), f)) {
        char* sp = strchr(line, ' ');
        if (!sp || line[0] == '#')
            continue;
        *sp = 0;
        char* name = sp + 1;
        name[strcspn(name, "\r\n")] = 0;
        if (sCount == cap) {
            cap = cap ? cap * 2 : 4096;
            Sym* grown = (Sym*)realloc(sByAddr, cap * sizeof(Sym));
            if (!grown)
                break;
            sByAddr = grown;
        }
        sByAddr[sCount].addr = strtoull(line, NULL, 16);
        sByAddr[sCount].name = strdup(name);
        if (sByAddr[sCount].name)
            ++sCount;
    }
    fclose(f);
    if (!sCount || !(sByName = (Sym**)malloc(sCount * sizeof(Sym*))))
        return;
    for (size_t i = 0; i < sCount; ++i)
        sByName[i] = &sByAddr[i];
    qsort(sByName, sCount, sizeof(Sym*), CmpName);
    /* Anchor on this module's functions; a second one catches a stale file. */
    const Sym* a = FindName("Port_Syms_Available");
    const Sym* b = FindName("Port_Syms_Resolve");
    if (!a || !b ||
        (int64_t)(b->addr - a->addr) != (int64_t)((uintptr_t)&Port_Syms_Resolve - (uintptr_t)&Port_Syms_Available)) {
        fprintf(stderr, "[syms] %s does not match this binary; save states stay build-specific\n", path);
        return;
    }
    sSlide = (int64_t)((uintptr_t)&Port_Syms_Available - a->addr);
    sState = 1;
}

int Port_Syms_Available(void) {
    if (sState == 0)
        Load();
    return sState == 1;
}

int Port_Syms_Lookup(const void* p, const char** name, uint64_t* offset) {
    if (!Port_Syms_Available())
        return 0;
    const uint64_t addr = (uint64_t)((int64_t)(uintptr_t)p - sSlide);
    size_t lo = 0, hi = sCount;
    while (lo < hi) { /* first symbol above addr */
        const size_t mid = (lo + hi) / 2;
        if (sByAddr[mid].addr <= addr)
            lo = mid + 1;
        else
            hi = mid;
    }
    if (lo == 0)
        return 0;
    *name = sByAddr[lo - 1].name;
    *offset = addr - sByAddr[lo - 1].addr;
    return 1;
}

void* Port_Syms_Resolve(const char* name) {
    if (!Port_Syms_Available())
        return NULL;
    const Sym* s = FindName(name);
    return s ? (void*)(uintptr_t)((int64_t)s->addr + sSlide) : NULL;
}
