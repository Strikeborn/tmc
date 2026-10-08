#ifndef PORT_SYMS_H
#define PORT_SYMS_H

/* The program's own symbol table, from tmc_pc.syms written beside the binary
 * at build time (xmake.lua). Save states name pointers into code and
 * constants by symbol so a later build, whose layout differs, can load them.
 * Link-time addresses are anchored to this module's own function, so the
 * table works wherever the OS loads the program. */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Nonzero when tmc_pc.syms exists and matches this binary. */
int Port_Syms_Available(void);
/* The symbol at or before p: its name (unique: the build gives a repeated
 * static name "#2", "#3"... in address order) and p's offset into it. */
int Port_Syms_Lookup(const void* p, const char** name, uint64_t* offset);
/* A symbol's address in this process, or NULL. */
void* Port_Syms_Resolve(const char* name);

#ifdef __cplusplus
}
#endif

#endif
