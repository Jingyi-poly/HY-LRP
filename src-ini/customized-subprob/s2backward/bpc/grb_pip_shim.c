#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>

typedef void *GRBenv;

typedef int (*emptyenv_fn)(GRBenv **, int, int, int, int,
                           void*, void*, void*, long, long, long, long);
typedef int (*startenv_fn)(GRBenv *);

static void *grb_handle = NULL;
static emptyenv_fn real_emptyenv = NULL;
static startenv_fn real_startenv = NULL;

static void init_real_fns(void) {
    if (grb_handle) return;
    /* Try RTLD_NEXT first (works for standalone binaries). */
    real_emptyenv = (emptyenv_fn)dlsym(RTLD_NEXT, "GRBemptyenvadvinternal");
    real_startenv = (startenv_fn)dlsym(RTLD_NEXT, "GRBstartenv");
    if (real_emptyenv && real_startenv) {
        grb_handle = (void*)1;  /* sentinel: RTLD_NEXT worked */
        return;
    }
    /* For Python/dlopen case: open the library by name. */
    const char *lib_path = getenv("GRB_PIP_SHIM_LIBPATH");
    if (!lib_path) lib_path = "libgurobi130.so";
    grb_handle = dlopen(lib_path, RTLD_NOW | RTLD_NOLOAD);
    if (!grb_handle) grb_handle = dlopen(lib_path, RTLD_NOW);
    if (!grb_handle) {
        fprintf(stderr, "[grb_pip_shim] ERROR: cannot dlopen %s: %s\n", lib_path, dlerror());
        return;
    }
    real_emptyenv = (emptyenv_fn)dlsym(grb_handle, "GRBemptyenvadvinternal");
    real_startenv = (startenv_fn)dlsym(grb_handle, "GRBstartenv");
}

int GRBemptyenvadvinternal(GRBenv **envP, int apitype,
                           int major, int minor, int tech,
                           void *p1, void *p2, void *p3,
                           long s1, long s2, long s3, long s4) {
    init_real_fns();
    if (!real_emptyenv) return 10001;
    /* Force apitype=0 (generic) — same as gurobipy uses. */
    return real_emptyenv(envP, 0, major, minor, tech, p1, p2, p3, s1, s2, s3, s4);
}

int GRBstartenv(GRBenv *env) {
    init_real_fns();
    if (!real_startenv) return 10001;
    return real_startenv(env);
}
