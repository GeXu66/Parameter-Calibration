/*
 * MATLAB Compiler: 24.2 (R2024b)
 * 日期: Fri Dec 27 20:05:47 2024
 * 参量:
 * "-B""macro_default""-W""lib:test_fmincon""-T""link:lib""test_fmincon.m"
 */

#ifndef test_fmincon_h
#define test_fmincon_h 1

#if defined(__cplusplus) && !defined(mclmcrrt_h) && defined(__linux__)
#  pragma implementation "mclmcrrt.h"
#endif
#include "mclmcrrt.h"
#ifdef __cplusplus
extern "C" { // sbcheck:ok:extern_c
#endif

/* This symbol is defined in shared libraries. Define it here
 * (to nothing) in case this isn't a shared library. 
 */
#ifndef LIB_test_fmincon_C_API 
#define LIB_test_fmincon_C_API /* No special import/export declaration */
#endif

/* GENERAL LIBRARY FUNCTIONS -- START */

extern LIB_test_fmincon_C_API 
bool MW_CALL_CONV test_fminconInitializeWithHandlers(
       mclOutputHandlerFcn error_handler, 
       mclOutputHandlerFcn print_handler);

extern LIB_test_fmincon_C_API 
bool MW_CALL_CONV test_fminconInitialize(void);
extern LIB_test_fmincon_C_API 
void MW_CALL_CONV test_fminconTerminate(void);

extern LIB_test_fmincon_C_API 
void MW_CALL_CONV test_fminconPrintStackTrace(void);

/* GENERAL LIBRARY FUNCTIONS -- END */

/* C INTERFACE -- MLX WRAPPERS FOR USER-DEFINED MATLAB FUNCTIONS -- START */

extern LIB_test_fmincon_C_API 
bool MW_CALL_CONV mlxTest_fmincon(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[]);

/* C INTERFACE -- MLX WRAPPERS FOR USER-DEFINED MATLAB FUNCTIONS -- END */

/* C INTERFACE -- MLF WRAPPERS FOR USER-DEFINED MATLAB FUNCTIONS -- START */

extern LIB_test_fmincon_C_API bool MW_CALL_CONV mlfTest_fmincon(int nargout, mxArray** x_opt, mxArray* x0, mxArray* A, mxArray* b);

#ifdef __cplusplus
}
#endif
/* C INTERFACE -- MLF WRAPPERS FOR USER-DEFINED MATLAB FUNCTIONS -- END */

#endif
