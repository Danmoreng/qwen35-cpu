// SPDX-License-Identifier: MIT
#pragma once
// Single-owner C ABI used by the browser worker and the native parity oracle.
// Every return value is JSON (including errors). The returned UTF-8 string is
// owned by the module and remains valid until the next API call.
#ifdef __cplusplus
extern "C" {
#endif
const char *qev_load(const char *directory, int threads, int context);
// Optional tuning before loading; defaults remain chunk=32 and profile=0.
const char *qev_options(int chunk, int profile);
const char *qev_unload(void);
const char *qev_begin(const char *request_json);
const char *qev_step(void);
const char *qev_cancel(void);
// Diagnostic prompt IDs and raw option logits for the last request.
const char *qev_debug(void);
#ifdef __cplusplus
}
#endif
