#pragma once
#include <iostream>

// #if defined(_WIN32) || defined(_WIN64)
// #include <intrin.h>
// #else
// #include <cpuid.h>
// #endif

#if defined(_WIN32) || defined(_WIN64)
#include <intrin.h>
#elif defined(__linux__) || defined(__APPLE__)
#include <immintrin.h>
#include <cpuid.h>
#include <x86intrin.h>
#endif

#ifndef _XCR_XFEATURE_ENABLED_MASK
#define _XCR_XFEATURE_ENABLED_MASK 0
#endif

void cpuid(int registers[4], int function_id, int subfunction_id);

bool detect_avx();
bool detect_avx2();
bool detect_avx512();
void print_avx_features();