#pragma once
#include <iostream>
#if defined(_WIN32) || defined(_WIN64)
#include <intrin.h>
#else
#include <cpuid.h>
#endif

void cpuid(int registers[4], int function_id);
bool detect_avx();
bool detect_avx2();
bool detect_avx512();
void print_avx_features();