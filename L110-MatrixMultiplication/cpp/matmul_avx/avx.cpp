#include "avx.h"

void cpuid(int registers[4], int function_id) {
#if defined(_WIN32) || defined(_WIN64)
  __cpuid(registers, function_id);
#else
  __cpuid(function_id, registers[0], registers[1], registers[2], registers[3]);
#endif
}

bool detect_avx() {
  int registers[4];
  cpuid(registers, 0);

  if (registers[0] < 1) {
    return false;
  }

  cpuid(registers, 1);
  bool osxsave = (registers[2] & (1 << 27)) != 0;
  bool avx = (registers[2] & (1 << 28)) != 0;

  if (osxsave && avx) {
    unsigned long long xcrFeatureMask = _xgetbv(_XCR_XFEATURE_ENABLED_MASK);
    return (xcrFeatureMask & 0x6) == 0x6;
  }

  return false;
}

bool detect_avx2() {
  int registers[4];
  cpuid(registers, 0);

  if (registers[0] < 7) {
    return false;
  }

  cpuid(registers, 1);
  bool osxsave = (registers[2] & (1 << 27)) != 0;

  cpuid(registers, 7);
  bool avx2 = (registers[1] & (1 << 5)) != 0;

  if (osxsave && avx2) {
    unsigned long long xcrFeatureMask = _xgetbv(_XCR_XFEATURE_ENABLED_MASK);
    return (xcrFeatureMask & 0x6) == 0x6;
  }

  return false;
}

bool detect_avx512() {
  int registers[4];

  // Get the highest function parameter for CPUID
  cpuid(registers, 0);
  if (registers[0] < 7) {
    return false;
  }

  // Check if OS has enabled XGETBV
  cpuid(registers, 1);
  bool osxsave = (registers[2] & (1 << 27)) != 0;
  if (!osxsave) {
    return false;
  }

  // Check for AVX-512 support in CPUID leaf 7
  cpuid(registers, 7);
  bool avx512f = (registers[1] & (1 << 16)) != 0;
  if (!avx512f) {
    return false;
  }

  // Check if OS has enabled AVX-512 by examining XCR0
  unsigned long long xcrFeatureMask = _xgetbv(_XCR_XFEATURE_ENABLED_MASK);
  // Check if the OS has enabled the necessary AVX-512 features (XMM, YMM, ZMM)
  return (xcrFeatureMask & 0xe6) == 0xe6;
}

void print_avx_features() {
  std::cout << "AVX support: " << (detect_avx() ? "Yes" : "No") << std::endl;
  std::cout << "AVX2 support: " << (detect_avx2() ? "Yes" : "No") << std::endl;
  std::cout << "AVX-512 support: " << (detect_avx512() ? "Yes" : "No")
            << std::endl;
}
