#include "avx.h"

void cpuid(int registers[4], int function_id, int subfunction_id = 0) {
#ifdef _WIN32
  __cpuidex(registers, function_id, subfunction_id);
#elif defined(__linux__) || defined(__APPLE__)
  __cpuid_count(function_id, subfunction_id, registers[0], registers[1],
                registers[2], registers[3]);
#endif
}

bool check_osxsave() {
  int registers[4];
  cpuid(registers, 1);
  return (registers[2] & (1 << 27)) != 0;
}

bool check_xcr_feature_mask() {
  unsigned long long xcrFeatureMask =
      _xgetbv(0);  // Use 0 instead of _XCR_XFEATURE_ENABLED_MASK
  return (xcrFeatureMask & 0x6) == 0x6;
}

bool check_xcr_feature_mask_avx512() {
  unsigned long long xcrFeatureMask = _xgetbv(0);
  return (xcrFeatureMask & 0xe6) == 0xe6;
}

bool detect_avx() {
  int registers[4];
  cpuid(registers, 1);
  bool avx = (registers[2] & (1 << 28)) != 0;

  return check_osxsave() && avx && check_xcr_feature_mask();
}
bool detect_avx2() {
  int registers[4];
  cpuid(registers, 0);

  if (registers[0] < 7) {
    return false;
  }

  cpuid(registers, 7);
  bool avx2 = (registers[1] & (1 << 5)) != 0;

  return check_osxsave() && avx2 && check_xcr_feature_mask();
}

bool detect_avx512() {
  // Check for CPUID level 7 support
  int cpuid_info[4] = {0};
  cpuid(cpuid_info, 1);
  if (cpuid_info[0] < 7) {
    std::cout << "CPUID level 7 is not supported" << std::endl;
    return false;
  }

  // Check for XCR support for AVX512
  // Check for the essential bits for AVX-512 (XMM, YMM, and ZMM support)
  unsigned long long xcr_feature_mask = _xgetbv(0);
  if ((xcr_feature_mask & 0xe6) != 0xe6) {
    std::cout << "XCR feature mask for AVX512 is not supported" << std::endl;
    return false;
  }

  // Check for OS XSAVE support
  bool os_xsave_supported = check_osxsave();
  if (!os_xsave_supported) {
    std::cout << "OS XSAVE is not supported" << std::endl;
    return false;
  }

  return true;
}

void print_avx_features() {
  std::cout << "AVX support: " << (detect_avx() ? "Yes" : "No") << std::endl;
  std::cout << "AVX2 support: " << (detect_avx2() ? "Yes" : "No") << std::endl;
  std::cout << "AVX-512 support: " << (detect_avx512() ? "Yes" : "No")
            << std::endl;
}
