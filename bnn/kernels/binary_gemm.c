/*
 * Packed binary XNOR+popcount GEMM + ternary bitplane GEMM.
 * Encoding (binary): bit 0 => +1, bit 1 => -1
 * Dot: <a,b> = N - 2 * popcount(a XOR b)
 *
 * Portability is a hard requirement: one source file must build and run
 * correctly on x86-64 (MSVC / GCC / Clang), ARM64 (Linux / macOS), and any
 * other architecture via the scalar path. The fastest legal path is chosen at
 * RUN time, so a binary built on one machine stays correct on another.
 *
 *   AVX-512 VPOPCNTDQ  hardware 64-bit vector popcount (Ice Lake+, Zen 4+)
 *   AVX2               nibble-LUT popcount via vpshufb (Haswell+, 2013+)
 *   NEON               vcntq_u8 + vpadalq (all ARM64)
 *   scalar             __popcnt64 / __builtin_popcountll (always available)
 *
 * Override with BNN_KERNEL=scalar|avx2|avx512|neon (validation / repro).
 *
 * Blocking: output rows M are shared across a block of 4 batch rows, so each
 * weight word is loaded once and reused 4x, and the OpenMP team is forked ONCE
 * per call instead of once per batch row.
 *
 * Note: MSVC OpenMP 2.0 requires the parallel-for index declared outside the
 * for-init (no C99 for(int i=...)) and has no `collapse`.
 */
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Architecture / compiler capability probing                          */
/* ------------------------------------------------------------------ */
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#  define BNN_X86 1
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
#  define BNN_ARM64 1
#endif

#if defined(BNN_X86)
#  if defined(_MSC_VER)
     /* MSVC emits intrinsics for ISAs beyond /arch:, so both paths compile
      * without making the whole DLL require AVX-512 to load. */
#    define BNN_HAVE_AVX2 1
#    if _MSC_VER >= 1920            /* VS 2019+ */
#      define BNN_HAVE_AVX512 1
#    endif
#    define BNN_TARGET(s)
#  elif defined(__GNUC__) || defined(__clang__)
#    define BNN_HAVE_AVX2 1
#    define BNN_TARGET(s) __attribute__((target(s)))
#    if (defined(__clang__) && __clang_major__ >= 6) || \
        (!defined(__clang__) && defined(__GNUC__) && __GNUC__ >= 7)
#      define BNN_HAVE_AVX512 1
#    endif
#  endif
#endif

#if defined(BNN_HAVE_AVX2) || defined(BNN_HAVE_AVX512)
#  include <immintrin.h>
#endif

#if defined(BNN_ARM64)
#  include <arm_neon.h>
#  define BNN_HAVE_NEON 1
#endif

#ifndef BNN_TARGET
#  define BNN_TARGET(s)
#endif

#ifdef _MSC_VER
#  include <intrin.h>
static inline int pop64(uint64_t x) { return (int)__popcnt64(x); }
#  define BNN_EXPORT __declspec(dllexport)
#else
static inline int pop64(uint64_t x) { return __builtin_popcountll(x); }
#  define BNN_EXPORT
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

/* Batch rows processed against a single weight row per pass. */
#define BNN_BR 4

enum {
    BNN_KERNEL_SCALAR = 0,
    BNN_KERNEL_AVX2   = 1,
    BNN_KERNEL_AVX512 = 2,
    BNN_KERNEL_NEON   = 3
};

/* ------------------------------------------------------------------ */
/* Thread control                                                      */
/* ------------------------------------------------------------------ */
static int g_num_threads = 0;   /* 0 => library / OpenMP default */

static inline int effective_threads(void) {
#ifdef _OPENMP
    if (g_num_threads > 0) {
        return g_num_threads;
    }
    return omp_get_max_threads();
#else
    (void)g_num_threads;
    return 1;
#endif
}

#ifdef _WIN32
__declspec(dllexport)
#endif
void binary_gemm_set_num_threads(int n) {
    g_num_threads = n > 0 ? n : 0;
#ifdef _OPENMP
    if (n > 0) {
        omp_set_num_threads(n);
    }
#endif
}

#ifdef _WIN32
__declspec(dllexport)
#endif
int binary_gemm_get_num_threads(void) {
    return effective_threads();
}

#ifdef _WIN32
__declspec(dllexport)
#endif
int binary_gemm_openmp_enabled(void) {
#ifdef _OPENMP
    return 1;
#else
    return 0;
#endif
}

/* ------------------------------------------------------------------ */
/* Runtime CPU detection                                               */
/* ------------------------------------------------------------------ */
#if defined(BNN_X86)
static void bnn_cpuid(int leaf, int subleaf, unsigned int regs[4]) {
#if defined(_MSC_VER)
    int out[4];
    __cpuidex(out, leaf, subleaf);
    regs[0] = (unsigned int)out[0];
    regs[1] = (unsigned int)out[1];
    regs[2] = (unsigned int)out[2];
    regs[3] = (unsigned int)out[3];
#else
    unsigned int a, b, c, d;
    __asm__ __volatile__("cpuid"
                         : "=a"(a), "=b"(b), "=c"(c), "=d"(d)
                         : "a"(leaf), "c"(subleaf));
    regs[0] = a; regs[1] = b; regs[2] = c; regs[3] = d;
#endif
}

static uint64_t bnn_xgetbv0(void) {
#if defined(_MSC_VER)
    return (uint64_t)_xgetbv(0);
#else
    unsigned int lo, hi;
    __asm__ __volatile__(".byte 0x0f, 0x01, 0xd0" : "=a"(lo), "=d"(hi) : "c"(0));
    return ((uint64_t)hi << 32) | lo;
#endif
}
#endif /* BNN_X86 */

/* Pick the best ISA this CPU *and* this OS actually support. */
static int bnn_detect_kernel(void) {
#if defined(BNN_X86)
    unsigned int r0[4], r1[4], r7[4];
    uint64_t xcr0;
    int have_osxsave, have_avx, have_avx2, have_f, have_vpopcnt;

    bnn_cpuid(0, 0, r0);
    if (r0[0] < 1) {
        return BNN_KERNEL_SCALAR;
    }
    bnn_cpuid(1, 0, r1);
    have_osxsave = (r1[2] & (1u << 27)) != 0;   /* ECX.OSXSAVE */
    have_avx     = (r1[2] & (1u << 28)) != 0;   /* ECX.AVX     */
    if (!have_osxsave || !have_avx) {
        return BNN_KERNEL_SCALAR;
    }

    /* The OS must have enabled XMM+YMM state saving, or AVX faults. */
    xcr0 = bnn_xgetbv0();
    if ((xcr0 & 0x6) != 0x6) {
        return BNN_KERNEL_SCALAR;
    }

    if (r0[0] < 7) {
        return BNN_KERNEL_SCALAR;
    }
    bnn_cpuid(7, 0, r7);
    have_avx2    = (r7[1] & (1u << 5))  != 0;   /* EBX.AVX2               */
    have_f       = (r7[1] & (1u << 16)) != 0;   /* EBX.AVX512F            */
    have_vpopcnt = (r7[2] & (1u << 14)) != 0;   /* ECX.AVX512_VPOPCNTDQ   */

#if defined(BNN_HAVE_AVX512)
    /* ZMM / opmask state must also be OS-enabled (bits 5,6,7). */
    if (have_f && have_vpopcnt && (xcr0 & 0xE6) == 0xE6) {
        return BNN_KERNEL_AVX512;
    }
#else
    (void)have_f; (void)have_vpopcnt;
#endif
#if defined(BNN_HAVE_AVX2)
    if (have_avx2) {
        return BNN_KERNEL_AVX2;
    }
#else
    (void)have_avx2;
#endif
    return BNN_KERNEL_SCALAR;
#elif defined(BNN_HAVE_NEON)
    return BNN_KERNEL_NEON;   /* NEON is mandatory on ARM64 */
#else
    return BNN_KERNEL_SCALAR;
#endif
}

static int bnn_kernel_available(int id) {
    if (id == BNN_KERNEL_SCALAR) {
        return 1;
    }
#if defined(BNN_X86)
    {
        int best = bnn_detect_kernel();
        if (id == BNN_KERNEL_AVX512) {
            return best == BNN_KERNEL_AVX512;
        }
        if (id == BNN_KERNEL_AVX2) {
            return best == BNN_KERNEL_AVX512 || best == BNN_KERNEL_AVX2;
        }
    }
#endif
#if defined(BNN_HAVE_NEON)
    if (id == BNN_KERNEL_NEON) {
        return 1;
    }
#endif
    return 0;
}

static int g_kernel = -1;   /* -1 => not yet resolved */

static int bnn_kernel_from_env(void) {
    const char* s;
#if defined(_MSC_VER)
    /* getenv is fine here; we never write to the returned buffer. */
#  pragma warning(suppress : 4996)
    s = getenv("BNN_KERNEL");
#else
    s = getenv("BNN_KERNEL");
#endif
    if (s == NULL || s[0] == '\0') {
        return -1;
    }
    if (strcmp(s, "scalar") == 0) return BNN_KERNEL_SCALAR;
    if (strcmp(s, "avx2")   == 0) return BNN_KERNEL_AVX2;
    if (strcmp(s, "avx512") == 0) return BNN_KERNEL_AVX512;
    if (strcmp(s, "neon")   == 0) return BNN_KERNEL_NEON;
    return -1;
}

static int bnn_kernel(void) {
    if (g_kernel < 0) {
        int want = bnn_kernel_from_env();
        if (want >= 0 && bnn_kernel_available(want)) {
            g_kernel = want;
        } else {
            g_kernel = bnn_detect_kernel();
        }
    }
    return g_kernel;
}

#ifdef _WIN32
__declspec(dllexport)
#endif
int binary_gemm_kernel_id(void) {
    return bnn_kernel();
}

/* Force a path. Returns the id actually in effect (falls back to scalar). */
#ifdef _WIN32
__declspec(dllexport)
#endif
int binary_gemm_set_kernel(int id) {
    if (id < 0) {
        g_kernel = -1;              /* re-detect */
    } else if (bnn_kernel_available(id)) {
        g_kernel = id;
    } else {
        g_kernel = BNN_KERNEL_SCALAR;
    }
    return bnn_kernel();
}

/* Bit 0 avx2, bit 1 avx512vpopcntdq, bit 2 neon — what the CPU can do. */
#ifdef _WIN32
__declspec(dllexport)
#endif
int binary_gemm_cpu_features(void) {
    int f = 0;
    if (bnn_kernel_available(BNN_KERNEL_AVX2))   f |= 1;
    if (bnn_kernel_available(BNN_KERNEL_AVX512)) f |= 2;
    if (bnn_kernel_available(BNN_KERNEL_NEON))   f |= 4;
    return f;
}

/* ------------------------------------------------------------------ */
/* Per-ISA row kernels                                                 */
/*                                                                     */
/* hamming4_*: Hamming distance of one weight row against BNN_BR       */
/* activation rows, reusing each loaded weight word BNN_BR times.      */
/* hamming1_*: single-row remainder.                                   */
/* ------------------------------------------------------------------ */
typedef void (*hamming4_fn)(const uint64_t* x0, const uint64_t* x1,
                            const uint64_t* x2, const uint64_t* x3,
                            const uint64_t* wm, int words, int* out4);
typedef int (*hamming1_fn)(const uint64_t* x, const uint64_t* wm, int words);

/* ---- scalar ------------------------------------------------------ */
static int hamming1_scalar(const uint64_t* x, const uint64_t* wm, int words) {
    int dist = 0;
    int w;
    /* 4-wide unroll breaks the accumulator dependency chain. */
    for (w = 0; w + 3 < words; w += 4) {
        dist += pop64(x[w] ^ wm[w]);
        dist += pop64(x[w + 1] ^ wm[w + 1]);
        dist += pop64(x[w + 2] ^ wm[w + 2]);
        dist += pop64(x[w + 3] ^ wm[w + 3]);
    }
    for (; w < words; ++w) {
        dist += pop64(x[w] ^ wm[w]);
    }
    return dist;
}

static void hamming4_scalar(const uint64_t* x0, const uint64_t* x1,
                            const uint64_t* x2, const uint64_t* x3,
                            const uint64_t* wm, int words, int* out4) {
    int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
    int e0 = 0, e1 = 0, e2 = 0, e3 = 0;
    const int w_pair = (words / 2) * 2;
    int w;
    /* Two words in flight per row: 8 independent popcounts hide latency. */
    for (w = 0; w < w_pair; w += 2) {
        const uint64_t v0 = wm[w];
        const uint64_t v1 = wm[w + 1];
        d0 += pop64(x0[w] ^ v0); e0 += pop64(x0[w + 1] ^ v1);
        d1 += pop64(x1[w] ^ v0); e1 += pop64(x1[w + 1] ^ v1);
        d2 += pop64(x2[w] ^ v0); e2 += pop64(x2[w + 1] ^ v1);
        d3 += pop64(x3[w] ^ v0); e3 += pop64(x3[w + 1] ^ v1);
    }
    for (; w < words; ++w) {
        const uint64_t v = wm[w];
        d0 += pop64(x0[w] ^ v);
        d1 += pop64(x1[w] ^ v);
        d2 += pop64(x2[w] ^ v);
        d3 += pop64(x3[w] ^ v);
    }
    out4[0] = d0 + e0;
    out4[1] = d1 + e1;
    out4[2] = d2 + e2;
    out4[3] = d3 + e3;
}

/* ---- AVX2 (nibble LUT popcount) ---------------------------------- */
#if defined(BNN_HAVE_AVX2)
BNN_TARGET("avx2")
static inline __m256i bnn_popcnt_epi64_avx2(__m256i v) {
    const __m256i lut = _mm256_setr_epi8(
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
    const __m256i lo_mask = _mm256_set1_epi8(0x0f);
    const __m256i lo = _mm256_and_si256(v, lo_mask);
    const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), lo_mask);
    const __m256i cnt = _mm256_add_epi8(_mm256_shuffle_epi8(lut, lo),
                                        _mm256_shuffle_epi8(lut, hi));
    /* SAD against zero sums each 8-byte group into a 64-bit lane. */
    return _mm256_sad_epu8(cnt, _mm256_setzero_si256());
}

BNN_TARGET("avx2")
static inline int bnn_hsum_epi64_avx2(__m256i v) {
    uint64_t t[4];
    _mm256_storeu_si256((__m256i*)t, v);
    return (int)(t[0] + t[1] + t[2] + t[3]);
}

BNN_TARGET("avx2")
static int hamming1_avx2(const uint64_t* x, const uint64_t* wm, int words) {
    const int w_vec = (words / 4) * 4;
    __m256i acc = _mm256_setzero_si256();
    int dist = 0;
    int w;
    for (w = 0; w < w_vec; w += 4) {
        const __m256i d = _mm256_xor_si256(
            _mm256_loadu_si256((const __m256i*)(x + w)),
            _mm256_loadu_si256((const __m256i*)(wm + w)));
        acc = _mm256_add_epi64(acc, bnn_popcnt_epi64_avx2(d));
    }
    for (; w < words; ++w) {
        dist += pop64(x[w] ^ wm[w]);
    }
    return dist + bnn_hsum_epi64_avx2(acc);
}

BNN_TARGET("avx2")
static void hamming4_avx2(const uint64_t* x0, const uint64_t* x1,
                          const uint64_t* x2, const uint64_t* x3,
                          const uint64_t* wm, int words, int* out4) {
    const int w_vec = (words / 4) * 4;
    __m256i a0 = _mm256_setzero_si256(), a1 = _mm256_setzero_si256();
    __m256i a2 = _mm256_setzero_si256(), a3 = _mm256_setzero_si256();
    int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
    int w;
    for (w = 0; w < w_vec; w += 4) {
        const __m256i wv = _mm256_loadu_si256((const __m256i*)(wm + w));
        a0 = _mm256_add_epi64(a0, bnn_popcnt_epi64_avx2(_mm256_xor_si256(
            _mm256_loadu_si256((const __m256i*)(x0 + w)), wv)));
        a1 = _mm256_add_epi64(a1, bnn_popcnt_epi64_avx2(_mm256_xor_si256(
            _mm256_loadu_si256((const __m256i*)(x1 + w)), wv)));
        a2 = _mm256_add_epi64(a2, bnn_popcnt_epi64_avx2(_mm256_xor_si256(
            _mm256_loadu_si256((const __m256i*)(x2 + w)), wv)));
        a3 = _mm256_add_epi64(a3, bnn_popcnt_epi64_avx2(_mm256_xor_si256(
            _mm256_loadu_si256((const __m256i*)(x3 + w)), wv)));
    }
    for (; w < words; ++w) {
        const uint64_t v = wm[w];
        d0 += pop64(x0[w] ^ v);
        d1 += pop64(x1[w] ^ v);
        d2 += pop64(x2[w] ^ v);
        d3 += pop64(x3[w] ^ v);
    }
    out4[0] = d0 + bnn_hsum_epi64_avx2(a0);
    out4[1] = d1 + bnn_hsum_epi64_avx2(a1);
    out4[2] = d2 + bnn_hsum_epi64_avx2(a2);
    out4[3] = d3 + bnn_hsum_epi64_avx2(a3);
}
#endif /* BNN_HAVE_AVX2 */

/* ---- AVX-512 VPOPCNTDQ (hardware vector popcount) ---------------- */
#if defined(BNN_HAVE_AVX512)
BNN_TARGET("avx512f,avx512vpopcntdq")
static int hamming1_avx512(const uint64_t* x, const uint64_t* wm, int words) {
    const int w_vec = (words / 8) * 8;
    __m512i acc = _mm512_setzero_si512();
    int dist = 0;
    int w;
    for (w = 0; w < w_vec; w += 8) {
        const __m512i d = _mm512_xor_si512(
            _mm512_loadu_si512((const void*)(x + w)),
            _mm512_loadu_si512((const void*)(wm + w)));
        acc = _mm512_add_epi64(acc, _mm512_popcnt_epi64(d));
    }
    for (; w < words; ++w) {
        dist += pop64(x[w] ^ wm[w]);
    }
    return dist + (int)_mm512_reduce_add_epi64(acc);
}

BNN_TARGET("avx512f,avx512vpopcntdq")
static void hamming4_avx512(const uint64_t* x0, const uint64_t* x1,
                            const uint64_t* x2, const uint64_t* x3,
                            const uint64_t* wm, int words, int* out4) {
    const int w_vec = (words / 8) * 8;
    __m512i a0 = _mm512_setzero_si512(), a1 = _mm512_setzero_si512();
    __m512i a2 = _mm512_setzero_si512(), a3 = _mm512_setzero_si512();
    int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
    int w;
    for (w = 0; w < w_vec; w += 8) {
        const __m512i wv = _mm512_loadu_si512((const void*)(wm + w));
        a0 = _mm512_add_epi64(a0, _mm512_popcnt_epi64(_mm512_xor_si512(
            _mm512_loadu_si512((const void*)(x0 + w)), wv)));
        a1 = _mm512_add_epi64(a1, _mm512_popcnt_epi64(_mm512_xor_si512(
            _mm512_loadu_si512((const void*)(x1 + w)), wv)));
        a2 = _mm512_add_epi64(a2, _mm512_popcnt_epi64(_mm512_xor_si512(
            _mm512_loadu_si512((const void*)(x2 + w)), wv)));
        a3 = _mm512_add_epi64(a3, _mm512_popcnt_epi64(_mm512_xor_si512(
            _mm512_loadu_si512((const void*)(x3 + w)), wv)));
    }
    for (; w < words; ++w) {
        const uint64_t v = wm[w];
        d0 += pop64(x0[w] ^ v);
        d1 += pop64(x1[w] ^ v);
        d2 += pop64(x2[w] ^ v);
        d3 += pop64(x3[w] ^ v);
    }
    out4[0] = d0 + (int)_mm512_reduce_add_epi64(a0);
    out4[1] = d1 + (int)_mm512_reduce_add_epi64(a1);
    out4[2] = d2 + (int)_mm512_reduce_add_epi64(a2);
    out4[3] = d3 + (int)_mm512_reduce_add_epi64(a3);
}
#endif /* BNN_HAVE_AVX512 */

/* ---- ARM64 NEON --------------------------------------------------- */
#if defined(BNN_HAVE_NEON)
/* vcntq_u8 counts bits per byte; vpadalq widens so the uint16 accumulator
 * cannot overflow (16 per step, 65535 headroom => thousands of words). */
static inline int bnn_hsum_u16(uint16x8_t acc) {
    return (int)vaddvq_u32(vpaddlq_u16(acc));
}

static int hamming1_neon(const uint64_t* x, const uint64_t* wm, int words) {
    const int w_vec = (words / 2) * 2;
    uint16x8_t acc = vdupq_n_u16(0);
    int dist = 0;
    int w;
    for (w = 0; w < w_vec; w += 2) {
        const uint8x16_t d = veorq_u8(vld1q_u8((const uint8_t*)(x + w)),
                                      vld1q_u8((const uint8_t*)(wm + w)));
        acc = vpadalq_u8(acc, vcntq_u8(d));
    }
    for (; w < words; ++w) {
        dist += pop64(x[w] ^ wm[w]);
    }
    return dist + bnn_hsum_u16(acc);
}

static void hamming4_neon(const uint64_t* x0, const uint64_t* x1,
                          const uint64_t* x2, const uint64_t* x3,
                          const uint64_t* wm, int words, int* out4) {
    const int w_vec = (words / 2) * 2;
    uint16x8_t a0 = vdupq_n_u16(0), a1 = vdupq_n_u16(0);
    uint16x8_t a2 = vdupq_n_u16(0), a3 = vdupq_n_u16(0);
    int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
    int w;
    for (w = 0; w < w_vec; w += 2) {
        const uint8x16_t wv = vld1q_u8((const uint8_t*)(wm + w));
        a0 = vpadalq_u8(a0, vcntq_u8(veorq_u8(vld1q_u8((const uint8_t*)(x0 + w)), wv)));
        a1 = vpadalq_u8(a1, vcntq_u8(veorq_u8(vld1q_u8((const uint8_t*)(x1 + w)), wv)));
        a2 = vpadalq_u8(a2, vcntq_u8(veorq_u8(vld1q_u8((const uint8_t*)(x2 + w)), wv)));
        a3 = vpadalq_u8(a3, vcntq_u8(veorq_u8(vld1q_u8((const uint8_t*)(x3 + w)), wv)));
    }
    for (; w < words; ++w) {
        const uint64_t v = wm[w];
        d0 += pop64(x0[w] ^ v);
        d1 += pop64(x1[w] ^ v);
        d2 += pop64(x2[w] ^ v);
        d3 += pop64(x3[w] ^ v);
    }
    out4[0] = d0 + bnn_hsum_u16(a0);
    out4[1] = d1 + bnn_hsum_u16(a1);
    out4[2] = d2 + bnn_hsum_u16(a2);
    out4[3] = d3 + bnn_hsum_u16(a3);
}
#endif /* BNN_HAVE_NEON */

/* ---- dispatch table ---------------------------------------------- */
static void bnn_select(hamming4_fn* h4, hamming1_fn* h1) {
    switch (bnn_kernel()) {
#if defined(BNN_HAVE_AVX512)
        case BNN_KERNEL_AVX512: *h4 = hamming4_avx512; *h1 = hamming1_avx512; return;
#endif
#if defined(BNN_HAVE_AVX2)
        case BNN_KERNEL_AVX2:   *h4 = hamming4_avx2;   *h1 = hamming1_avx2;   return;
#endif
#if defined(BNN_HAVE_NEON)
        case BNN_KERNEL_NEON:   *h4 = hamming4_neon;   *h1 = hamming1_neon;   return;
#endif
        default: break;
    }
    *h4 = hamming4_scalar;
    *h1 = hamming1_scalar;
}

/* ------------------------------------------------------------------ */
/* Binary GEMM                                                         */
/*                                                                     */
/* X: (B, words)  W: (M, words)  Y: (B, M), all row-major.             */
/* One OpenMP team for the whole call; weight rows streamed ceil(B/4)   */
/* times instead of B times.                                           */
/* ------------------------------------------------------------------ */
/*
 * Core driver. `alpha` / `bias` are per-output-column vectors of length M, or
 * NULL. Folding them in here is what keeps a wrapped Linear fast: once the
 * GEMM got vectorised, a separate NumPy `y *= alpha; y += bias` pass cost as
 * much as the GEMM itself, purely in memory traffic over the (B, M) output.
 * Fused, the scale happens while the value is still in a register.
 */
static void binary_gemm_impl(
    const uint64_t* X,
    const uint64_t* W,
    float* Y,
    const float* alpha,
    const float* bias,
    int B, int M, int words, int n
) {
    const int nth = effective_threads();
    hamming4_fn h4;
    hamming1_fn h1;

    if (B <= 0 || M <= 0 || words <= 0) {
        return;
    }
    bnn_select(&h4, &h1);

#ifdef _OPENMP
#pragma omp parallel num_threads(nth) if(nth > 1 && (double)B * M * words >= 8192.0)
#endif
    {
        const int b_full = (B / BNN_BR) * BNN_BR;
        int bb, m;

        for (bb = 0; bb < b_full; bb += BNN_BR) {
            const uint64_t* x0 = X + (size_t)(bb + 0) * words;
            const uint64_t* x1 = X + (size_t)(bb + 1) * words;
            const uint64_t* x2 = X + (size_t)(bb + 2) * words;
            const uint64_t* x3 = X + (size_t)(bb + 3) * words;
            float* y0 = Y + (size_t)(bb + 0) * M;
            float* y1 = Y + (size_t)(bb + 1) * M;
            float* y2 = Y + (size_t)(bb + 2) * M;
            float* y3 = Y + (size_t)(bb + 3) * M;
            /* nowait is safe: each bb block writes a disjoint slice of Y and
             * only reads X / W, so no thread can observe a partial result. */
#ifdef _OPENMP
#pragma omp for schedule(static) nowait
#endif
            for (m = 0; m < M; ++m) {
                int d[BNN_BR];
                float a, c;
                h4(x0, x1, x2, x3, W + (size_t)m * words, words, d);
                a = alpha ? alpha[m] : 1.0f;
                c = bias ? bias[m] : 0.0f;
                y0[m] = a * (float)(n - 2 * d[0]) + c;
                y1[m] = a * (float)(n - 2 * d[1]) + c;
                y2[m] = a * (float)(n - 2 * d[2]) + c;
                y3[m] = a * (float)(n - 2 * d[3]) + c;
            }
        }

        for (bb = b_full; bb < B; ++bb) {
            const uint64_t* xb = X + (size_t)bb * words;
            float* yb = Y + (size_t)bb * M;
#ifdef _OPENMP
#pragma omp for schedule(static) nowait
#endif
            for (m = 0; m < M; ++m) {
                const float a = alpha ? alpha[m] : 1.0f;
                const float c = bias ? bias[m] : 0.0f;
                yb[m] = a * (float)(n - 2 * h1(xb, W + (size_t)m * words, words)) + c;
            }
        }
    }
}

#ifdef _WIN32
__declspec(dllexport)
#endif
void binary_gemm_u64(
    const uint64_t* X,
    const uint64_t* W,
    float* Y,
    int B, int M, int words, int n
) {
    binary_gemm_impl(X, W, Y, NULL, NULL, B, M, words, n);
}

/* Y = alpha * (n - 2*hamming) + bias, fused. alpha / bias may be NULL. */
#ifdef _WIN32
__declspec(dllexport)
#endif
void binary_gemm_u64_scaled(
    const uint64_t* X,
    const uint64_t* W,
    float* Y,
    const float* alpha,
    const float* bias,
    int B, int M, int words, int n
) {
    binary_gemm_impl(X, W, Y, alpha, bias, B, M, words, n);
}

/* ------------------------------------------------------------------ */
/* Ternary bitplane GEMM                                               */
/*                                                                     */
/* ±1 activations X, ternary weights W in {-1,0,+1} as bitplanes        */
/* Wp / Wn (1 where W==+1 / W==-1). X encoding matches the binary path. */
/*                                                                     */
/*   y = scale * ( |Wp| - 2 pop(X&Wp) - |Wn| + 2 pop(X&Wn) )            */
/*                                                                     */
/* pop_p / pop_n hold precomputed |Wp| / |Wn| per row (length M) when   */
/* non-NULL; otherwise they are computed on the fly.                    */
/* ------------------------------------------------------------------ */
static inline int popcount_and(const uint64_t* a, const uint64_t* b, int words) {
    int acc = 0;
    int w;
    for (w = 0; w + 3 < words; w += 4) {
        acc += pop64(a[w] & b[w]);
        acc += pop64(a[w + 1] & b[w + 1]);
        acc += pop64(a[w + 2] & b[w + 2]);
        acc += pop64(a[w + 3] & b[w + 3]);
    }
    for (; w < words; ++w) {
        acc += pop64(a[w] & b[w]);
    }
    return acc;
}

#if defined(BNN_HAVE_AVX512)
BNN_TARGET("avx512f,avx512vpopcntdq")
static int popcount_and_avx512(const uint64_t* a, const uint64_t* b, int words) {
    const int w_vec = (words / 8) * 8;
    __m512i acc = _mm512_setzero_si512();
    int rest = 0;
    int w;
    for (w = 0; w < w_vec; w += 8) {
        const __m512i v = _mm512_and_si512(
            _mm512_loadu_si512((const void*)(a + w)),
            _mm512_loadu_si512((const void*)(b + w)));
        acc = _mm512_add_epi64(acc, _mm512_popcnt_epi64(v));
    }
    for (; w < words; ++w) {
        rest += pop64(a[w] & b[w]);
    }
    return rest + (int)_mm512_reduce_add_epi64(acc);
}
#endif

#if defined(BNN_HAVE_AVX2)
BNN_TARGET("avx2")
static int popcount_and_avx2(const uint64_t* a, const uint64_t* b, int words) {
    const int w_vec = (words / 4) * 4;
    __m256i acc = _mm256_setzero_si256();
    int rest = 0;
    int w;
    for (w = 0; w < w_vec; w += 4) {
        const __m256i v = _mm256_and_si256(
            _mm256_loadu_si256((const __m256i*)(a + w)),
            _mm256_loadu_si256((const __m256i*)(b + w)));
        acc = _mm256_add_epi64(acc, bnn_popcnt_epi64_avx2(v));
    }
    for (; w < words; ++w) {
        rest += pop64(a[w] & b[w]);
    }
    return rest + bnn_hsum_epi64_avx2(acc);
}
#endif

#if defined(BNN_HAVE_NEON)
static int popcount_and_neon(const uint64_t* a, const uint64_t* b, int words) {
    const int w_vec = (words / 2) * 2;
    uint16x8_t acc = vdupq_n_u16(0);
    int rest = 0;
    int w;
    for (w = 0; w < w_vec; w += 2) {
        const uint8x16_t v = vandq_u8(vld1q_u8((const uint8_t*)(a + w)),
                                      vld1q_u8((const uint8_t*)(b + w)));
        acc = vpadalq_u8(acc, vcntq_u8(v));
    }
    for (; w < words; ++w) {
        rest += pop64(a[w] & b[w]);
    }
    return rest + bnn_hsum_u16(acc);
}
#endif

typedef int (*popand_fn)(const uint64_t* a, const uint64_t* b, int words);

static popand_fn bnn_select_popand(void) {
    switch (bnn_kernel()) {
#if defined(BNN_HAVE_AVX512)
        case BNN_KERNEL_AVX512: return popcount_and_avx512;
#endif
#if defined(BNN_HAVE_AVX2)
        case BNN_KERNEL_AVX2:   return popcount_and_avx2;
#endif
#if defined(BNN_HAVE_NEON)
        case BNN_KERNEL_NEON:   return popcount_and_neon;
#endif
        default: return popcount_and;
    }
}

static inline int popcount_row(const uint64_t* row, int words) {
    int acc = 0;
    int w;
    for (w = 0; w < words; ++w) {
        acc += pop64(row[w]);
    }
    return acc;
}

#ifdef _WIN32
__declspec(dllexport)
#endif
void ternary_gemm_u64(
    const uint64_t* X,
    const uint64_t* Wp,
    const uint64_t* Wn,
    const int* pop_p,
    const int* pop_n,
    float* Y,
    float scale,
    int B, int M, int words
) {
    const int nth = effective_threads();
    const popand_fn pand = bnn_select_popand();
    int* local_pop_p = NULL;
    int* local_pop_n = NULL;
    const int* use_p = pop_p;
    const int* use_n = pop_n;
    int m;

    if (B <= 0 || M <= 0 || words <= 0) {
        return;
    }

    if (use_p == NULL || use_n == NULL) {
        local_pop_p = (int*)malloc((size_t)M * sizeof(int));
        local_pop_n = (int*)malloc((size_t)M * sizeof(int));
        if (!local_pop_p || !local_pop_n) {
            free(local_pop_p);
            free(local_pop_n);
            return;
        }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(nth) if(M >= 32 && nth > 1)
#endif
        for (m = 0; m < M; ++m) {
            local_pop_p[m] = popcount_row(Wp + (size_t)m * words, words);
            local_pop_n[m] = popcount_row(Wn + (size_t)m * words, words);
        }
        use_p = local_pop_p;
        use_n = local_pop_n;
    }

#ifdef _OPENMP
#pragma omp parallel num_threads(nth) if(nth > 1 && (double)B * M * words >= 8192.0)
#endif
    {
        int b, mm;
        for (b = 0; b < B; ++b) {
            const uint64_t* xb = X + (size_t)b * words;
            float* yb = Y + (size_t)b * M;
#ifdef _OPENMP
#pragma omp for schedule(static) nowait
#endif
            for (mm = 0; mm < M; ++mm) {
                const uint64_t* wp = Wp + (size_t)mm * words;
                const uint64_t* wn = Wn + (size_t)mm * words;
                const int and_p = pand(xb, wp, words);
                const int and_n = pand(xb, wn, words);
                const float dot =
                    (float)(use_p[mm] - 2 * and_p - use_n[mm] + 2 * and_n);
                yb[mm] = scale * dot;
            }
        }
    }

    free(local_pop_p);
    free(local_pop_n);
}
