/*
 * Pedagogy WASM binary GEMM (W2.T06 / M1).
 *
 * Same math as bnn/kernels/binary_gemm.c (read-only reference):
 *   Encoding: bit 0 => +1, bit 1 => -1
 *   Dot: <a,b> = N - 2 * popcount(a XOR b)
 *
 * Paths:
 *   - scalar: __builtin_popcountll (or portable SWAR)
 *   - wasm_simd128: i8x16.popcnt over XOR words (when built with -msimd128)
 *
 * NON-GOALS (honest):
 *   - Not a drop-in replacement for native CPU kernels (AVX-512 / AVX2 / NEON / OpenMP)
 *   - No latency or "32x" claims vs host kernels — browser/edge pedagogy only
 *   - No OpenMP; single-threaded; no fused alpha/bias (keep the demo tiny)
 *
 * Build (optional): see wasm/build.py / wasm/README.md
 */
#include <stdint.h>
#include <stddef.h>

#if defined(__wasm_simd128__)
#  include <wasm_simd128.h>
#  define BNN_WASM_SIMD 1
#endif

#if defined(__GNUC__) || defined(__clang__)
static inline int pop64(uint64_t x) { return __builtin_popcountll(x); }
#else
/* Portable SWAR popcount for odd toolchains. */
static inline int pop64(uint64_t x) {
    x -= (x >> 1) & 0x5555555555555555ULL;
    x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
    x = (x + (x >> 4)) & 0x0f0f0f0f0f0f0f0fULL;
    return (int)((x * 0x0101010101010101ULL) >> 56);
}
#endif

enum {
    BNN_WASM_KERNEL_SCALAR = 0,
    BNN_WASM_KERNEL_SIMD128 = 1
};

static int g_kernel = -1;

static int bnn_wasm_detect(void) {
#if defined(BNN_WASM_SIMD)
    return BNN_WASM_KERNEL_SIMD128;
#else
    return BNN_WASM_KERNEL_SCALAR;
#endif
}

int binary_gemm_wasm_kernel_id(void) {
    if (g_kernel < 0) {
        g_kernel = bnn_wasm_detect();
    }
    return g_kernel;
}

int binary_gemm_wasm_set_kernel(int id) {
    if (id < 0) {
        g_kernel = -1;
    } else if (id == BNN_WASM_KERNEL_SCALAR) {
        g_kernel = BNN_WASM_KERNEL_SCALAR;
#if defined(BNN_WASM_SIMD)
    } else if (id == BNN_WASM_KERNEL_SIMD128) {
        g_kernel = BNN_WASM_KERNEL_SIMD128;
#endif
    } else {
        g_kernel = BNN_WASM_KERNEL_SCALAR;
    }
    return binary_gemm_wasm_kernel_id();
}

/* ---- scalar Hamming ------------------------------------------------ */
static int hamming1_scalar(const uint64_t* x, const uint64_t* wm, int words) {
    int dist = 0;
    int w;
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

#if defined(BNN_WASM_SIMD)
/*
 * Two uint64 words = 16 bytes = one v128. XOR then i8x16.popcnt; sum bytes.
 * Remainder words fall back to scalar pop64.
 */
static inline int popcnt_v128_bytes(v128_t v) {
    v128_t c = wasm_i8x16_popcnt(v);
    v128_t s16 = wasm_i16x8_extadd_pairwise_i8x16(c);
    v128_t s32 = wasm_i32x4_extadd_pairwise_i16x8(s16);
    return (int)wasm_i32x4_extract_lane(s32, 0)
         + (int)wasm_i32x4_extract_lane(s32, 1)
         + (int)wasm_i32x4_extract_lane(s32, 2)
         + (int)wasm_i32x4_extract_lane(s32, 3);
}

static int hamming1_simd128(const uint64_t* x, const uint64_t* wm, int words) {
    const int w_vec = (words / 2) * 2;
    int dist = 0;
    int w;
    for (w = 0; w < w_vec; w += 2) {
        v128_t xv = wasm_v128_load(x + w);
        v128_t wv = wasm_v128_load(wm + w);
        dist += popcnt_v128_bytes(wasm_v128_xor(xv, wv));
    }
    for (; w < words; ++w) {
        dist += pop64(x[w] ^ wm[w]);
    }
    return dist;
}
#endif

static int hamming1(const uint64_t* x, const uint64_t* wm, int words) {
#if defined(BNN_WASM_SIMD)
    if (binary_gemm_wasm_kernel_id() == BNN_WASM_KERNEL_SIMD128) {
        return hamming1_simd128(x, wm, words);
    }
#endif
    return hamming1_scalar(x, wm, words);
}

/*
 * X: (B, words)  W: (M, words)  Y: (B, M)  — all row-major.
 * n: original feature width (for n - 2*hamming).
 */
void binary_gemm_wasm_u64(
    const uint64_t* X,
    const uint64_t* W,
    float* Y,
    int B, int M, int words, int n
) {
    int b, m;
    if (B <= 0 || M <= 0 || words <= 0) {
        return;
    }
    for (b = 0; b < B; ++b) {
        const uint64_t* xb = X + (size_t)b * (size_t)words;
        float* yb = Y + (size_t)b * (size_t)M;
        for (m = 0; m < M; ++m) {
            yb[m] = (float)(n - 2 * hamming1(xb, W + (size_t)m * (size_t)words, words));
        }
    }
}
