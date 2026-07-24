/*
 * Packed binary XNOR+popcount GEMM + ternary bitplane GEMM.
 * Encoding (binary): bit 0 => +1, bit 1 => -1
 * Dot: <a,b> = N - 2 * popcount(a XOR b)
 *
 * OpenMP: parallelize over output rows (M). Thread count via
 * binary_gemm_set_num_threads / BNN_NUM_THREADS (Python).
 * MSVC: /openmp + __popcnt64. GCC/Clang: -fopenmp + __builtin_popcountll.
 *
 * Note: MSVC OpenMP 2.0 requires the parallel for index declared outside
 * the for-init (no C99 for(int i=...)).
 */
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>

#ifdef _MSC_VER
#include <intrin.h>
static inline int pop64(uint64_t x) { return (int)__popcnt64(x); }
#else
static inline int pop64(uint64_t x) { return __builtin_popcountll(x); }
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

/* 0 => library / OpenMP default */
static int g_num_threads = 0;

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

/* Accumulate Hamming distance for one (x_row, w_row) pair. */
static inline int hamming_words(const uint64_t* xb, const uint64_t* wm, int words) {
    int dist = 0;
    int w;
    /* 8-wide unroll: keeps __popcnt64 fed without depending on AVX2 popcnt. */
    for (w = 0; w + 7 < words; w += 8) {
        dist += pop64(xb[w] ^ wm[w]);
        dist += pop64(xb[w + 1] ^ wm[w + 1]);
        dist += pop64(xb[w + 2] ^ wm[w + 2]);
        dist += pop64(xb[w + 3] ^ wm[w + 3]);
        dist += pop64(xb[w + 4] ^ wm[w + 4]);
        dist += pop64(xb[w + 5] ^ wm[w + 5]);
        dist += pop64(xb[w + 6] ^ wm[w + 6]);
        dist += pop64(xb[w + 7] ^ wm[w + 7]);
    }
    for (; w + 3 < words; w += 4) {
        dist += pop64(xb[w] ^ wm[w]);
        dist += pop64(xb[w + 1] ^ wm[w + 1]);
        dist += pop64(xb[w + 2] ^ wm[w + 2]);
        dist += pop64(xb[w + 3] ^ wm[w + 3]);
    }
    for (; w < words; ++w) {
        dist += pop64(xb[w] ^ wm[w]);
    }
    return dist;
}

static inline int popcount_and(const uint64_t* a, const uint64_t* b, int words) {
    int acc = 0;
    int w;
    for (w = 0; w + 7 < words; w += 8) {
        acc += pop64(a[w] & b[w]);
        acc += pop64(a[w + 1] & b[w + 1]);
        acc += pop64(a[w + 2] & b[w + 2]);
        acc += pop64(a[w + 3] & b[w + 3]);
        acc += pop64(a[w + 4] & b[w + 4]);
        acc += pop64(a[w + 5] & b[w + 5]);
        acc += pop64(a[w + 6] & b[w + 6]);
        acc += pop64(a[w + 7] & b[w + 7]);
    }
    for (; w < words; ++w) {
        acc += pop64(a[w] & b[w]);
    }
    return acc;
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
void binary_gemm_u64(
    const uint64_t* X,
    const uint64_t* W,
    float* Y,
    int B, int M, int words, int n
) {
    const int nth = effective_threads();
    int b;
    for (b = 0; b < B; ++b) {
        const uint64_t* xb = X + (size_t)b * words;
        float* yb = Y + (size_t)b * M;
        int m;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(nth) if(M >= 32 && nth > 1)
#endif
        for (m = 0; m < M; ++m) {
            const uint64_t* wm = W + (size_t)m * words;
            int dist = hamming_words(xb, wm, words);
            yb[m] = (float)(n - 2 * dist);
        }
    }
}

/*
 * Ternary bitplane GEMM for ±1 activations X and ternary weights W in {-1,0,+1}.
 * Wp / Wn: packed bitplanes (1 where W==+1 / W==-1). Encoding of X matches binary.
 *
 * y = scale * ( |Wp| - 2 pop(X&Wp) - |Wn| + 2 pop(X&Wn) )
 *
 * If pop_p / pop_n are non-NULL they hold precomputed |Wp|/|Wn| per row (length M);
 * otherwise they are computed on the fly.
 */
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
    int* local_pop_p = NULL;
    int* local_pop_n = NULL;
    const int* use_p = pop_p;
    const int* use_n = pop_n;
    int b;
    int m;

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

    for (b = 0; b < B; ++b) {
        const uint64_t* xb = X + (size_t)b * words;
        float* yb = Y + (size_t)b * M;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(nth) if(M >= 32 && nth > 1)
#endif
        for (m = 0; m < M; ++m) {
            const uint64_t* wp = Wp + (size_t)m * words;
            const uint64_t* wn = Wn + (size_t)m * words;
            int and_p = popcount_and(xb, wp, words);
            int and_n = popcount_and(xb, wn, words);
            float dot = (float)(use_p[m] - 2 * and_p - use_n[m] + 2 * and_n);
            yb[m] = scale * dot;
        }
    }

    free(local_pop_p);
    free(local_pop_n);
}
