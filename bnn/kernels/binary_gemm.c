#include <stdint.h>
#include <stddef.h>
#ifdef _MSC_VER
#include <intrin.h>
static inline int pop64(uint64_t x) { return (int)__popcnt64(x); }
#else
static inline int pop64(uint64_t x) { return __builtin_popcountll(x); }
#endif

#ifdef _WIN32
__declspec(dllexport)
#endif
void binary_gemm_u64(
    const uint64_t* X,
    const uint64_t* W,
    float* Y,
    int B, int M, int words, int n
) {
    for (int b = 0; b < B; ++b) {
        const uint64_t* xb = X + (size_t)b * words;
        float* yb = Y + (size_t)b * M;
        for (int m = 0; m < M; ++m) {
            const uint64_t* wm = W + (size_t)m * words;
            int dist = 0;
            int w = 0;
            for (; w + 3 < words; w += 4) {
                dist += pop64(xb[w] ^ wm[w]);
                dist += pop64(xb[w + 1] ^ wm[w + 1]);
                dist += pop64(xb[w + 2] ^ wm[w + 2]);
                dist += pop64(xb[w + 3] ^ wm[w + 3]);
            }
            for (; w < words; ++w) {
                dist += pop64(xb[w] ^ wm[w]);
            }
            yb[m] = (float)(n - 2 * dist);
        }
    }
}
