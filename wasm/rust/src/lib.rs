//! Pedagogy WASM binary GEMM with optional SIMD128 popcount (W2.T06).
//!
//! Math matches `bnn/kernels/binary_gemm.c` (read-only): bit0=+1, bit1=-1,
//! `<a,b> = N - 2 * popcount(a XOR b)`.
//!
//! Non-goal: replace host AVX/NEON/OpenMP kernels or claim wall-clock wins.

#![cfg(target_arch = "wasm32")]
#![allow(clippy::missing_safety_doc)]

use core::arch::wasm32::*;

const KERNEL_SCALAR: i32 = 0;
const KERNEL_SIMD128: i32 = 1;

static mut G_KERNEL: i32 = -1;

#[inline]
fn pop64(x: u64) -> i32 {
    x.count_ones() as i32
}

#[inline]
fn detect() -> i32 {
    KERNEL_SIMD128
}

#[no_mangle]
pub extern "C" fn binary_gemm_wasm_kernel_id() -> i32 {
    unsafe {
        if G_KERNEL < 0 {
            G_KERNEL = detect();
        }
        G_KERNEL
    }
}

#[no_mangle]
pub extern "C" fn binary_gemm_wasm_set_kernel(id: i32) -> i32 {
    unsafe {
        G_KERNEL = if id < 0 {
            -1
        } else if id == KERNEL_SCALAR || id == KERNEL_SIMD128 {
            id
        } else {
            KERNEL_SCALAR
        };
    }
    binary_gemm_wasm_kernel_id()
}

fn hamming1_scalar(x: &[u64], wm: &[u64]) -> i32 {
    let mut dist = 0i32;
    let words = x.len().min(wm.len());
    let mut w = 0;
    while w + 3 < words {
        dist += pop64(x[w] ^ wm[w]);
        dist += pop64(x[w + 1] ^ wm[w + 1]);
        dist += pop64(x[w + 2] ^ wm[w + 2]);
        dist += pop64(x[w + 3] ^ wm[w + 3]);
        w += 4;
    }
    while w < words {
        dist += pop64(x[w] ^ wm[w]);
        w += 1;
    }
    dist
}

#[inline]
unsafe fn popcnt_v128_bytes(v: v128) -> i32 {
    let c = u8x16_popcnt(v);
    let s16 = i16x8_extadd_pairwise_i8x16(c);
    let s32 = i32x4_extadd_pairwise_i16x8(s16);
    i32x4_extract_lane::<0>(s32)
        + i32x4_extract_lane::<1>(s32)
        + i32x4_extract_lane::<2>(s32)
        + i32x4_extract_lane::<3>(s32)
}

fn hamming1_simd128(x: &[u64], wm: &[u64]) -> i32 {
    let words = x.len().min(wm.len());
    let w_vec = (words / 2) * 2;
    let mut dist = 0i32;
    let mut w = 0;
    while w < w_vec {
        unsafe {
            let xv = v128_load(x.as_ptr().add(w) as *const v128);
            let wv = v128_load(wm.as_ptr().add(w) as *const v128);
            dist += popcnt_v128_bytes(v128_xor(xv, wv));
        }
        w += 2;
    }
    while w < words {
        dist += pop64(x[w] ^ wm[w]);
        w += 1;
    }
    dist
}

fn hamming1(x: &[u64], wm: &[u64]) -> i32 {
    if binary_gemm_wasm_kernel_id() == KERNEL_SIMD128 {
        hamming1_simd128(x, wm)
    } else {
        hamming1_scalar(x, wm)
    }
}

/// Row-major X (B, words), W (M, words), Y (B, M).
#[no_mangle]
pub unsafe extern "C" fn binary_gemm_wasm_u64(
    x: *const u64,
    w: *const u64,
    y: *mut f32,
    b: i32,
    m: i32,
    words: i32,
    n: i32,
) {
    if b <= 0 || m <= 0 || words <= 0 || x.is_null() || w.is_null() || y.is_null() {
        return;
    }
    let b = b as usize;
    let m = m as usize;
    let words = words as usize;
    let n = n as i32;
    for bi in 0..b {
        let xb = core::slice::from_raw_parts(x.add(bi * words), words);
        for mi in 0..m {
            let wm = core::slice::from_raw_parts(w.add(mi * words), words);
            let dist = hamming1(xb, wm);
            *y.add(bi * m + mi) = (n - 2 * dist) as f32;
        }
    }
}
