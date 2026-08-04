/**
 * Pedagogy packed binary GEMM for Node / browser (W2.T06).
 *
 * Encoding matches bnn/kernels/binary_gemm.c:
 *   bit 0 => +1, bit 1 => -1
 *   <a,b> = N - 2 * popcount(a XOR b)
 *
 * Always provides a scalar JS path. Optionally loads wasm/dist/binary_gemm_wasm.wasm
 * (SIMD128 when the module was built with +simd128).
 */

const POP8 = Uint8Array.from({ length: 256 }, (_, i) => {
  let n = i, c = 0;
  while (n) {
    c += n & 1;
    n >>= 1;
  }
  return c;
});

/** Popcount of a JS BigInt treated as unsigned 64-bit. */
export function pop64(u) {
  let x = BigInt.asUintN(64, BigInt(u));
  let c = 0;
  for (let i = 0; i < 8; i++) {
    c += POP8[Number(x & 0xffn)];
    x >>= 8n;
  }
  return c;
}

/**
 * @param {BigUint64Array} xRow length = words
 * @param {BigUint64Array} wRow length = words
 */
export function hamming1(xRow, wRow) {
  const words = Math.min(xRow.length, wRow.length);
  let dist = 0;
  for (let w = 0; w < words; w++) {
    dist += pop64(xRow[w] ^ wRow[w]);
  }
  return dist;
}

/**
 * Scalar JS GEMM.
 * @param {BigUint64Array} X length B*words
 * @param {BigUint64Array} W length M*words
 * @returns {Float32Array} length B*M
 */
export function binaryGemmScalar(X, W, B, M, words, n) {
  const Y = new Float32Array(B * M);
  for (let b = 0; b < B; b++) {
    const xOff = b * words;
    for (let m = 0; m < M; m++) {
      const wOff = m * words;
      let dist = 0;
      for (let w = 0; w < words; w++) {
        dist += pop64(X[xOff + w] ^ W[wOff + w]);
      }
      Y[b * M + m] = n - 2 * dist;
    }
  }
  return Y;
}

/**
 * Pack ±1 Float32 row-major into BigUint64Array (little-endian bit order).
 * bit set when value <= 0 (matches pack_binary_pm1).
 */
export function packPm1(flat, rows, cols) {
  const words = Math.ceil(cols / 64);
  const out = new BigUint64Array(rows * words);
  for (let r = 0; r < rows; r++) {
    for (let w = 0; w < words; w++) {
      let word = 0n;
      for (let b = 0; b < 64; b++) {
        const j = w * 64 + b;
        if (j >= cols) break;
        const v = flat[r * cols + j];
        if (v <= 0) word |= 1n << BigInt(b);
      }
      out[r * words + w] = word;
    }
  }
  return { packed: out, n: cols, words };
}

let _wasm = null;

/**
 * Try to instantiate a compiled module. Returns null if missing / unsupported.
 * @param {string|URL|Uint8Array} source path or bytes
 */
export async function loadWasm(source) {
  let bytes;
  if (source instanceof Uint8Array) {
    bytes = source;
  } else if (typeof source === "string" || source instanceof URL) {
    if (typeof fetch === "function") {
      const res = await fetch(source);
      if (!res.ok) return null;
      bytes = new Uint8Array(await res.arrayBuffer());
    } else {
      const { readFile } = await import("node:fs/promises");
      bytes = new Uint8Array(await readFile(source));
    }
  } else {
    return null;
  }
  try {
    const { instance } = await WebAssembly.instantiate(bytes, {});
    const exp = instance.exports;
    if (typeof exp.binary_gemm_wasm_u64 !== "function") {
      return null;
    }
    _wasm = instance;
    return instance;
  } catch {
    return null;
  }
}

export function wasmKernelId() {
  if (!_wasm) return -1;
  const f = _wasm.exports.binary_gemm_wasm_kernel_id;
  return typeof f === "function" ? f() : -1;
}

/**
 * Run GEMM via WASM exports using linear memory views.
 */
export function binaryGemmWasm(X, W, B, M, words, n) {
  if (!_wasm) return null;
  const exp = _wasm.exports;
  const memory = exp.memory;
  if (!(memory instanceof WebAssembly.Memory)) {
    return null;
  }
  const base = 64;
  const xBytes = B * words * 8;
  const wBytes = M * words * 8;
  const yBytes = B * M * 4;
  const bytesNeeded = base + xBytes + wBytes + yBytes;
  const needPages = Math.ceil(bytesNeeded / 65536);
  const havePages = memory.buffer.byteLength / 65536;
  if (needPages > havePages) {
    memory.grow(needPages - havePages);
  }
  const xPtr = base;
  const wPtr = xPtr + xBytes;
  const yPtr = wPtr + wBytes;
  const u8 = new Uint8Array(memory.buffer);
  u8.set(new Uint8Array(X.buffer, X.byteOffset, X.byteLength), xPtr);
  u8.set(new Uint8Array(W.buffer, W.byteOffset, W.byteLength), wPtr);
  exp.binary_gemm_wasm_u64(xPtr, wPtr, yPtr, B, M, words, n);
  return Float32Array.from(new Float32Array(memory.buffer, yPtr, B * M));
}

/**
 * Prefer WASM when loaded and usable; else scalar JS.
 */
export function binaryGemm(X, W, B, M, words, n) {
  const viaWasm = binaryGemmWasm(X, W, B, M, words, n);
  if (viaWasm) return { y: viaWasm, backend: "wasm" };
  return { y: binaryGemmScalar(X, W, B, M, words, n), backend: "js-scalar" };
}
