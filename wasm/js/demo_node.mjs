#!/usr/bin/env node
/**
 * Node pedagogy demo for packed binary GEMM (W2.T06).
 *
 * Runs scalar JS (always) and optional wasm/dist/binary_gemm_wasm.wasm.
 * Prints backend + max|err| vs a tiny FP32 reference — never claims host speedups.
 */

import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { access } from "node:fs/promises";
import {
  binaryGemm,
  binaryGemmScalar,
  loadWasm,
  packPm1,
  wasmKernelId,
} from "./binary_gemm.mjs";

const __dir = dirname(fileURLToPath(import.meta.url));
const distWasm = join(__dir, "..", "dist", "binary_gemm_wasm.wasm");

function mulberry32(a) {
  return function () {
    let t = (a += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function pm1Matrix(rows, cols, seed) {
  const rnd = mulberry32(seed);
  const flat = new Float32Array(rows * cols);
  for (let i = 0; i < flat.length; i++) {
    flat[i] = rnd() < 0.5 ? -1 : 1;
  }
  return flat;
}

function fp32Gemm(x, w, B, N, M) {
  const y = new Float32Array(B * M);
  for (let b = 0; b < B; b++) {
    for (let m = 0; m < M; m++) {
      let s = 0;
      for (let n = 0; n < N; n++) {
        s += x[b * N + n] * w[m * N + n];
      }
      y[b * M + m] = s;
    }
  }
  return y;
}

function maxAbsErr(a, b) {
  let m = 0;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    if (d > m) m = d;
  }
  return m;
}

async function main() {
  const B = 4,
    N = 128,
    M = 16;
  const x = pm1Matrix(B, N, 1);
  const w = pm1Matrix(M, N, 2);
  const { packed: xp, words } = packPm1(x, B, N);
  const { packed: wp } = packPm1(w, M, N);
  const yFp = fp32Gemm(x, w, B, N, M);
  const yJs = binaryGemmScalar(xp, wp, B, M, words, N);
  const errJs = maxAbsErr(yJs, yFp);

  let wasmNote = "not loaded";
  try {
    await access(distWasm);
    const inst = await loadWasm(distWasm);
    if (inst) {
      wasmNote = `loaded kernel_id=${wasmKernelId()}`;
    } else {
      wasmNote = "present but instantiate failed (SIMD/host?)";
    }
  } catch {
    wasmNote = "dist/*.wasm absent — run: python wasm/build.py --rust";
  }

  const { y, backend } = binaryGemm(xp, wp, B, M, words, N);
  const err = maxAbsErr(y, yFp);

  console.log("BNN pedagogy WASM / JS binary GEMM demo (W2.T06)");
  console.log(`  shape B=${B} N=${N} M=${M} words=${words}`);
  console.log(`  js-scalar max|err| vs FP32 = ${errJs}`);
  console.log(`  wasm artifact: ${wasmNote}`);
  console.log(`  binaryGemm backend=${backend} max|err| vs FP32 = ${err}`);
  console.log(
    "  note: pedagogy only — not a substitute for native CPU kernels (docs/41)."
  );
  if (errJs !== 0 || err !== 0) {
    console.error("FAIL: expected exact parity with FP32 ±1 GEMM");
    process.exit(1);
  }
  console.log("PASS");
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
