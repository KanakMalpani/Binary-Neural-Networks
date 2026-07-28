# Kernels
Packed XNOR/popcount GEMM with runtime SIMD dispatch. See
[Portable SIMD kernel](../41_PORTABLE_SIMD_KERNEL.md) for the design.

## Runtime dispatch

::: bnn.kernels.packed.kernel_name
::: bnn.kernels.packed.available_kernels
::: bnn.kernels.packed.cpu_features
::: bnn.kernels.packed.set_kernel

## Packing

::: bnn.kernels.packed.pack_binary_pm1
::: bnn.kernels.packed.theoretical_ops

## GEMM

::: bnn.kernels.packed.binary_gemm_packed
::: bnn.kernels.packed.binary_gemm_numpy_prepacked
::: bnn.kernels.packed.binary_gemm_native_prepacked
::: bnn.kernels.packed.binary_gemm_native_scaled
::: bnn.kernels.packed.fp32_gemm

## Threads

::: bnn.kernels.packed.set_num_threads
::: bnn.kernels.packed.get_num_threads
::: bnn.kernels.packed.openmp_enabled

## Building the native library

::: bnn.kernels.compile_native.unix_compile_commands
