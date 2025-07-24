use ndarray::{Array2, ArrayView2, ArrayViewMut2};
use ndarray::s;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*; 
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// --- The Public Dispatcher Function ---
pub fn matmul(a: &ArrayView2<f32>, b: &ArrayView2<f32>, c: &mut Array2<f32>) {
    // For simplicity, this example assumes matrix dimensions are divisible by block sizes.
    
    // Use an unsafe block because the underlying kernels are unsafe.
    unsafe {
        #[cfg(all(target_arch = "x86_64", target_feature = "fma"))]
        {
            // If we are on x86_64 with FMA support, call the AVX2 kernel.
            packed_matmul_x86_fma(a, b, c);
        }
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            // If we are on aarch64 with NEON support, call the NEON kernel.
            packed_matmul_aarch64_neon(a, b, c);
        }
        #[cfg(not(any(
            all(target_arch = "x86_64", target_feature = "fma"),
            all(target_arch = "aarch64", target_feature = "neon")
        )))]
        {
            // For any other architecture, fall back to a safe, scalar version.
            packed_matmul_scalar(a, b, c);
        }
    }
}

// --- x86_64 AVX2/FMA Implementation ---
#[cfg(all(target_arch = "x86_64", target_feature = "fma"))]
#[target_feature(enable = "fma")]
unsafe fn packed_matmul_x86_fma(a: &ArrayView2<f32>, b: &ArrayView2<f32>, c: &mut Array2<f32>) {
    const TILE_SIZE_M: usize = 4;
    const TILE_SIZE_N: usize = 8;
    const BLOCK_SIZE_M: usize = 32;
    const BLOCK_SIZE_N: usize = 32;
    const BLOCK_SIZE_K: usize = 32;

    let (m, k) = a.dim();
    let (_k, n) = b.dim();

    let mut pack_a = vec![0.0; BLOCK_SIZE_M * BLOCK_SIZE_K];
    let mut pack_b = vec![0.0; BLOCK_SIZE_K * BLOCK_SIZE_N];

    for l in (0..k).step_by(BLOCK_SIZE_K) {
        let k_block_size = BLOCK_SIZE_K;
        for j in (0..n).step_by(BLOCK_SIZE_N) {
            let n_block_size = BLOCK_SIZE_N;
            // Pack B
            let mut b_idx = 0;
            for depth in l..(l + k_block_size) {
                for col in j..(j + n_block_size) {
                    pack_b[b_idx] = b[[depth, col]];
                    b_idx += 1;
                }
            }
            for i in (0..m).step_by(BLOCK_SIZE_M) {
                let m_block_size = BLOCK_SIZE_M;
                // Pack A
                let mut a_idx = 0;
                for depth in l..(l + k_block_size) {
                    for row in i..(i + m_block_size) {
                        pack_a[a_idx] = a[[row, depth]];
                        a_idx += 1;
                    }
                }
                // Inner kernel tiling
                for j_inner in (0..n_block_size).step_by(TILE_SIZE_N) {
                    for i_inner in (0..m_block_size).step_by(TILE_SIZE_M) {
                        let a_ptr = pack_a.as_ptr().add(i_inner * k_block_size);
                        let b_ptr = pack_b.as_ptr().add(j_inner);
                        let mut c_slice = c.slice_mut(s![i + i_inner.., j + j_inner..]);
                        micro_kernel_4x8_simd(k_block_size, a_ptr, b_ptr, &mut c_slice, n);
                    }
                }
            }
        }
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "fma"))]
unsafe fn micro_kernel_4x8_simd(
    k_size: usize, pack_a: *const f32, pack_b: *const f32,
    c: &mut ArrayViewMut2<f32>, c_stride: usize,
) {
    const TILE_SIZE_M: usize = 4;
    const TILE_SIZE_N: usize = 8;
    let mut c0_0_7 = _mm256_setzero_ps();
    let mut c1_0_7 = _mm256_setzero_ps();
    let mut c2_0_7 = _mm256_setzero_ps();
    let mut c3_0_7 = _mm256_setzero_ps();

    for depth in 0..k_size {
        let b_vals = _mm256_loadu_ps(pack_b.add(depth * TILE_SIZE_N));
        let a_val0 = _mm256_set1_ps(*pack_a.add(depth * TILE_SIZE_M + 0));
        let a_val1 = _mm256_set1_ps(*pack_a.add(depth * TILE_SIZE_M + 1));
        let a_val2 = _mm256_set1_ps(*pack_a.add(depth * TILE_SIZE_M + 2));
        let a_val3 = _mm256_set1_ps(*pack_a.add(depth * TILE_SIZE_M + 3));

        c0_0_7 = _mm256_fmadd_ps(a_val0, b_vals, c0_0_7);
        c1_0_7 = _mm256_fmadd_ps(a_val1, b_vals, c1_0_7);
        c2_0_7 = _mm256_fmadd_ps(a_val2, b_vals, c2_0_7);
        c3_0_7 = _mm256_fmadd_ps(a_val3, b_vals, c3_0_7);
    }
    
    let c_ptr = c.as_mut_ptr();
    _mm256_storeu_ps(c_ptr.add(0), _mm256_add_ps(_mm256_loadu_ps(c_ptr.add(0)), c0_0_7));
    _mm256_storeu_ps(c_ptr.add(c_stride), _mm256_add_ps(_mm256_loadu_ps(c_ptr.add(c_stride)), c1_0_7));
    _mm256_storeu_ps(c_ptr.add(2 * c_stride), _mm256_add_ps(_mm256_loadu_ps(c_ptr.add(2 * c_stride)), c2_0_7));
    _mm256_storeu_ps(c_ptr.add(3 * c_stride), _mm256_add_ps(_mm256_loadu_ps(c_ptr.add(3 * c_stride)), c3_0_7));
}


// --- aarch64 NEON Implementation ---
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[target_feature(enable = "neon")]
unsafe fn packed_matmul_aarch64_neon(a: &ArrayView2<f32>, b: &ArrayView2<f32>, c: &mut Array2<f32>) {
    const TILE_SIZE_M: usize = 4;
    const TILE_SIZE_N: usize = 4;
    const BLOCK_SIZE_M: usize = 32;
    const BLOCK_SIZE_N: usize = 32;
    const BLOCK_SIZE_K: usize = 32;

    let (m, k) = a.dim();
    let (_k, n) = b.dim();

    let mut pack_a = vec![0.0; BLOCK_SIZE_M * BLOCK_SIZE_K];
    let mut pack_b = vec![0.0; BLOCK_SIZE_K * BLOCK_SIZE_N];

    for l in (0..k).step_by(BLOCK_SIZE_K) {
        let k_block_size = BLOCK_SIZE_K;
        for j in (0..n).step_by(BLOCK_SIZE_N) {
            let n_block_size = BLOCK_SIZE_N;
            // Pack B
            let mut b_idx = 0;
            for depth in l..(l + k_block_size) {
                for col in j..(j + n_block_size) {
                    pack_b[b_idx] = b[[depth, col]];
                    b_idx += 1;
                }
            }
            for i in (0..m).step_by(BLOCK_SIZE_M) {
                let m_block_size = BLOCK_SIZE_M;
                // Pack A
                let mut a_idx = 0;
                for depth in l..(l + k_block_size) {
                    for row in i..(i + m_block_size) {
                        pack_a[a_idx] = a[[row, depth]];
                        a_idx += 1;
                    }
                }
                // Inner kernel tiling
                for j_inner in (0..n_block_size).step_by(TILE_SIZE_N) {
                    for i_inner in (0..m_block_size).step_by(TILE_SIZE_M) {
                        let a_ptr = pack_a.as_ptr().add(i_inner * k_block_size);
                        let b_ptr = pack_b.as_ptr().add(j_inner);
                        let mut c_slice = c.slice_mut(s![i + i_inner.., j + j_inner..]);
                        micro_kernel_4x4_neon(k_block_size, a_ptr, b_ptr, &mut c_slice, n);
                    }
                }
            }
        }
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
unsafe fn micro_kernel_4x4_neon(
    k_size: usize, pack_a: *const f32, pack_b: *const f32,
    c: &mut ArrayViewMut2<f32>, c_stride: usize,
) {
    const TILE_SIZE_M: usize = 4;
    const TILE_SIZE_N: usize = 4;
    // Four 128-bit registers to hold the 4x4 C tile
    let mut c0_3_v = vdupq_n_f32(0.0);
    let mut c1_3_v = vdupq_n_f32(0.0);
    let mut c2_3_v = vdupq_n_f32(0.0);
    let mut c3_3_v = vdupq_n_f32(0.0);

    for depth in 0..k_size {
        // Load one row (4 floats) from packed B
        let b_v = vld1q_f32(pack_b.add(depth * TILE_SIZE_N));

        // Load 4 scalars from packed A and broadcast them to vectors
        let a0_v = vdupq_n_f32(*pack_a.add(depth * TILE_SIZE_M + 0));
        let a1_v = vdupq_n_f32(*pack_a.add(depth * TILE_SIZE_M + 1));
        let a2_v = vdupq_n_f32(*pack_a.add(depth * TILE_SIZE_M + 2));
        let a3_v = vdupq_n_f32(*pack_a.add(depth * TILE_SIZE_M + 3));

        // Fused multiply-add: c += a * b
        c0_3_v = vfmaq_f32(c0_3_v, a0_v, b_v);
        c1_3_v = vfmaq_f32(c1_3_v, a1_v, b_v);
        c2_3_v = vfmaq_f32(c2_3_v, a2_v, b_v);
        c3_3_v = vfmaq_f32(c3_3_v, a3_v, b_v);
    }

    let c_ptr = c.as_mut_ptr();
    // Load existing C values, add results, and store back
    vst1q_f32(c_ptr.add(0), vaddq_f32(vld1q_f32(c_ptr.add(0)), c0_3_v));
    vst1q_f32(c_ptr.add(c_stride), vaddq_f32(vld1q_f32(c_ptr.add(c_stride)), c1_3_v));
    vst1q_f32(c_ptr.add(2 * c_stride), vaddq_f32(vld1q_f32(c_ptr.add(2 * c_stride)), c2_3_v));
    vst1q_f32(c_ptr.add(3 * c_stride), vaddq_f32(vld1q_f32(c_ptr.add(3 * c_stride)), c3_3_v));
}


// --- Scalar Fallback Implementation ---
// This runs on any architecture that doesn't have a specialized SIMD kernel.
fn packed_matmul_scalar(a: &ArrayView2<f32>, b: &ArrayView2<f32>, c: &mut Array2<f32>) {
    const BLOCK_SIZE: usize = 32;
    let (m, k) = a.dim();
    let (_k, n) = b.dim();
    
    let mut pack_a = vec![0.0; BLOCK_SIZE * BLOCK_SIZE];
    let mut pack_b = vec![0.0; BLOCK_SIZE * BLOCK_SIZE];

    for l in (0..k).step_by(BLOCK_SIZE) {
        let k_block_end = std::cmp::min(l + BLOCK_SIZE, k);
        let k_block_size = k_block_end - l;
        for j in (0..n).step_by(BLOCK_SIZE) {
            let n_block_end = std::cmp::min(j + BLOCK_SIZE, n);
            let n_block_size = n_block_end - j;
            // Pack B
            let mut b_idx = 0;
            for depth in l..k_block_end {
                for col in j..n_block_end {
                    pack_b[b_idx] = b[[depth, col]];
                    b_idx += 1;
                }
            }
            for i in (0..m).step_by(BLOCK_SIZE) {
                let m_block_end = std::cmp::min(i + BLOCK_SIZE, m);
                let m_block_size = m_block_end - i;
                // Pack A
                let mut a_idx = 0;
                for depth in l..k_block_end {
                    for row in i..m_block_end {
                        pack_a[a_idx] = a[[row, depth]];
                        a_idx += 1;
                    }
                }
                // Scalar micro-kernel
                for row in 0..m_block_size {
                    for col in 0..n_block_size {
                        let mut sum = 0.0;
                        for depth in 0..k_block_size {
                            sum += pack_a[row * k_block_size + depth] * pack_b[depth * n_block_size + col];
                        }
                        c[[i + row, j + col]] += sum;
                    }
                }
            }
        }
    }
}
