#[cfg(target_arch = "x86_64")]
use ndarray::{ArrayView2, ArrayViewMut2};
use ndarray::s;
use core::arch::x86_64::*; 

const TILE_SIZE_M: usize = 4; 
const TILE_SIZE_N: usize = 8; 

#[cfg_attr(all(target_arch = "x86_64", target_feature = "fma"), target_feature(enable = "fma"))]
pub unsafe fn packed_matmul_simd(a: &ArrayView2<f32>, b: &ArrayView2<f32>, c: &mut Array2<f32>){
    const BLOCK_SIZE_M: usize = 32; // Can be tuned
    const BLOCK_SIZE_N: usize = 32;
    const BLOCK_SIZE_K: usize = 32;

    let (m, k) = a.dim();
    let (_k, n) = b.dim();

    assert_eq!(k, _k);
    assert_eq!(c.dim(), (m, n));
    
    let mut pack_a = vec![0.0; BLOCK_SIZE_M * BLOCK_SIZE_K];
    let mut pack_b = vec![0.0; BLOCK_SIZE_K * BLOCK_SIZE_N];

    for l in (0..k).step_by(BLOCK_SIZE_K) {
        let k_block_end = std::cmp::min(l + BLOCK_SIZE_K, k);
        let k_block_size = k_block_end - l;

        for j in (0..n).step_by(BLOCK_SIZE_N) {
            let n_block_end = std::cmp::min(j + BLOCK_SIZE_N, n);
            let n_block_size = n_block_end - j;

            // --- Pack B (Row-Major) ---
            let mut b_idx = 0;
            for depth in l..k_block_end {
                for col in j..n_block_end {
                    pack_b[b_idx] = b[[depth, col]];
                    b_idx += 1;
                }
            }
            
            for i in (0..m).step_by(BLOCK_SIZE_M) {
                let m_block_end = std::cmp::min(i + BLOCK_SIZE_M, m);
                let m_block_size = m_block_end - i;

                // --- Pack A (Column-Major) ---
                let mut a_idx = 0;
                for depth in l..k_block_end {
                    for row in i..m_block_end {
                        pack_a[a_idx] = a[[row, depth]];
                        a_idx += 1;
                    }
                }
                
                // --- Call Micro-Kernel on Packed Blocks ---
                for j_inner in (0..BLOCK_SIZE_N).step_by(TILE_SIZE_N){
                    for i_inner in (0..BLOCK_SIZE_M).step_by(TILE_SIZE_M){
                        // Get pointers into the packed buffers for this specific tile
                        let a_ptr = pack_a.as_ptr().add(i_inner * k_block_size);
                        let b_ptr = pack_b.as_ptr().add(j_inner); // Pointer to start of a row in B
                        let mut c_slice = c.slice_mut(s![i + i_inner.., j + j_inner..]);

                        micro_kernel_4x8_simd(
                            k_block_size,
                            a_ptr, b_ptr,
                            &mut c_slice,
                            n
                        );
                    }
                }
            }
        }
    }
}


#[cfg(all(target_arch = "x86_64", target_feature="fma"))]
unsafe fn micro_kernel_4x8_simd(
    k_size: usize, 
    pack_a: *const f32, 
    pack_b: *const f32, 
    c: &mut ndarray::ArrayViewMut2<f32>,
    c_stride: usize,
) {
    // 8 registers to hold the 4x8 block of C
    let mut c0_0_7 = _mm256_setzero_ps();
    let mut c1_0_7 = _mm256_setzero_ps();
    let mut c2_0_7 = _mm256_setzero_ps();
    let mut c3_0_7 = _mm256_setzero_ps();

    let n_stride_b = TILE_SIZE_N;

    for depth in 0..k_size {
        let b_vals = _mm256_loadu_ps(pack_b.add(depth*n_stride_b));

        let a_val0 = _mm256_set1_ps(*a_ptr.add(depth*TILE_SIZE_M + 0));
        let a_val1 = _mm256_set1_ps(*a_ptr.add(depth*TILE_SIZE_M + 1));
        let a_val2 = _mm256_set1_ps(*a_ptr.add(depth*TILE_SIZE_M + 2));
        let a_val3 = _mm256_set1_ps(*a_ptr.add(depth*TILE_SIZE_M + 3));

        c0_0_7 = _mm256_fmadd_ps(a_val0, b_vals, c0_0_7);
        c1_0_7 = _mm256_fmadd_ps(a_val1, b_vals, c1_0_7);
        c2_0_7 = _mm256_fmadd_ps(a_val2, b_vals, c2_0_7);
        c3_0_7 = _mm256_fmadd_ps(a_val3, b_vals, c3_0_7);
    }
    let c_ptr = c.as_mut_ptr();
    // Store the results from registers back to memory
    _mm256_storeu_ps(c_ptr.add(0), _mm256_add_ps(_mm256_loadu_ps(c_ptr.add(0)), c0_0_7));
    _mm256_storeu_ps(c_ptr.add(c_stride), _mm256_add_ps(_mm256_loadu_ps.add(c_stride), c1_0_7));
    _mm256_storeu_ps(c_ptr.add(2*c_stride), _mm256_add_ps(_mm256_loadu_ps(2*c_stride), c2_0_7));
    _mm256_storeu_ps(c_ptr.add(3*c_stride), _mm256_add_ps(_mm256_loadu_ps(3*c_stride), c3_0_7));
}
