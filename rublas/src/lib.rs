use ndarray::{Array2, ArrayView2};
use ndarray::s;
pub mod simd; 


/// Naive matrix multiplication.
pub fn naive_matmul(a: &ArrayView2<f32>, b: &ArrayView2<f32>, c: &mut Array2<f32>) {
    /* Perform a simple matmul between array a and b to get c
    The code uses a simple triple for loop. */
    let (m, k) = a.dim();
    let (_k, n) = b.dim();

    // Ensure dimensions are compatible
    assert_eq!(k, _k);
    assert_eq!(c.dim(), (m, n));

    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for l in 0..k {
                sum += a[[i, l]] * b[[l, j]];
            }
            c[[i, j]] = sum;
        }
    }
}

// Blocking strategy (Cache Locality)
pub fn blocked_matmul(a: &ArrayView2<f32>, b: &ArrayView2<f32>, c: &mut Array2<f32>){
    /* The following function implements the blocking strategy 
    The idea is to compute the matmul across blocks of matrices */
    const BLOCK_SIZE: usize = 32; // Do we need to fine tune this? 
    let (m, k) = a.dim(); 
    let (_k, n) = b.dim(); 
    assert_eq!(k, _k); 
    assert_eq!(c.dim(), (m, n));

    // block iteration
    for i in (0..m).step_by(BLOCK_SIZE) {
        for j in (0..n).step_by(BLOCK_SIZE) {
            for l in (0..k).step_by(BLOCK_SIZE) {
                for row in i..std::cmp::min(i + BLOCK_SIZE, m){
                    for col in j..std::cmp::min(j + BLOCK_SIZE, n){
                        let mut sum = 0.0; 
                        for depth in l..std::cmp::min(l + BLOCK_SIZE, k){
                            sum += a[[row, depth]] * b[[depth, col]];
                        }
                        c[[row, col]]+=sum;
                    }
                }
            }
        }
    }
}

pub fn packed_matmul(a: &ArrayView2<f32>, b: &ArrayView2<f32>, c: &mut Array2<f32>) {
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
                micro_kernel(
                    m_block_size, n_block_size, k_block_size,
                    &pack_a, &pack_b,
                    &mut c.slice_mut(s![i..m_block_end, j..n_block_end]),
                );
            }
        }
    }
}

// This kernel multiplies the packed, contiguous blocks.
fn micro_kernel(
    m: usize, n: usize, k: usize,
    pack_a: &[f32],
    pack_b: &[f32],
    c: &mut ndarray::ArrayViewMut2<f32>,
) {
    for depth in 0..k {
        for row in 0..m {
            // Because A is packed column-major, this access is linear
            let a_val = pack_a[row * k + depth];
            for col in 0..n {
                // Because B is packed row-major, this access is linear
                c[[row, col]] += a_val * pack_b[depth * n + col];
            }
        }
    }
}