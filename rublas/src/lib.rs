use ndarray::{Array2, ArrayView2};

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
pub fn blocked_matmul(a: &ArrayView2<f32>, b: &ArrayView2<f32>, c: &mut ArrayView2<f32>){
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
                // matmul on the blocks
                // TODO: make this more efficient + add iterating way to ade
                for row in i..std::cmp::min(i+BLOCK_SIZE,m){
                    for col in j..std::cmp::min(j+BLOCK_SIZE,n){
                        let mut sum = 0.0; 
                        for depth in l..std::cmp::min(l+BLOCK_SIZE, k){
                            sum += a[[row, depth]]*b[[depth, col]];
                        }
                        c[[row, col]] += sum;
                    }
                }
            }
        }
    }
}