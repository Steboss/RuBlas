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