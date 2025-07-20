use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;
use rand;
use ndarray::{Array, Array2};
use rublas::naive_matmul; 

fn bench_naive_matmul(c: &mut Criterion) {
    // Define matrix dimensions
    let m = 256;
    let k = 256;
    let n = 256;

    // Create random matrices
    let a = Array::from_shape_fn((m, k), |_| rand::random());
    let b = Array::from_shape_fn((k, n), |_| rand::random());
    let result = Array2::<f32>::zeros((m, n));

    // The benchmark
    c.bench_function("naive_matmul_256", |bencher| {
        // Create the result matrix inside the bencher to reset it for each run
        let mut result = Array2::<f32>::zeros((m, n));
        
        bencher.iter(|| {
            // Pass the inputs and the mutable result directly
            // black_box the output to ensure the work isn't optimized away
            naive_matmul(&a.view(), &b.view(), &mut result);
            black_box(&result);
        })
    });
}

criterion_group!(benches, bench_naive_matmul);
criterion_main!(benches);