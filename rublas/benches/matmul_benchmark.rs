use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;
use rand;
use ndarray::{Array, Array2};
use rublas::{naive_matmul, blocked_matmul, packed_matmul}; 
use rublas::simd::matmul;

fn benchmark_suite(c: &mut Criterion) -> &mut Criterion {
    // Define matrix dimensions
    let m = 256;
    let k = 256;
    let n = 256;

    // Create random matrices
    let a = Array::from_shape_fn((m, k), |_| rand::random());
    let b = Array::from_shape_fn((k, n), |_| rand::random());

    // NAIVE 
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
    // BLOCK 
    c.bench_function("blocked_matmul_256", | bencher| {
        let mut result = Array2::<f32>::zeros((m, n));
        bencher.iter(||{
            blocked_matmul(&a.view(), &b.view(), &mut result);
            black_box(&result);
        })
    });
    // PACK 
    c.bench_function("packed_matmul_256", | bencher|{
        let mut result = Array2::<f32>::zeros((m, n));
        bencher.iter(||{
            packed_matmul(&a.view(), &b.view(), &mut result);
            black_box(&result);
        })
    });
    // SIMD 
    c.bench_function("packed_simd_matmul_256", | bencher | {
        let mut result = Array2::<f32>::zeros((m, n)); 
        bencher.iter(|| unsafe{
            matmul(&a.view(), &b.view(), &mut result);
            black_box(&result);
        })
    })
}

criterion_group!(benches, benchmark_suite);
criterion_main!(benches);