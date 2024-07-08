extern crate ndarray;
extern crate ndarray_rand;
extern crate rand;

use ndarray::prelude::*;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use std::env;
use std::io::Write;
use std::time::Instant;

const LOOP: usize = 200;

// Classic CPU matrix multiplication using for loop (slow)
#[allow(non_snake_case)]
fn matmul_cpu(A: &Array2<f32>, B: &Array2<f32>, C: &mut Array2<f32>, m: usize, n: usize, k: usize) {
    for i in 0..m {
        for j in 0..n {
            C[[i, j]] = 0.0;
            for p in 0..k {
                C[[i, j]] += A[[i, p]] * B[[p, j]];
            }
        }
    }
}

#[allow(non_snake_case)]
fn initialize_data(m: usize, n: usize, k: usize) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
    let A = Array2::random((m, k), Uniform::new(-1.0, 1.0));
    let B = Array2::random((k, n), Uniform::new(-1.0, 1.0));
    let C = Array2::<f32>::zeros((m, n));
    (A, B, C)
}

fn get_small_matrix_size() -> (usize, usize, usize) {
    (200, 150, 100)
}

fn get_medium_matrix_size() -> (usize, usize, usize) {
    (500, 300, 200)
}

fn get_large_matrix_size() -> (usize, usize, usize) {
    (4096, 1024, 1024)
}

fn parse_command_args() -> (usize, usize, usize) {
    let args: Vec<String> = env::args().collect();

    if args.len() == 2 {
        match args[1].as_str() {
            "s" => get_small_matrix_size(),
            "m" => get_medium_matrix_size(),
            "l" => get_large_matrix_size(),
            _ => {
                eprintln!("Invalid size argument. Use 's', 'm', 'l' or specify dimensions.");
                std::process::exit(1);
            }
        }
    } else if args.len() == 4 {
        let m = args[1].parse().expect("Invalid argument for m");
        let n = args[2].parse().expect("Invalid argument for n");
        let k = args[3].parse().expect("Invalid argument for k");
        (m, n, k)
    } else {
        eprintln!("Invalid arguments. Use 's', 'm', 'l' for predefined sizes or specify dimensions m, n, k.");
        std::process::exit(1);
    }
}

#[allow(non_snake_case)]
fn main() {
    let (m, n, k) = parse_command_args();

    println!(
        "Matrix Multiplication: A({}x{}) * B({}x{}) = C({}x{})",
        m, k, k, n, m, n
    );

    // Allocate memory for matrices A, B, and C
    let (A, B, mut C) = initialize_data(m, n, k);

    // Perform CPU matrix multiplication for verification
    matmul_cpu(&A, &B, &mut C, m, n, k);

    let start_cpu = Instant::now();

    for _ in 0..LOOP {
        print!(".");
        std::io::stdout().flush().unwrap();
        matmul_cpu(&A, &B, &mut C, m, n, k);
    }
    let end_cpu = Instant::now();
    let duration = end_cpu.duration_since(start_cpu);

    println!();
    println!("CPU time: {:.2} seconds", duration.as_secs_f64());

    let sum: f32 = C.sum();
    println!("Sum: {}", sum);
}
