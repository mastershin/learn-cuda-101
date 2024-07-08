use std::env;
use std::io::{self, Write};
use std::time::{Instant};

const SIZE: usize = 1000 * 1000 * 200;

fn cpu_vector_add(x: &[f32], y: &[f32], out: &mut [f32], size: usize) {
    for i in 0..size {
        out[i] = x[i] + y[i];
    }
}

fn initialize_data(x: &mut [f32], y: &mut [f32], size: usize) {
    for i in 0..size {
        let v = (i % 10 + 1) as f32;
        if i % 2 == 0 {
            x[i] = v * 2.0;
            y[i] = -v;
        } else {
            x[i] = v;
            y[i] = -v * 3.0;
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut loop_count = 50; // Default value

    if args.len() > 1 {
        loop_count = args[1].parse().unwrap_or(50);
    }

    println!("LOOP: {}", loop_count);

    // Allocate memory for vectors x and y
    let mut x: Vec<f32> = vec![0.0; SIZE];
    let mut y: Vec<f32> = vec![0.0; SIZE];
    let mut out: Vec<f32> = vec![0.0; SIZE];

    // Initialize vectors x and y
    initialize_data(&mut x, &mut y, SIZE);

    // CPU vector addition
    let start_cpu = Instant::now();

    for _ in 0..loop_count {
        print!(".");
        io::stdout().flush().unwrap(); // Flush the output
        cpu_vector_add(&x, &y, &mut out, SIZE);
    }
    let end_cpu = Instant::now();
    let cpu_duration = end_cpu.duration_since(start_cpu);

    println!();
    println!("CPU time: {} seconds", cpu_duration.as_secs_f64());

    // Calculate the average using a loop
    let mut avg: f32 = 0.0;
    for i in 0..SIZE {
        avg += out[i];
    }
    avg /= SIZE as f32;
    println!("Avg: {}", avg);
}
