# Fundamentals

1. **Vector Addition:** The "Hello World" of GPU programming. Add two arrays element-wise on the GPU, illustrating data transfer and kernel execution.
    * Python: Use libraries like CuPy or PyCUDA.
    * C/C++: Direct CUDA API interaction.
    * Rust: Explore rust-cuda or accelerate.
2. **Matrix Multiplication:**  A step up in complexity, showcasing thread organization and memory access patterns for performance.
    * Python: NumPy-like syntax with CuPy.
    * C/C++: Implement tiled matrix multiplication.
    * Rust: Leverage ndarray crate for matrix operations.
3. **Array Reduction (Sum/Max/Min):** Learn to efficiently combine data from multiple threads, introducing reduction techniques and atomic operations.
    * Python:  CuPy's reduction functions.
    * C/C++: Explore different reduction approaches.
    * Rust: Use Rayon for parallel iteration and reduction.

# Image Processing

4. **Image Blur:**  Apply a convolution filter (like a Gaussian blur) to an image, highlighting parallel image processing concepts. 
    * Python:  Use libraries like OpenCV with CUDA support.
    * C/C++: Manipulate pixel data directly.
5. **Image Histogram:** Calculate the color histogram of an image, introducing parallel reduction and memory access patterns for efficient data gathering.
    * JavaScript: Display the histogram results using libraries like Chart.js or D3.js.

# Numerical Computation

6. **Mandelbrot Set:**  Generate a fractal image by iterating a complex function, demonstrating parallel computation of independent pixels.
    * JavaScript: Render the resulting Mandelbrot set on a canvas element.
7. **Monte Carlo Pi Estimation:** Use random sampling to estimate the value of Pi, illustrating how CUDA can accelerate Monte Carlo simulations.

# Slightly More Advanced

8. **Particle Simulation (Basic):** Simulate a simple particle system with forces (e.g., gravity) and collisions. Introduce basic particle-particle interactions on the GPU.
    * JavaScript: Animate the particle movement in the browser.
9. **Game of Life (CUDA Kernel):** Implement the core rule update for Conway's Game of Life on the GPU, highlighting parallel neighbor calculations.
    * Python/C++/Rust: Handle display and user interaction in the host language.
10. **Simple Ray Tracing (First Hit):**  A challenging but rewarding project. Calculate the intersection of rays with a scene (e.g., spheres) to render a basic image.

# Key Objectives

* **Gradual Progression:** Start with the most straightforward examples (vector addition) and build up complexity.
* **Explain Concepts:**  Emphasize data parallelism, kernel execution, memory management, and how CUDA differs from traditional programming.
* **Language Focus:** Choose one primary language (like Python with CuPy) and potentially show snippets or concepts in others to illustrate different approaches.
* **Visualization:** Where applicable, link results to visual outputs (images, charts) to make the learning process more engaging. 
