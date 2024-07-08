"""
TODO: (2024-07-08): C++ version is still faster than cuDF
TODO: Find out why cuDF is slower than C++ (both GPU and non-GPU) version

To install cuDF, https://docs.rapids.ai/install

## Install cuDF, using pip, CUDA12
pip install \
    --extra-index-url=https://pypi.nvidia.com \
    cudf-cu12==24.6.* dask-cudf-cu12==24.6.* cuml-cu12==24.6.* \
    cugraph-cu12==24.6.* cuspatial-cu12==24.6.* cuproj-cu12==24.6.* \
    cuxfilter-cu12==24.6.* cucim-cu12==24.6.* pylibraft-cu12==24.6.* \
    raft-dask-cu12==24.6.* cuvs-cu12==24.6.*
"""

import cudf
import numpy as np
import argparse
import time

LOOP = 200


def initialize_data(m, n, k):
    # Calculate the size of matrices A, B, and C
    sizeA = m * k
    sizeB = k * n
    sizeC = m * n

    # Initialize matrices A and B with sequential float numbers
    A = np.zeros(sizeA, dtype=np.float32)
    B = np.zeros(sizeB, dtype=np.float32)
    C = np.zeros(sizeC, dtype=np.float32)

    for i in range(sizeA):
        A[i] = float(i) if i % 2 == 0 else -float(i)

    for i in range(sizeB):
        if i % 3 == 0:
            B[i] = 1.0
        elif i % 3 == 1:
            B[i] = 2.0
        else:
            B[i] = 3.0

    # Convert numpy arrays to cuDF DataFrames and reshape them to 2D matrices
    A = cudf.DataFrame.from_records(A.reshape(m, k))
    B = cudf.DataFrame.from_records(B.reshape(k, n))
    C = cudf.DataFrame.from_records(C.reshape(m, n))

    return A, B, C


def get_small_matrix_size():
    m = 200
    n = 150
    k = 100
    return m, n, k


def get_medium_matrix_size():
    m = 500
    n = 300
    k = 200
    return m, n, k


def get_large_matrix_size():
    m = 4096
    n = 1024
    k = 1024
    return m, n, k


def matrix_multiplication(A, B):
    # Perform matrix multiplication using cuDF
    C = A.dot(B)
    return C


def parse_args():
    parser = argparse.ArgumentParser(description="Matrix Multiplication with cuDF")
    parser.add_argument(
        "size",
        type=str,
        nargs="?",
        help="Matrix size: s (small), m (medium), l (large) or specify m, n, k",
        default="s",
    )
    parser.add_argument(
        "dimensions", type=int, nargs="*", help="Specify dimensions m, n, k"
    )

    args = parser.parse_args()

    if args.size in ["s", "m", "l"] and not args.dimensions:
        if args.size == "s":
            m, n, k = get_small_matrix_size()
        elif args.size == "m":
            m, n, k = get_medium_matrix_size()
        elif args.size == "l":
            m, n, k = get_large_matrix_size()
        else:
            raise ValueError(
                "Invalid size argument. Use 's', 'm', 'l' or specify dimensions."
            )
    elif len(args.dimensions) == 3:
        m, n, k = args.dimensions
    else:
        raise ValueError(
            "Invalid arguments. Use 's', 'm', 'l' for predefined sizes or specify dimensions m, n, k."
        )

    return m, n, k


if __name__ == "__main__":

    m, n, k = parse_args()
    print(f"Matrix size: {m}x{k} * {k}x{n} = {m}x{n}")

    A, B, C = initialize_data(m, n, k)

    # Time the matrix multiplication loop
    start_time = time.time()
    for _ in range(LOOP):
        print(".", end="", flush=True)

        C = matrix_multiplication(A, B)
    print()

    end_time = time.time()
    duration = end_time - start_time

    print(f"Time taken for {LOOP} matrix multiplications: {duration:.2f} seconds")
