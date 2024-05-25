import cupy as cp
import numpy as np
import time

# start = time.time()
# rand1 = np.random.rand(1_000, 1_000)
# end = time.time() - start
# print(end * 1000, "ms")

# start = time.time()
# rand2 = cp.random.rand(1_000, 1_000)
# end = time.time() - start
# print(end * 1000, "ms")

def test_cupy() -> None:
    X = cp.random.rand(784, 60000)
    W1 = cp.random.rand(10, 784) - 0.5
    B1 = cp.random.rand(10, 1) - 0.5
    start = time.time()
    Z1 = W1.dot(X) + B1
    end = time.time()
    print((end - start) * 1000, "ms")

def test_numpy() -> None:
    X = np.random.rand(784, 60000)
    W1 = np.random.rand(10, 784) - 0.5
    B1 = np.random.rand(10, 1) - 0.5
    start = time.time()
    Z1 = W1.dot(X) + B1
    end = time.time()
    print((end - start) * 1000, "ms")

def main():
    test_cupy()
    test_numpy()

main()