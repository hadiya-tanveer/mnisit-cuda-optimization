# MNIST Digit Classification Using Neural Networks and CUDA Optimization

---

## Abstract

The MNIST handwritten digit classification problem is a well-known benchmark in computer vision and deep learning. It includes 60,000 training and 10,000 testing images of 28x28 grayscale digits (0–9).

Neural networks with ReLU activation and softmax output are ideal for this task. The model is trained using forward and backward propagation with optimization techniques like gradient descent.
![MNIST Dataset Sample]([images/mnist_sample.png](https://images.app.goo.gl/QNWwcdFGvkwhdSyw6))

---

## Overview

This project used a neural network to classify digits from the MNIST dataset. The network consisted of an input layer, hidden layer with ReLU, and an output layer with softmax. Training involved comparing predicted outputs to actual labels and updating weights through backpropagation.

Once trained, the model’s performance was evaluated on unseen test data.

---

## Code Overview

Profiling revealed the `Forward()` and `Backward()` functions as the most time-consuming. Optimization involved:

- Offloading weights, biases, and hidden variables to the GPU.
- Replacing 2D CPU loops with GPU kernels.
- Reducing memory transfers and kernel launches.
- Using `float` instead of `double` for better performance.

---

## Implementation

### 1. Serial Implementation (V1)

- Forward pass: input → hidden (ReLU) → output (softmax)
- Backpropagation: error used to adjust weights with gradient descent
- Evaluation on test data

**Performance:**

| Epoch | Accuracy (%) | Time (s) |
|-------|--------------|----------|
| 01    | 92.02        | 26.13    |
| 02    | 96.89        | 26.14    |
| 03    | 97.86        | 26.25    |

- **Total Accuracy:** 97.00%  
- **Total Training Time:** 78.65s  

---

### 2. Naive Implementation (V2)

- GPU used for computations, but with frequent `cudaMemcpy` for each variable.
- Showed benefits of GPU parallelism but suffered from memory transfer overhead.

**Performance:**

| Epoch | Accuracy (%) | Time (s) |
|-------|--------------|----------|
| 01    | 91.92        | 8.34     |
| 02    | 96.91        | 8.34     |
| 03    | 97.90        | 8.32     |

- **Total Accuracy:** 96.00%  
- **Total Training Time:** 24.93s  

---

### 3. Optimized Implementation (V3)

Key improvements:

- Used shared memory to reduce latency.
- Reduced CUDA kernel launches.
- Combined operations into fewer loops.
- Used atomic functions to ensure conflict-free updates.
- Minimized `cudaMemcpy` calls (from 16 to ~4).
- Adjusted launch configuration for better resource use.

**Performance:**

| Epoch | Accuracy (%) | Time (s) |
|-------|--------------|----------|
| 01    | 91.86        | 4.77     |
| 02    | 96.89        | 4.78     |
| 03    | 97.80        | 4.77     |

- **Total Accuracy:** 96.82%  
- **Total Training Time:** 14.31s  

> Used `float` instead of `double` for better performance, as precision loss was acceptable.

---

### 4. Tensor Cores Implementation (V4)

- Used NVIDIA tensor cores via cuBLAS library (e.g., `cublasDgemv`, `cublasDger`).
- Offloaded key linear algebra operations for faster computation.
- Reduced computational time and improved memory efficiency.

**Performance:**

| Epoch | Accuracy (%) | Time (s) |
|-------|--------------|----------|
| 01    | 92.02        | 2.97     |
| 02    | 96.87        | 2.98     |
| 03    | 97.81        | 2.97     |

- **Total Accuracy:** 96.99%  
- **Total Training Time:** 8.93s  

---

## Performance Analysis

| Version                  | Execution Time (s) | Speedup (vs V1) |
|--------------------------|--------------------|------------------|
| Version 2 (Naive)        | 24.93               | 3.29×            |
| Version 3 (Optimized)    | 14.31               | 5.64×            |
| Version 4 (Tensor Cores) | 9.00                | 8.73×            |

- Performance improved significantly with CUDA optimizations.
- Key techniques included reducing kernel launches, optimizing shared memory, using `float`, and minimizing memory transfers.

---

## Conclusion

Using a neural network for MNIST digit classification produced successful results. CUDA acceleration significantly sped up training, particularly in forward and backward propagation.

By offloading heavy computations to the GPU, minimizing memory transfers, and using optimized kernel configurations, the system achieved faster and more efficient training.

These results demonstrate how parallel computing with CUDA can improve scalability and responsiveness in deep learning applications.
---
