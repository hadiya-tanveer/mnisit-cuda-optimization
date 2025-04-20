# MNIST Neural Network Classification (CUDA Optimized)

This project implements a feedforward neural network for handwritten digit classification using the MNIST dataset. It explores performance optimization through different versions, starting from a serial CPU-based implementation and gradually integrating CUDA for GPU acceleration. Each version focuses on enhancing performance through better use of GPU resources, memory management, and parallelism.

## Project Versions

### **Version 1: Serial Implementation (CPU)**
- Basic feedforward neural network.
- Forward and backward propagation done entirely on the CPU.
- No GPU acceleration.
- Acts as the baseline for performance comparison.

### **Version 2: Naive CUDA Implementation**
- Introduces CUDA kernels for matrix operations and activations.
- Forward and backward passes parallelized using simple GPU kernels.
- Frequent memory transfers between CPU and GPU lead to suboptimal performance.

### **Version 3: Optimized CUDA Implementation**
- Profiling revealed that `Forward()` and `Backward()` functions were the bottlenecks.
- Weights, biases, and hidden variables were offloaded to the GPU to reduce memory transfers.
- 2D GPU kernels were used for computationally intensive loops.
- Kernel launch parameters were tuned for performance, using 256 threads per block.
- Float data type was chosen over double to boost speed while accepting minimal precision loss.
- Achieves best overall execution time.

### **Version 4: cuBLAS-Accelerated Implementation**
- Leverages NVIDIA's cuBLAS library for high-performance matrix multiplications.
- Replaces custom CUDA kernels with cuBLAS operations for forward and backward propagation.
- While cuBLAS provides efficient operations, the overhead of integrating with smaller custom layers leads to performance trade-offs in this case.

---

## Performance Comparison

| Version | Description                | Execution Time |
|---------|----------------------------|----------------|
| V1      | Serial (CPU only)          | 79s            |
| V2      | Naive CUDA                 | 24s            |
| V3      | Optimized CUDA             | 14s            |
| V4      | cuBLAS                     | 28s            |

---

## Conclusion

The progression from a serial implementation to an optimized CUDA version demonstrates the impact of GPU acceleration and memory optimization in deep learning workloads. While cuBLAS offers powerful matrix operations, custom-tuned kernels in V3 proved more effective for this specific task. This project highlights the importance of profiling and targeted optimization when working with GPU-based neural networks.
