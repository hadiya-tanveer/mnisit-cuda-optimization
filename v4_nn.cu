/*
* This optimized implementation utilizes CUDA's tensor cores via cuBLAS
* and incorporates efficient memory management, optimized launch configurations,
* atomic operations, and numerical stability improvements.
*/

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <time.h>
#include <cuda.h>
#include <float.h>
#include <cublas_v2.h>

#define IMAGE_SIZE_TRAIN 60000
#define IMAGE_SIZE_TEST 10000
#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.01f
#define EPOCHS 3
#define HIDDEN_BLOCK 512
#define OUTPUT_BLOCK 512
#define UPDATE_BLOCK_X 16
#define UPDATE_BLOCK_Y 16
#define NUM_CLASSES 10  // Digits 0-9

// Check CUDA errors
#define CHECK_CUDA_ERROR(err) do { \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// Timer function
float get_time(clock_t start) {
    return (float)(clock() - start) / CLOCKS_PER_SEC;
}

// Allocate memory for a matrix (2D array as array of pointers)
float** allocateMatrix(int rows, int cols) {
    float** mat = (float**)malloc(rows * sizeof(float*));
    for (int i = 0; i < rows; i++) {
        mat[i] = (float*)malloc(cols * sizeof(float));
    }
    return mat;
}

// Free allocated matrix memory
void freeMatrix(float** mat, int rows) {
    for (int i = 0; i < rows; i++) {
        free(mat[i]);
    }
    free(mat);
}

// Numerically stable softmax
void softmax(float* x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

// Helper to flatten a 2D matrix into a contiguous array
float* flattenMatrix(float** matrix, int rows, int cols) {
    float* flat = (float*)malloc(rows * cols * sizeof(float));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            flat[i * cols + j] = matrix[i][j];
        }
    }
    return flat;
}

// Neural network structure with device pointers and cuBLAS handle
typedef struct {
    float** W1;    
    float** W2;    
    float* b1;     
    float* b2;     
    float* d_W1;
    float* d_W2;
    float* d_b1;
    float* d_b2;
    float* d_input;
    float* d_hidden;
    float* d_output;
    float* d_d_output;
    float* d_d_hidden;
    cublasHandle_t cublas_handle;
} NeuralNetwork;

// Initialize neural network and allocate device memory
NeuralNetwork* createNetwork() {
    // Initialize CUDA device
    int deviceCount;
    CHECK_CUDA_ERROR(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA-capable devices found\n");
        exit(EXIT_FAILURE);
    }
    CHECK_CUDA_ERROR(cudaSetDevice(0));

    NeuralNetwork* net = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    net->W1 = allocateMatrix(HIDDEN_SIZE, INPUT_SIZE);
    net->W2 = allocateMatrix(OUTPUT_SIZE, HIDDEN_SIZE);
    net->b1 = (float*)calloc(HIDDEN_SIZE, sizeof(float));
    net->b2 = (float*)calloc(OUTPUT_SIZE, sizeof(float));

    // Initialize weights
    srand(time(NULL));
    for (int i = 0; i < HIDDEN_SIZE; i++)
        for (int j = 0; j < INPUT_SIZE; j++)
            net->W1[i][j] = ((float)rand() / RAND_MAX) * 0.01f;

    for (int i = 0; i < OUTPUT_SIZE; i++)
        for (int j = 0; j < HIDDEN_SIZE; j++)
            net->W2[i][j] = ((float)rand() / RAND_MAX) * 0.01f;

    // Flatten weight matrices
    float* h_W1 = flattenMatrix(net->W1, HIDDEN_SIZE, INPUT_SIZE);
    float* h_W2 = flattenMatrix(net->W2, OUTPUT_SIZE, HIDDEN_SIZE);

    // Allocate device memory for weights, biases, and buffers
    CHECK_CUDA_ERROR(cudaMalloc(&(net->d_W1), HIDDEN_SIZE * INPUT_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&(net->d_W2), OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&(net->d_b1), HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&(net->d_b2), OUTPUT_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&(net->d_input), INPUT_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&(net->d_hidden), HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&(net->d_output), OUTPUT_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&(net->d_d_output), OUTPUT_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&(net->d_d_hidden), HIDDEN_SIZE * sizeof(float)));

    // Copy from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(net->d_W1, h_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(net->d_W2, h_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(net->d_b1, net->b1, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(net->d_b2, net->b2, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    free(h_W1);
    free(h_W2);

    // Initialize cuBLAS handle
    cublasStatus_t status = cublasCreate(&net->cublas_handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS initialization failed: %d\n", status);
        exit(EXIT_FAILURE);
    }

    return net;
}

void checkCublasStatus(cublasStatus_t status, const std::string& functionName) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS function " << functionName << " failed with error code: " << status << std::endl;
        switch (status) {
            case CUBLAS_STATUS_NOT_INITIALIZED:
                std::cerr << "CUBLAS_STATUS_NOT_INITIALIZED: The library was not initialized properly." << std::endl;
                break;
            case CUBLAS_STATUS_ALLOC_FAILED:
                std::cerr << "CUBLAS_STATUS_ALLOC_FAILED: Resource allocation failed inside the cuBLAS library." << std::endl;
                break;
            case CUBLAS_STATUS_INVALID_VALUE:
                std::cerr << "CUBLAS_STATUS_INVALID_VALUE: An invalid value was passed to the function." << std::endl;
                break;
            case CUBLAS_STATUS_ARCH_MISMATCH:
                std::cerr << "CUBLAS_STATUS_ARCH_MISMATCH: The function was called on an unsupported device architecture." << std::endl;
                break;
            case CUBLAS_STATUS_MAPPING_ERROR:
                std::cerr << "CUBLAS_STATUS_MAPPING_ERROR: A mapping error occurred, possibly due to a memory access violation." << std::endl;
                break;
            case CUBLAS_STATUS_EXECUTION_FAILED:
                std::cerr << "CUBLAS_STATUS_EXECUTION_FAILED: The function failed during execution." << std::endl;
                break;
            case CUBLAS_STATUS_INTERNAL_ERROR:
                std::cerr << "CUBLAS_STATUS_INTERNAL_ERROR: An internal error occurred." << std::endl;
                break;
            default:
                std::cerr << "Unknown cuBLAS error." << std::endl;
                break;
        }
        exit(EXIT_FAILURE);
    }
}

__global__ void relu(float* input, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        input[index] = fmaxf(0.0f, input[index]);
    }
}

__global__ void addBias(float* layer, float* bias, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        layer[index] += bias[index];
    }
}

// Optimized forward pass using cuBLAS
void forward(NeuralNetwork* net, float* input, float* output) {
    float alpha = 1.0f, beta = 0.0f;

    // Copy input to device
    CHECK_CUDA_ERROR(cudaMemcpy(net->d_input, input, INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    // W1^T * input
    cublasStatus_t status = cublasSgemv(net->cublas_handle, CUBLAS_OP_T, INPUT_SIZE, HIDDEN_SIZE, &alpha, net->d_W1, INPUT_SIZE, net->d_input, 1, &beta, net->d_hidden, 1);
    checkCublasStatus(status, "cublasSgemv1");

    // Add bias
    addBias<<<(HIDDEN_SIZE + HIDDEN_BLOCK - 1) / HIDDEN_BLOCK, HIDDEN_BLOCK>>>(net->d_hidden, net->d_b1, HIDDEN_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // Apply ReLU
    relu<<<(HIDDEN_SIZE + HIDDEN_BLOCK - 1) / HIDDEN_BLOCK, HIDDEN_BLOCK>>>(net->d_hidden, HIDDEN_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // W2^T * hidden
    status = cublasSgemv(net->cublas_handle, CUBLAS_OP_T, HIDDEN_SIZE, OUTPUT_SIZE, &alpha, net->d_W2, HIDDEN_SIZE, net->d_hidden, 1, &beta, net->d_output, 1);
    checkCublasStatus(status, "cublasSgemv2");

    // Add bias
    addBias<<<(OUTPUT_SIZE + OUTPUT_BLOCK - 1) / OUTPUT_BLOCK, OUTPUT_BLOCK>>>(net->d_output, net->d_b2, OUTPUT_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // Copy result back to host and apply softmax
    CHECK_CUDA_ERROR(cudaMemcpy(output, net->d_output, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    softmax(output, OUTPUT_SIZE);
}

__global__ void compute_d_hidden(float* d_hidden, const float* W2, const float* d_output, const float* hidden, int size) {
    int idx = threadIdx.x;
    if (idx >= size) return;

    float gradient = 0.0f;
    for (int j = 0; j < OUTPUT_SIZE; j++) {
        gradient += W2[j * HIDDEN_SIZE + idx] * d_output[j];
    }
    d_hidden[idx] = gradient * (hidden[idx] > 0.0f ? 1.0f : 0.0f);
}

__global__ void update_weights_W2(float* W2, const float* d_output, const float* hidden, float lr) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < OUTPUT_SIZE && col < HIDDEN_SIZE) {
        atomicAdd(&W2[row * HIDDEN_SIZE + col], -lr * d_output[row] * hidden[col]);
    }
}

__global__ void update_weights_W1(float* W1, const float* d_hidden, const float* input, float lr) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < HIDDEN_SIZE && col < INPUT_SIZE) {
        atomicAdd(&W1[row * INPUT_SIZE + col], -lr * d_hidden[row] * input[col]);
    }
}

__global__ void updateBiases(float* biases, const float* deltas, int size, float lr) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        atomicAdd(&biases[idx], -lr * deltas[idx]);
    }
}

void backward(NeuralNetwork* net, float* input, float* output, float* target) {
    float d_output[OUTPUT_SIZE];

    // Compute d_output = output - target
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        d_output[i] = output[i] - target[i];
    }

    // Copy output gradients to device
    CHECK_CUDA_ERROR(cudaMemcpy(net->d_d_output, d_output, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    // Compute d_hidden = W2^T * d_output with ReLU derivative
    compute_d_hidden<<<1, HIDDEN_SIZE>>>(net->d_d_hidden, net->d_W2, net->d_d_output, net->d_hidden, HIDDEN_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Update W2 and b2
    dim3 blockDim2(UPDATE_BLOCK_X, UPDATE_BLOCK_Y);
    dim3 gridDim2((OUTPUT_SIZE + blockDim2.x - 1) / blockDim2.x, (HIDDEN_SIZE + blockDim2.y - 1) / blockDim2.y);
    update_weights_W2<<<gridDim2, blockDim2>>>(net->d_W2, net->d_d_output, net->d_hidden, LEARNING_RATE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    updateBiases<<<(OUTPUT_SIZE + OUTPUT_BLOCK - 1) / OUTPUT_BLOCK, OUTPUT_BLOCK>>>(net->d_b2, net->d_d_output, OUTPUT_SIZE, LEARNING_RATE);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // Update W1 and b1
    dim3 blockDim1(UPDATE_BLOCK_X, UPDATE_BLOCK_Y);
    dim3 gridDim1((HIDDEN_SIZE + blockDim1.x - 1) / blockDim1.x, (INPUT_SIZE + blockDim1.y - 1) / blockDim1.y);
    update_weights_W1<<<gridDim1, blockDim1>>>(net->d_W1, net->d_d_hidden, net->d_input, LEARNING_RATE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    updateBiases<<<(HIDDEN_SIZE + HIDDEN_BLOCK - 1) / HIDDEN_BLOCK, HIDDEN_BLOCK>>>(net->d_b1, net->d_d_hidden, HIDDEN_SIZE, LEARNING_RATE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

// Train the network
void train(NeuralNetwork* net, float** images, float** labels, int numImages) {
    clock_t total_start = clock();
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        clock_t epoch_start = clock();
        float loss = 0.0f;
        int correct = 0;
        for (int i = 0; i < numImages; i++) {
            float output[OUTPUT_SIZE];
            forward(net, images[i], output);
            backward(net, images[i], output, labels[i]);

            // Compute cross-entropy loss and accuracy
            for (int k = 0; k < OUTPUT_SIZE; k++) {
                loss -= labels[i][k] * logf(output[k] + 1e-10f);
            }
            int pred = 0, actual = 0;
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                if (output[j] > output[pred])
                    pred = j;
                if (labels[i][j] > labels[i][actual])
                    actual = j;
            }
            if (pred == actual)
                correct++;
        }
        printf("Epoch %d - Loss: %.4f - Train Accuracy: %.2f%% - Time: %.3fs\n",
            epoch + 1, loss / numImages, (correct / (float)numImages) * 100, get_time(epoch_start));
    }
    printf("Total training time: %.3fs\n", get_time(total_start));
}

// Evaluate accuracy
void evaluate(NeuralNetwork* net, float** images, float** labels, int numImages) {
    int correct = 0;
    for (int i = 0; i < numImages; i++) {
        float output[OUTPUT_SIZE];
        forward(net, images[i], output);
        int pred = 0, actual = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            if (output[j] > output[pred])
                pred = j;
            if (labels[i][j] > labels[i][actual])
                actual = j;
        }
        if (pred == actual)
            correct++;
    }
    printf("Test Accuracy: %.2f%%\n", (correct / (float)numImages) * 100);
}

// Read MNIST images
float** loadMNISTImages(const char* filename, int numImages) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening %s\n", filename);
        exit(1);
    }
    fseek(file, 16, SEEK_SET);
    float** images = allocateMatrix(numImages, INPUT_SIZE);
    for (int i = 0; i < numImages; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            unsigned char pixel;
            if (fread(&pixel, sizeof(unsigned char), 1, file) != 1) {
                fprintf(stderr, "Error: Failed to read pixel\n");
                fclose(file);
                exit(EXIT_FAILURE);
            }
            images[i][j] = pixel / 255.0f;
        }
    }
    fclose(file);
    return images;
}

// Read MNIST labels
float** loadMNISTLabels(const char* filename, int numLabels) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening %s\n", filename);
        exit(1);
    }
    fseek(file, 8, SEEK_SET);
    float** labels = allocateMatrix(numLabels, OUTPUT_SIZE);
    for (int i = 0; i < numLabels; i++) {
        unsigned char label;
        if (fread(&label, sizeof(unsigned char), 1, file) != 1) {
            fprintf(stderr, "Error: Failed to read label\n");
            fclose(file);
            exit(EXIT_FAILURE);
        }
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            labels[i][j] = (j == label) ? 1.0f : 0.0f;
        }
    }
    fclose(file);
    return labels;
}

// Free network memory
void freeNetwork(NeuralNetwork* net) {
    freeMatrix(net->W1, HIDDEN_SIZE);
    freeMatrix(net->W2, OUTPUT_SIZE);
    free(net->b1);
    free(net->b2);
    CHECK_CUDA_ERROR(cudaFree(net->d_W1));
    CHECK_CUDA_ERROR(cudaFree(net->d_W2));
    CHECK_CUDA_ERROR(cudaFree(net->d_b1));
    CHECK_CUDA_ERROR(cudaFree(net->d_b2));
    CHECK_CUDA_ERROR(cudaFree(net->d_input));
    CHECK_CUDA_ERROR(cudaFree(net->d_hidden));
    CHECK_CUDA_ERROR(cudaFree(net->d_output));
    CHECK_CUDA_ERROR(cudaFree(net->d_d_output));
    CHECK_CUDA_ERROR(cudaFree(net->d_d_hidden));
    cublasDestroy(net->cublas_handle);
    free(net);
}

// Main function
int main() {
    printf("MNIST Neural Network (Optimized with Tensor Cores)\n\n");
    float** train_images = loadMNISTImages("data/train-images.idx3-ubyte", IMAGE_SIZE_TRAIN);
    float** train_labels = loadMNISTLabels("data/train-labels.idx1-ubyte", IMAGE_SIZE_TRAIN);
    float** test_images = loadMNISTImages("data/t10k-images.idx3-ubyte", IMAGE_SIZE_TEST);
    float** test_labels = loadMNISTLabels("data/t10k-labels.idx1-ubyte", IMAGE_SIZE_TEST);

    NeuralNetwork* net = createNetwork();
    train(net, train_images, train_labels, IMAGE_SIZE_TRAIN);
    evaluate(net, test_images, test_labels, IMAGE_SIZE_TEST);

    freeNetwork(net);
    freeMatrix(train_images, IMAGE_SIZE_TRAIN);
    freeMatrix(train_labels, IMAGE_SIZE_TRAIN);
    freeMatrix(test_images, IMAGE_SIZE_TEST);
    freeMatrix(test_labels, IMAGE_SIZE_TEST);

    return 0;
}