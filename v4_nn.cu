/*
* This implementation utilizes CUDA's tensor cores.
* These are made specifically to handle matrix multiplication
* in a fast and efficient manner.
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
#define LEARNING_RATE 0.01
#define EPOCHS 3
#define BATCH_SIZE 64
#define NUM_CLASSES 10  // Digits 0-9

// Check CUDA errors
#define CHECK_CUDA_ERROR(err) do { \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// Timer function
double get_time(clock_t start) {
    return (double)(clock() - start) / CLOCKS_PER_SEC;
}

// Allocate memory for a matrix (2D array as array of pointers)
double** allocateMatrix(int rows, int cols) {
    double** mat = (double**)malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++) {
        mat[i] = (double*)malloc(cols * sizeof(double));
    }
    return mat;
}

// Free allocated matrix memory
void freeMatrix(double** mat, int rows) {
    for (int i = 0; i < rows; i++) {
        free(mat[i]);
    }
    free(mat);
}

void softmax(double* x, int size) {
    double sum = 0;
    for (int i = 0; i < size; i++) {
        x[i] = exp(x[i]);
        sum += x[i];
    }
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

// Helper to flatten a 2D matrix into a contiguous array
double* flattenMatrix(double** matrix, int rows, int cols) {
    double* flat = (double*)malloc(rows * cols * sizeof(double));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            flat[i * cols + j] = matrix[i][j];
        }
    }
    return flat;
}

// Neural network structure with additional device pointers and cuBLAS handle
typedef struct {
    double** W1;    
    double** W2;    
    double* b1;     
    double* b2;     
    double* d_W1;
    double* d_W2;
    double* d_b1;
    double* d_b2;
    cublasHandle_t cublas_handle; // Added cuBLAS handle
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
    net->b1 = (double*)calloc(HIDDEN_SIZE, sizeof(double));
    net->b2 = (double*)calloc(OUTPUT_SIZE, sizeof(double));

    srand(time(NULL));
    for (int i = 0; i < HIDDEN_SIZE; i++)
        for (int j = 0; j < INPUT_SIZE; j++)
            net->W1[i][j] = ((double)rand() / RAND_MAX) * 0.01;

    for (int i = 0; i < OUTPUT_SIZE; i++)
        for (int j = 0; j < HIDDEN_SIZE; j++)
            net->W2[i][j] = ((double)rand() / RAND_MAX) * 0.01;

    // Flatten weight matrices
    double* h_W1 = flattenMatrix(net->W1, HIDDEN_SIZE, INPUT_SIZE);
    double* h_W2 = flattenMatrix(net->W2, OUTPUT_SIZE, HIDDEN_SIZE);

    // Allocate device memory for weights and biases
    CHECK_CUDA_ERROR(cudaMalloc(&(net->d_W1), HIDDEN_SIZE * INPUT_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&(net->d_W2), OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&(net->d_b1), HIDDEN_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&(net->d_b2), OUTPUT_SIZE * sizeof(double)));

    // Copy from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(net->d_W1, h_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(net->d_W2, h_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(net->d_b1, net->b1, HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(net->d_b2, net->b2, OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));

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

__global__ void relu(double* input, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        input[index] = (input[index] > 0) ? input[index] : 0;
    }
}

__global__ void addBias(double* layer, double* bias, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        layer[index] += bias[index];
    }
}

// Optimized forward pass using cuBLAS
void forward(NeuralNetwork* net, double* input, double* hidden, double* output) {
    double alpha = 1.0, beta = 0.0;
    double *d_input, *d_hidden, *d_output;

    CHECK_CUDA_ERROR(cudaMalloc(&d_input, INPUT_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_hidden, HIDDEN_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, OUTPUT_SIZE * sizeof(double)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_input, input, INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));

    // W1^T * input
    cublasStatus_t status = cublasDgemv(net->cublas_handle, CUBLAS_OP_T, INPUT_SIZE, HIDDEN_SIZE, &alpha, net->d_W1, INPUT_SIZE, d_input, 1, &beta, d_hidden, 1);
    checkCublasStatus(status, "cublasDgemv1");

    // Add bias
    addBias<<<(HIDDEN_SIZE + 255) / 256, 256>>>(d_hidden, net->d_b1, HIDDEN_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // Apply ReLU
    relu<<<(HIDDEN_SIZE + 255) / 256, 256>>>(d_hidden, HIDDEN_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // W2^T * hidden
    status = cublasDgemv(net->cublas_handle, CUBLAS_OP_T, HIDDEN_SIZE, OUTPUT_SIZE, &alpha, net->d_W2, HIDDEN_SIZE, d_hidden, 1, &beta, d_output, 1);
    checkCublasStatus(status, "cublasDgemv2");

    // Add bias
    addBias<<<(OUTPUT_SIZE + 255) / 256, 256>>>(d_output, net->d_b2, OUTPUT_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // Copy result back to host and apply softmax
    CHECK_CUDA_ERROR(cudaMemcpy(output, d_output, OUTPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
    softmax(output, OUTPUT_SIZE);

    CHECK_CUDA_ERROR(cudaMemcpy(hidden, d_hidden, HIDDEN_SIZE * sizeof(double), cudaMemcpyDeviceToHost));

    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_hidden));
    CHECK_CUDA_ERROR(cudaFree(d_output));
}

__global__ void reluDerivative(double* hidden, double* d_hidden, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        d_hidden[index] = (hidden[index] > 0) ? d_hidden[index] : 0.0;
    }
}

__global__ void updateBias(double* bias, double* delta, int size, double learning_rate) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        bias[index] -= learning_rate * delta[index];
    }
}

__global__ void updateW2(double* W2, const double* d_output, const double* hidden, double learning_rate) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;  // output index
    int col = blockIdx.y * blockDim.y + threadIdx.y;  // hidden index
    if (row < OUTPUT_SIZE && col < HIDDEN_SIZE) {
        int idx = row * HIDDEN_SIZE + col;
        W2[idx] -= learning_rate * d_output[row] * hidden[col];
    }
}

__global__ void updateW1(double* W1, const double* d_hidden, const double* input, double learning_rate) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;  // hidden index
    int col = blockIdx.y * blockDim.y + threadIdx.y;  // input index
    if (row < HIDDEN_SIZE && col < INPUT_SIZE) {
        int idx = row * INPUT_SIZE + col;
        W1[idx] -= learning_rate * d_hidden[row] * input[col];
    }
}

void backward(NeuralNetwork* net, double* input, double* hidden, double* output, double* target) {
    double d_output[OUTPUT_SIZE], d_hidden[HIDDEN_SIZE];

    // Compute d_output = output - target
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        d_output[i] = output[i] - target[i];
    }

    // Allocate device memory
    double *d_input, *d_output_d, *d_hidden_d, *d_hidden_forward;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, INPUT_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output_d, OUTPUT_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_hidden_d, HIDDEN_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_hidden_forward, HIDDEN_SIZE * sizeof(double)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_input, input, INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_output_d, d_output, OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_hidden_forward, hidden, HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice));

    // Compute d_hidden = W2^T * d_output
    double alpha = 1.0, beta = 0.0;
    cublasStatus_t status = cublasDgemv(net->cublas_handle, CUBLAS_OP_N, HIDDEN_SIZE, OUTPUT_SIZE, &alpha, net->d_W2, HIDDEN_SIZE, d_output_d, 1, &beta, d_hidden_d, 1);
    checkCublasStatus(status, "cublasDgemv3");

    // Apply ReLU derivative
    int THREADS_PER_BLOCK = 256;
    int BLOCKS = (HIDDEN_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    reluDerivative<<<BLOCKS, THREADS_PER_BLOCK>>>(d_hidden_forward, d_hidden_d, HIDDEN_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    CHECK_CUDA_ERROR(cudaMemcpy(d_hidden, d_hidden_d, HIDDEN_SIZE * sizeof(double), cudaMemcpyDeviceToHost));

    dim3 blockDim(16, 16);
    dim3 gridDim_W2((OUTPUT_SIZE + blockDim.x - 1) / blockDim.x, (HIDDEN_SIZE + blockDim.y - 1) / blockDim.y);
    updateW2<<<gridDim_W2, blockDim>>>(net->d_W2, d_output_d, d_hidden_forward, LEARNING_RATE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    dim3 gridDim_W1((HIDDEN_SIZE + blockDim.x - 1) / blockDim.x, (INPUT_SIZE + blockDim.y - 1) / blockDim.y);
    updateW1<<<gridDim_W1, blockDim>>>(net->d_W1, d_hidden_d, d_input, LEARNING_RATE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Update biases
    updateBias<<<BLOCKS, THREADS_PER_BLOCK>>>(net->d_b2, d_output_d, OUTPUT_SIZE, LEARNING_RATE);
    updateBias<<<BLOCKS, THREADS_PER_BLOCK>>>(net->d_b1, d_hidden_d, HIDDEN_SIZE, LEARNING_RATE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Cleanup
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output_d));
    CHECK_CUDA_ERROR(cudaFree(d_hidden_d));
    CHECK_CUDA_ERROR(cudaFree(d_hidden_forward));
}

// Train the network
void train(NeuralNetwork* net, double** images, double** labels, int numImages) {
    clock_t total_start = clock();
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        clock_t epoch_start = clock();
        double loss = 0.0;
        int correct = 0;
        for (int i = 0; i < numImages; i++) {
            double hidden[HIDDEN_SIZE], output[OUTPUT_SIZE];
            forward(net, images[i], hidden, output);
            backward(net, images[i], hidden, output, labels[i]);

            // Compute cross-entropy loss and accuracy
            for (int k = 0; k < OUTPUT_SIZE; k++) {
                loss -= labels[i][k] * log(output[k] + 1e-10); // Add small epsilon to avoid log(0)
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
            epoch + 1, loss / numImages, (correct / (double)numImages) * 100, get_time(epoch_start));
    }
    printf("Total training time: %.3fs\n", get_time(total_start));
}

// Evaluate accuracy
void evaluate(NeuralNetwork* net, double** images, double** labels, int numImages) {
    int correct = 0;
    for (int i = 0; i < numImages; i++) {
        double hidden[HIDDEN_SIZE], output[OUTPUT_SIZE];
        forward(net, images[i], hidden, output);
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
    printf("Test Accuracy: %.2f%%\n", (correct / (double)numImages) * 100);
}

// Read MNIST images
double** loadMNISTImages(const char* filename, int numImages) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening %s\n", filename);
        exit(1);
    }
    fseek(file, 16, SEEK_SET);
    double** images = allocateMatrix(numImages, INPUT_SIZE);
    for (int i = 0; i < numImages; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            unsigned char pixel;
            if (fread(&pixel, sizeof(unsigned char), 1, file) != 1) {
                fprintf(stderr, "Error: Failed to read pixel\n");
                fclose(file);
                exit(EXIT_FAILURE);
            }
            images[i][j] = pixel / 255.0;
        }
    }
    fclose(file);
    return images;
}

// Read MNIST labels
double** loadMNISTLabels(const char* filename, int numLabels) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening %s\n", filename);
        exit(1);
    }
    fseek(file, 8, SEEK_SET);
    double** labels = allocateMatrix(numLabels, OUTPUT_SIZE);
    for (int i = 0; i < numLabels; i++) {
        unsigned char label;
        if (fread(&label, sizeof(unsigned char), 1, file) != 1) {
            fprintf(stderr, "Error: Failed to read label\n");
            fclose(file);
            exit(EXIT_FAILURE);
        }
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            labels[i][j] = (j == label) ? 1.0 : 0.0;
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
    cublasDestroy(net->cublas_handle); // Destroy cuBLAS handle
    free(net);
}

// Main function
int main() {
    printf("MNIST Neural Network\n\n");
    double** train_images = loadMNISTImages("data/train-images.idx3-ubyte", IMAGE_SIZE_TRAIN);
    double** train_labels = loadMNISTLabels("data/train-labels.idx1-ubyte", IMAGE_SIZE_TRAIN);
    double** test_images = loadMNISTImages("data/t10k-images.idx3-ubyte", IMAGE_SIZE_TEST);
    double** test_labels = loadMNISTLabels("data/t10k-labels.idx1-ubyte", IMAGE_SIZE_TEST);

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