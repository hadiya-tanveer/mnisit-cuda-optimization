/*
* Optimized neural network classification with CUDA, shared memory, batch processing, and reduced host-device transfers.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda.h>
#include <float.h>

#define IMAGE_SIZE_TRAIN 60000
#define IMAGE_SIZE_TEST 10000
#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.01
#define EPOCHS 3
#define BATCH_SIZE 64
#define NUM_CLASSES 10  // Digits 0-9

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

// Helper to flatten a 2D matrix into a contiguous array.
double* flattenMatrix(double** matrix, int rows, int cols) {
    double* flat = (double*)malloc(rows * cols * sizeof(double));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            flat[i * cols + j] = matrix[i][j];
        }
    }
    return flat;
}

// Neural network structure with device pointers and batch buffers.
typedef struct {
    // Host side weights and biases.
    double** W1;    
    double** W2;    
    double* b1;     
    double* b2;     

    // Device side weights and biases.
    double* d_W1;
    double* d_W2;
    double* d_b1;
    double* d_b2;

    // Device buffers for forward/backward passes.
    double* d_input;
    double* d_hidden;
    double* d_output;
    double* d_d_output;
    double* d_d_hidden;

} NeuralNetwork;

// Initialize neural network and allocate device memory.
NeuralNetwork* createNetwork() {
    //printf("Here -5\n");

    NeuralNetwork* net = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    net->W1 = allocateMatrix(HIDDEN_SIZE, INPUT_SIZE);
    net->W2 = allocateMatrix(OUTPUT_SIZE, HIDDEN_SIZE);
    net->b1 = (double*)calloc(HIDDEN_SIZE, sizeof(double));
    net->b2 = (double*)calloc(OUTPUT_SIZE, sizeof(double));
    //printf("Here -4\n");

    srand(time(NULL));
    for (int i = 0; i < HIDDEN_SIZE; i++)
        for (int j = 0; j < INPUT_SIZE; j++)
            net->W1[i][j] = ((double)rand() / RAND_MAX) * 0.01;

    for (int i = 0; i < OUTPUT_SIZE; i++)
        for (int j = 0; j < HIDDEN_SIZE; j++)
            net->W2[i][j] = ((double)rand() / RAND_MAX) * 0.01;

    // Flatten weight matrices.
    double* h_W1 = flattenMatrix(net->W1, HIDDEN_SIZE, INPUT_SIZE);
    double* h_W2 = flattenMatrix(net->W2, OUTPUT_SIZE, HIDDEN_SIZE);
    //printf("Here -3\n");

    // Allocate device memory for weights and biases.
    cudaMalloc(&(net->d_W1), HIDDEN_SIZE * INPUT_SIZE * sizeof(double));
    cudaMalloc(&(net->d_W2), OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double));
    cudaMalloc(&(net->d_b1), HIDDEN_SIZE * sizeof(double));
    cudaMalloc(&(net->d_b2), OUTPUT_SIZE * sizeof(double));

    // Copy from host to device.
    cudaMemcpy(net->d_W1, h_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(net->d_W2, h_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(net->d_b1, net->b1, HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(net->d_b2, net->b2, OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);

    free(h_W1);
    free(h_W2);
    //printf("Here -2\n");

    // Allocate device memory for forward/backward passes.
    cudaMalloc(&net->d_input, INPUT_SIZE * sizeof(double));
    cudaMalloc(&net->d_hidden, HIDDEN_SIZE * sizeof(double));
    cudaMalloc(&net->d_output, OUTPUT_SIZE * sizeof(double));
    cudaMalloc(&net->d_d_output, OUTPUT_SIZE * sizeof(double));
    cudaMalloc(&net->d_d_hidden, HIDDEN_SIZE * sizeof(double));
    //printf("Here -1\n");
    return net;
}
// Kernel to compute hidden layer with shared memory for input.
__global__ void forward_hidden_kernel(const double *W1, const double *B1, const double *input, double *hidden) {
    extern __shared__ double s_input[];
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Load input into shared memory
    for (int i = tid; i < INPUT_SIZE; i += blockDim.x) {
        s_input[i] = input[i];
    }
    __syncthreads();

    if (index >= HIDDEN_SIZE) return;
    
    double sum = B1[index];
    for (int j = 0; j < INPUT_SIZE; j++) {
        sum += W1[index * INPUT_SIZE + j] * s_input[j];
    }
    hidden[index] = (sum > 0) ? sum : 0;
}

// Kernel to compute output layer with shared memory for hidden.
__global__ void forward_output_kernel(const double *W2, const double *b2, const double *hidden, double *output) {
    extern __shared__ double s_hidden[];
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Load hidden into shared memory
    for (int i = tid; i < HIDDEN_SIZE; i += blockDim.x) {
        s_hidden[i] = hidden[i];
    }
    __syncthreads();

    if (index >= OUTPUT_SIZE) return;
    
    double sum = b2[index];
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        sum += W2[index * HIDDEN_SIZE + j] * s_hidden[j];
    }
    output[index] = sum;
}
// Optimized forward pass using pre-allocated device memory.
void forward(NeuralNetwork* net, double* input, double* hidden, double* output) {
    cudaMemcpy(net->d_input, input, INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocks = (HIDDEN_SIZE + threadsPerBlock - 1) / threadsPerBlock;
    forward_hidden_kernel<<<blocks, threadsPerBlock, INPUT_SIZE * sizeof(double)>>>(
        net->d_W1, net->d_b1, net->d_input, net->d_hidden);
    cudaDeviceSynchronize();

    threadsPerBlock = 256;
    blocks = (OUTPUT_SIZE + threadsPerBlock - 1) / threadsPerBlock;
    forward_output_kernel<<<blocks, threadsPerBlock, HIDDEN_SIZE * sizeof(double)>>>(
        net->d_W2, net->d_b2, net->d_hidden, net->d_output);
    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(output, net->d_output, OUTPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(hidden, net->d_hidden, HIDDEN_SIZE * sizeof(double), cudaMemcpyDeviceToHost);

    // Apply softmax on host
    softmax(output, OUTPUT_SIZE);
}

// Kernel to compute hidden layer gradients with shared memory for d_output.
__global__ void compute_d_hidden(double* d_hidden, const double* W2, const double* d_output, const double* hidden) {
    extern __shared__ double s_d_output[];
    int index = threadIdx.x;
    if (index >= HIDDEN_SIZE) return;

    // Load d_output into shared memory
    for (int i = index; i < OUTPUT_SIZE; i += blockDim.x) {
        s_d_output[i] = d_output[i];
    }
    __syncthreads();

    double gradient = 0;
    for (int j = 0; j < OUTPUT_SIZE; j++) {
        gradient += W2[j * HIDDEN_SIZE + index] * s_d_output[j];
    }
    d_hidden[index] = gradient * (hidden[index] > 0);
}

// Kernel to update W2 with shared memory for d_output and hidden.
__global__ void updateW2(double* W2, const double* d_output, const double* hidden, double learning_rate) {
    extern __shared__ double shared_mem[];
    double* s_d_output = shared_mem;
    double* s_hidden = shared_mem + blockDim.x;

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    // Load d_output and hidden into shared memory
    if (threadIdx.y == 0 && row < OUTPUT_SIZE) {
        s_d_output[threadIdx.x] = d_output[row];
    }
    if (threadIdx.x == 0 && col < HIDDEN_SIZE) {
        s_hidden[threadIdx.y] = hidden[col];
    }
    __syncthreads();

    if (row < OUTPUT_SIZE && col < HIDDEN_SIZE) {
        int idx = row * HIDDEN_SIZE + col;
        W2[idx] -= learning_rate * s_d_output[threadIdx.x] * s_hidden[threadIdx.y];
    }
}

// Kernel to update W1 with shared memory for d_hidden and input.
__global__ void updateW1(double* W1, const double* d_hidden, const double* input, double learning_rate) {
    extern __shared__ double shared_mem[];
    double* s_d_hidden = shared_mem;
    double* s_input = shared_mem + blockDim.x;

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    // Load d_hidden and input into shared memory
    if (threadIdx.y == 0 && row < HIDDEN_SIZE) {
        s_d_hidden[threadIdx.x] = d_hidden[row];
    }
    if (threadIdx.x == 0 && col < INPUT_SIZE) {
        s_input[threadIdx.y] = input[col];
    }
    __syncthreads();

    if (row < HIDDEN_SIZE && col < INPUT_SIZE) {
        int idx = row * INPUT_SIZE + col;
        W1[idx] -= learning_rate * s_d_hidden[threadIdx.x] * s_input[threadIdx.y];
    }
}

// Backward pass with pre-allocated memory and shared kernels.
void backward(NeuralNetwork* net, double* input, double* hidden, double* output, double* target) {
    double d_output_host[OUTPUT_SIZE];
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        d_output_host[i] = output[i] - target[i];
    }

    cudaMemcpy(net->d_d_output, d_output_host, OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    //printf("yahan 1?\n");
    // Compute d_hidden using device pointers
    int threadsPerBlock = 128;
    int blocks = (HIDDEN_SIZE + threadsPerBlock - 1) / threadsPerBlock;
    compute_d_hidden<<<blocks, threadsPerBlock, OUTPUT_SIZE * sizeof(double)>>>(
        net->d_d_hidden, net->d_W2, net->d_d_output, net->d_hidden);
    cudaDeviceSynchronize();
    //printf("yahan 2?\n");

    // Update W2
    dim3 blockDim(16, 16);
    dim3 gridDim_W2((OUTPUT_SIZE + blockDim.x - 1) / blockDim.x, (HIDDEN_SIZE + blockDim.y - 1) / blockDim.y);
    size_t sharedMemSize = (blockDim.x + blockDim.y) * sizeof(double);
    updateW2<<<gridDim_W2, blockDim, sharedMemSize>>>(net->d_W2, net->d_d_output, net->d_hidden, LEARNING_RATE);
    cudaDeviceSynchronize();
    //printf("yahan 3\n?");

    // Update W1
    dim3 gridDim_W1((HIDDEN_SIZE + blockDim.x - 1) / blockDim.x, (INPUT_SIZE + blockDim.y - 1) / blockDim.y);
    updateW1<<<gridDim_W1, blockDim, sharedMemSize>>>(net->d_W1, net->d_d_hidden, net->d_input, LEARNING_RATE);
    cudaDeviceSynchronize();
    //printf("yahan 4?\n");

    // Update biases on host and copy to device
    for (int i = 0; i < OUTPUT_SIZE; i++) 
        net->b2[i] -= LEARNING_RATE * d_output_host[i];
    //printf("where ?\n");
    double d_hidden_host[HIDDEN_SIZE];
    cudaMemcpy(d_hidden_host, net->d_d_hidden, HIDDEN_SIZE * sizeof(double), cudaMemcpyDeviceToHost);

    for (int i = 0; i < HIDDEN_SIZE; i++) {
        net->b1[i] -= LEARNING_RATE * d_hidden_host[i];
    }
    //printf("yahan 5?\n");

    cudaMemcpy(net->d_b2, net->b2, OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(net->d_b1, net->b1, HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    //printf("yahan 6?\n");
}

// Batch training function
void train(NeuralNetwork* net, double** images, double** labels, int numImages) {
    clock_t total_start = clock();
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        clock_t epoch_start = clock();
        double loss = 0.0;
        int correct = 0;
        for (int i = 0; i < numImages; i ++) {
            double hidden[HIDDEN_SIZE], output[OUTPUT_SIZE];
            //printf("yahan 1\n");
            forward(net, images[i], hidden, output);
            //printf("yahan 2\n");
            backward(net, images[i], hidden, output, labels[i]);
            //printf("yahan 3\n");
            // Compute cross-entropy loss and accuracy.
            for (int k = 0; k < OUTPUT_SIZE; k++) {
                loss -= labels[i][k] * log(output[k]);
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
// Evaluate accuracy on test images.
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

// Read MNIST images.
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

// Read MNIST labels.
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

// Free network memory, including device memory.
void freeNetwork(NeuralNetwork* net) {
    freeMatrix(net->W1, HIDDEN_SIZE);
    freeMatrix(net->W2, OUTPUT_SIZE);
    free(net->b1);
    free(net->b2);
    free(net);
    
    cudaFree(net->d_W1);
    cudaFree(net->d_W2);
    cudaFree(net->d_b1);
    cudaFree(net->d_b2);
}

// Main function.
int main() {
    printf("MNIST Neural Network 1\n\n");
    //printf("Here 1\n");
    double** train_images = loadMNISTImages("data/train-images.idx3-ubyte", IMAGE_SIZE_TRAIN);
    double** train_labels = loadMNISTLabels("data/train-labels.idx1-ubyte", IMAGE_SIZE_TRAIN);
    double** test_images = loadMNISTImages("data/t10k-images.idx3-ubyte", IMAGE_SIZE_TEST);
    double** test_labels = loadMNISTLabels("data/t10k-labels.idx1-ubyte", IMAGE_SIZE_TEST);
    //printf("Here 2\n");
    NeuralNetwork* net = createNetwork();
    //printf("Here 3\n");
    train(net, train_images, train_labels, IMAGE_SIZE_TRAIN);
    //printf("Here 4\n");
    evaluate(net, test_images, test_labels, IMAGE_SIZE_TEST);
    //printf("Here 5\n");
    freeNetwork(net);
    //printf("Here 6\n");
    return 0;
}
