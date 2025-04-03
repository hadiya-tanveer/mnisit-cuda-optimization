#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda.h>

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

// Activation functions on host (used only for initial testing)
void relu(double* x, int size) {
    for (int i = 0; i < size; i++) {
        x[i] = (x[i] > 0) ? x[i] : 0;
    }
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

// Neural network structure with additional device pointers.
typedef struct {
    double** W1;    // host 2D weight matrix for layer 1
    double** W2;    // host 2D weight matrix for layer 2
    double* b1;     // host biases for hidden layer
    double* b2;     // host biases for output layer

    // Flattened device arrays (row-major order)
    double* d_W1;
    double* d_W2;
    double* d_b1;
    double* d_b2;
} NeuralNetwork;

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

// Initialize neural network and allocate device memory.
NeuralNetwork* createNetwork() {
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

    // Flatten weight matrices.
    double* h_W1 = flattenMatrix(net->W1, HIDDEN_SIZE, INPUT_SIZE);
    double* h_W2 = flattenMatrix(net->W2, OUTPUT_SIZE, HIDDEN_SIZE);

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

    return net;
}

// Kernel to compute hidden layer with ReLU activation.
__global__ void forward_hidden_kernel(const double *W1, const double *b1,
                                        const double *input, double *hidden) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < HIDDEN_SIZE) {
        double sum = b1[i];
        for (int j = 0; j < INPUT_SIZE; j++) {
            sum += W1[i * INPUT_SIZE + j] * input[j];
        }
        hidden[i] = (sum > 0) ? sum : 0;
    }
}

// Kernel to compute output layer (logits).
__global__ void forward_output_kernel(const double *W2, const double *b2,
                                        const double *hidden, double *output) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < OUTPUT_SIZE) {
        double sum = b2[i];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            sum += W2[i * HIDDEN_SIZE + j] * hidden[j];
        }
        output[i] = sum;
    }
}

// Kernel to perform softmax on the output vector.
__global__ void softmax_kernel(double *output) {
    __shared__ double sum_shared;
    int i = threadIdx.x;
    double val = (i < OUTPUT_SIZE) ? exp(output[i]) : 0.0;
    if (i < OUTPUT_SIZE) {
        output[i] = val;
    }
    __syncthreads();

    if (i == 0) {
        double s = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            s += output[j];
        }
        sum_shared = s;
    }
    __syncthreads();

    if (i < OUTPUT_SIZE) {
        output[i] /= sum_shared;
    }
}

// Optimized forward pass using the GPU.
void forward(NeuralNetwork* net,
             double* input,   // host input vector (size INPUT_SIZE)
             double* hidden,  // host hidden vector (size HIDDEN_SIZE)
             double* output)  // host output vector (size OUTPUT_SIZE)
{
    double *d_input, *d_hidden, *d_output;
    cudaMalloc(&d_input, INPUT_SIZE * sizeof(double));
    cudaMalloc(&d_hidden, HIDDEN_SIZE * sizeof(double));
    cudaMalloc(&d_output, OUTPUT_SIZE * sizeof(double));

    cudaMemcpy(d_input, input, INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);

    int threadsPerBlock = 128;
    int numBlocks = (HIDDEN_SIZE + threadsPerBlock - 1) / threadsPerBlock;
    forward_hidden_kernel<<<numBlocks, threadsPerBlock>>>(net->d_W1, net->d_b1, d_input, d_hidden);
    cudaDeviceSynchronize();

    threadsPerBlock = 32;
    numBlocks = (OUTPUT_SIZE + threadsPerBlock - 1) / threadsPerBlock;
    forward_output_kernel<<<numBlocks, threadsPerBlock>>>(net->d_W2, net->d_b2, d_hidden, d_output);
    cudaDeviceSynchronize();

    softmax_kernel<<<1, OUTPUT_SIZE>>>(d_output);
    cudaDeviceSynchronize();

    cudaMemcpy(hidden, d_hidden, HIDDEN_SIZE * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(output, d_output, OUTPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_hidden);
    cudaFree(d_output);
}

// Kernel to compute hidden layer gradient for backpropagation.
__global__ void compute_d_hidden(double* d_hidden, const double* W2, const double* d_output, const double* hidden) {
    int i = threadIdx.x;
    if (i < HIDDEN_SIZE) {
        double grad = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            grad += W2[j * HIDDEN_SIZE + i] * d_output[j];
        }
        d_hidden[i] = grad * (hidden[i] > 0);
    }
}

// Kernel to update output layer weights.
__global__ void updateW2(double* W2, const double* d_output, const double* hidden, double learning_rate) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;  // output index
    int col = blockIdx.y * blockDim.y + threadIdx.y;  // hidden index
    if (row < OUTPUT_SIZE && col < HIDDEN_SIZE) {
        int idx = row * HIDDEN_SIZE + col;
        W2[idx] -= learning_rate * d_output[row] * hidden[col];
    }
}

// Kernel to update hidden layer weights.
__global__ void updateW1(double* W1, const double* d_hidden, const double* input, double learning_rate) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;  // hidden index
    int col = blockIdx.y * blockDim.y + threadIdx.y;    // input index
    if (row < HIDDEN_SIZE && col < INPUT_SIZE) {
        int idx = row * INPUT_SIZE + col;
        W1[idx] -= learning_rate * d_hidden[row] * input[col];
    }
}

// Backward pass: compute gradients and update device weights.
void backward(NeuralNetwork* net, double* input, double* hidden, double* output, double* target) {
    double d_output[OUTPUT_SIZE], d_hidden[HIDDEN_SIZE];
    // Compute gradient for output layer.
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        d_output[i] = output[i] - target[i];
    }

    // Allocate device memory for input and intermediate gradients.
    double *d_input, *d_hidden_d, *d_output_d;
    cudaMalloc(&d_input, INPUT_SIZE * sizeof(double));
    cudaMalloc(&d_hidden_d, HIDDEN_SIZE * sizeof(double));
    cudaMalloc(&d_output_d, OUTPUT_SIZE * sizeof(double));

    // Copy input and output gradients to device.
    cudaMemcpy(d_input, input, INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output_d, d_output, OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    // Also copy hidden layer (from forward pass) to device.
    double *d_hidden_forward;
    cudaMalloc(&d_hidden_forward, HIDDEN_SIZE * sizeof(double));
    cudaMemcpy(d_hidden_forward, hidden, HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice);

    // Compute hidden layer gradient on device.
    compute_d_hidden<<<1, HIDDEN_SIZE>>>(d_hidden_d, net->d_W2, d_output_d, d_hidden_forward);
    cudaDeviceSynchronize();

    // Copy the computed hidden gradient back to host (if needed for bias update)
    cudaMemcpy(d_hidden, d_hidden_d, HIDDEN_SIZE * sizeof(double), cudaMemcpyDeviceToHost);
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        // Use the host copy of d_hidden for bias update (if desired)
        d_hidden[i] = d_hidden[i]; // This line is just for clarity.
    }

    // Define grid and block dimensions for weight update kernels.
    dim3 blockDim(16, 16);
    dim3 gridDim_W2((OUTPUT_SIZE + blockDim.x - 1) / blockDim.x, (HIDDEN_SIZE + blockDim.y - 1) / blockDim.y);
    updateW2<<<gridDim_W2, blockDim>>>(net->d_W2, d_output_d, d_hidden_forward, LEARNING_RATE);
    cudaDeviceSynchronize();

    dim3 gridDim_W1((HIDDEN_SIZE + blockDim.x - 1) / blockDim.x, (INPUT_SIZE + blockDim.y - 1) / blockDim.y);
    updateW1<<<gridDim_W1, blockDim>>>(net->d_W1, d_hidden_d, d_input, LEARNING_RATE);
    cudaDeviceSynchronize();

    // Update biases on host and copy to device.
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        net->b2[i] -= LEARNING_RATE * d_output[i];
    }
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        // For d_hidden bias update, you could use the d_hidden computed on device.
        net->b1[i] -= LEARNING_RATE * d_hidden[i];
    }
    cudaMemcpy(net->d_b2, net->b2, OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(net->d_b1, net->b1, HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice);

    // Free allocated device memory.
    cudaFree(d_input);
    cudaFree(d_hidden_d);
    cudaFree(d_output_d);
    cudaFree(d_hidden_forward);
}

// Train the network on all training images.
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
    cudaFree(net->d_W1);
    cudaFree(net->d_W2);
    cudaFree(net->d_b1);
    cudaFree(net->d_b2);
    free(net);
}

// Main function.
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
    return 0;
}
