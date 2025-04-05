%%writefile tensorcore_nn.cu

/*
* This implementation utilizes CUDA's tensor cores.
* These are made specifically to handle matrix multiplication
* in fast and efficient manner.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <cublas_v2.h>

#define IMAGE_SIZE 60000
#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.01
#define EPOCHS 3
#define BATCH_SIZE 64
#define NUM_CLASSES 10  // Digits 0-9

// Weights initialized at Device side.
double* d_W1, * d_W2, *d_B1;

// Timer function
double get_time(clock_t start) {
    return (double)(clock() - start) / CLOCKS_PER_SEC;
}

// Allocate memory for a matrix
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

// Activation functions
// NOTE: Since we have made 'hidden' a gpu variable, to avoid memcpy RELU is made into kernel.
__global__ void reluKernel(double* x, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        x[idx] = (x[idx] > 0) ? x[idx] : 0;
    }
}

void relu(double* d_x, int size) {
    // Use 128 threads per block, just one block since size is fixed at 128
    int blockSize = 128;
    int gridSize = 1; // Only one block

    // Launch the ReLU kernel
    reluKernel<<<gridSize, blockSize>>>(d_x, size);

    // Synchronize to ensure computation is finished
    cudaDeviceSynchronize();
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

// Neural network structure
typedef struct {
    double** W1;
    double** W2;
    double* b1;
    double* b2;
} NeuralNetwork;

// Initialize neural network
NeuralNetwork* createNetwork() {
    NeuralNetwork* net = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    net->W1 = allocateMatrix(HIDDEN_SIZE, INPUT_SIZE);
    net->W2 = allocateMatrix(OUTPUT_SIZE, HIDDEN_SIZE);
    net->b1 = (double*)calloc(HIDDEN_SIZE, sizeof(double));
    net->b2 = (double*)calloc(OUTPUT_SIZE, sizeof(double));

    // Note: These are calculated as GPU variable and then copied back before testing.
    /*
    srand(time(NULL));
    for (int i = 0; i < HIDDEN_SIZE; i++)
        for (int j = 0; j < INPUT_SIZE; j++)
            net->W1[i][j] = ((double)rand() / RAND_MAX) * 0.01;

    for (int i = 0; i < OUTPUT_SIZE; i++)
        for (int j = 0; j < HIDDEN_SIZE; j++)
            net->W2[i][j] = ((double)rand() / RAND_MAX) * 0.01;
    */
    return net;
}

void forward(NeuralNetwork* net, double* input, double* hidden, double* output) {
    // TODO: Use cublas SGEMV.
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        hidden[i] = net->b1[i];
        for (int j = 0; j < INPUT_SIZE; j++)
            hidden[i] += net->W1[i][j] * input[j];
    }

    relu(hidden, HIDDEN_SIZE);

    // TODO: Use cublas SGEMV.
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output[i] = net->b2[i];
        for (int j = 0; j < HIDDEN_SIZE; j++)
            output[i] += net->W2[i][j] * hidden[j];
    }

    softmax(output, OUTPUT_SIZE);
}

// Weights initialization done using GPU.
__global__ void initializeMatrix(double* matrix, int size, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curandState state;
        curand_init(seed, idx, 0, &state);  // Initialize random state
        matrix[idx] = curand_uniform(&state); // Generate random number between 0 and 1
    }
}

void initialize_weights_gpu() {
    // Allocate device memory
    cudaMalloc((void**)&d_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(double));
    cudaMalloc((void**)&d_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double));
    cudaMalloc((void**)&d_B1, HIDDEN_SIZE * sizeof(double));


    // Define grid and block dimensions
    int blockSize = 256;
    int gridSize_W1 = (HIDDEN_SIZE * INPUT_SIZE + blockSize - 1) / blockSize;
    int gridSize_W2 = (OUTPUT_SIZE * HIDDEN_SIZE + blockSize - 1) / blockSize;

    // Launch kernel to initialize matrices on GPU
    initializeMatrix<<<gridSize_W1, blockSize>>>(d_W1, HIDDEN_SIZE * INPUT_SIZE, time(NULL));
    initializeMatrix<<<gridSize_W2, blockSize>>>(d_W2, OUTPUT_SIZE * HIDDEN_SIZE, time(NULL) + 1);
    cudaMemset(d_B1, 0, HIDDEN_SIZE * sizeof(double));

    // Wait for GPU to finish
    cudaDeviceSynchronize();
}

__global__ void add_bias(double* d_hidden, double* d_B1, int hidden_size) {
    int index = threadIdx.x; // threadIdx.x will directly give us the index in this case
    if (index < hidden_size) {
        d_hidden[index] += d_B1[index];
    }
}

// Forward pass
void forward(double* b2, double* input, double* d_hidden, double* output) {
    double* d_input;
    cudaMalloc((void**)&d_input, INPUT_SIZE * sizeof(double));
    cudaMemcpy(d_input, input, INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    // TODO: Use cublas DGEMV.
    double alpha = 1.0f, beta = 0.0f;
    cublasDgemv(handle, CUBLAS_OP_N, HIDDEN_SIZE, INPUT_SIZE, &alpha, d_W1, HIDDEN_SIZE, d_input, 1, &beta, d_hidden, 1);

    // Adding the bias in it.
    dim3 blockDim(128); // Block size, since hidden_size is 128
    dim3 gridDim(1);    // Grid size, only one block is needed for 128 elements

    // CUDA kernel to add bias
    add_bias<<<gridDim, blockDim>>>(d_hidden, d_B1, HIDDEN_SIZE);
    cudaDeviceSynchronize();

    // NOTE: RELU is now made into kernel to avoid cudaMemcpy(hidden, d_hidden).
    relu(d_hidden, HIDDEN_SIZE);

    // TODO: Convert output[OUTPUT_SIZE] to CUDA variable.
    double* d_output;
    cudaMalloc((void**)&d_output, OUTPUT_SIZE * sizeof(double));
    cudaMemcpy(d_output, output, OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);

    // TODO: Use cublas SGEMV.
    cublasDgemv(handle, CUBLAS_OP_N, OUTPUT_SIZE, HIDDEN_SIZE, &alpha, d_W2, OUTPUT_SIZE, d_hidden, 1, &beta, d_output, 1);

    cudaMemcpy(output, d_output, OUTPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost);

    for (int index = 0; index < OUTPUT_SIZE; index++) {
        output[index] += b2[index];
    }

    softmax(output, OUTPUT_SIZE);

    cublasDestroy(handle);
    cudaFree(d_output);
    cudaFree(d_input);
}

__global__ void relu_derivative_and_multiply(double* d_x, double* d_out, double* d_hidden, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        // Apply ReLU derivative and element-wise multiplication in a single step.
        double relu_derivative = (d_x[index] > 0.0f) ? 1.0f : 0.0f;
        d_out[index] = d_hidden[index] * relu_derivative;
    }
}

__global__ void update_b1(double* d_B1, double* hidden_d, double learning_rate, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        d_B1[index] -= learning_rate * hidden_d[index];
    }
}


// Backpropagation
void backward(double* b2, double* input, double* d_hidden, double* output, double* target) {
    // --- Error checking for all CUDA operations ---
    #define CHECK_CUDA(call) { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    }

    // --- 1. Allocate Device Memory ---
    double *d_input, *hidden_d, *dd_output;
    CHECK_CUDA(cudaMalloc(&d_input, INPUT_SIZE * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&hidden_d, HIDDEN_SIZE * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&dd_output, OUTPUT_SIZE * sizeof(double)));

    // --- 2. Copy Input Data ---
    CHECK_CUDA(cudaMemcpy(d_input, input, INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));

    // --- 3. Compute Output Gradient ---
    double h_d_output[OUTPUT_SIZE];
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        h_d_output[i] = output[i] - target[i];
    }
    CHECK_CUDA(cudaMemcpy(dd_output, h_d_output, OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));

    // --- 4. Get CUBLAS Handle (should be created once at startup) ---
    cublasHandle_t handle;
    cublasCreate(&handle);

    // --- 5. Hidden Layer Gradient ---
    double alpha = 1.0, beta = 0.0;
    cublasDgemv(handle, CUBLAS_OP_T,
               HIDDEN_SIZE, OUTPUT_SIZE,
               &alpha, d_W2, HIDDEN_SIZE,
               dd_output, 1, &beta, hidden_d, 1);

    // Apply ReLU derivative
    relu_derivative_and_multiply<<<(HIDDEN_SIZE+127)/128, 128>>>(d_hidden, hidden_d, d_hidden, HIDDEN_SIZE);
    CHECK_CUDA(cudaDeviceSynchronize());

    // --- 6. Weight Updates ---
    double lr = -LEARNING_RATE;

    // Update W2 (output layer)
    cublasDger(handle, OUTPUT_SIZE, HIDDEN_SIZE,
              &lr, dd_output, 1, d_hidden, 1, d_W2, OUTPUT_SIZE);

    // Update W1 (hidden layer)
    cublasDger(handle, HIDDEN_SIZE, INPUT_SIZE,
              &lr, hidden_d, 1, d_input, 1, d_W1, HIDDEN_SIZE);

    // --- 7. Bias Updates ---
    update_b1<<<(HIDDEN_SIZE+127)/128, 128>>>(d_B1, hidden_d, LEARNING_RATE, HIDDEN_SIZE);

    // Update output bias (b2)
    CHECK_CUDA(cudaMemcpy(h_d_output, dd_output, OUTPUT_SIZE*sizeof(double), cudaMemcpyDeviceToHost));
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        b2[i] -= LEARNING_RATE * h_d_output[i];
    }

    // --- 8. Cleanup ---
    cublasDestroy(handle);
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(hidden_d));
    CHECK_CUDA(cudaFree(dd_output));
}

// Train network
void train(double* b2, double** images, double** labels, int numImages) {
    clock_t total_start = clock();
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        clock_t epoch_start = clock();
        double loss = 0.0;
        int correct = 0;

        for (int i = 0; i < numImages; i++) {
            // TODO (DONE): Make hidden[HIDDEN_SIZE] as GPU side variable.
            //double hidden[HIDDEN_SIZE], output[OUTPUT_SIZE];
            double output[OUTPUT_SIZE];
            double *d_hidden;
            cudaMalloc((void**)&d_hidden, HIDDEN_SIZE * sizeof(double));

            forward(b2, images[i], d_hidden, output);
            backward(b2, images[i], d_hidden, output, labels[i]);

            // Compute loss & accuracy
            for (int k = 0; k < OUTPUT_SIZE; k++) loss -= labels[i][k] * log(output[k]);
            int pred = 0, actual = 0;
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                if (output[j] > output[pred]) pred = j;
                if (labels[i][j] > labels[i][actual]) actual = j;
            }
            if (pred == actual) correct++;

            cudaFree(d_hidden);
        }

        printf("Epoch %d - Loss: %.4f - Train Accuracy: %.2f%% - Time: %.3fs\n",
               epoch + 1, loss / numImages, (correct / (double)numImages) * 100, get_time(epoch_start));
    }
    printf("Total training time: %.3fs\n", get_time(total_start));
}

// Evaluate accuracy on test data
void evaluate(NeuralNetwork* net, double** images, double** labels, int numImages) {
    int correct = 0;
    for (int i = 0; i < numImages; i++) {
        double hidden[HIDDEN_SIZE], output[OUTPUT_SIZE];
        forward(net, images[i], hidden, output);
        int pred = 0, actual = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            if (output[j] > output[pred]) pred = j;
            if (labels[i][j] > labels[i][actual]) actual = j;
        }
        if (pred == actual) correct++;
    }
    printf("Test Accuracy: %.2f%%\n", (correct / (double)numImages) * 100);
}

// Read MNIST dataset
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

            // fread(&pixel, sizeof(unsigned char), 1, file);
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
        // fread(&label, sizeof(unsigned char), 1, file);
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
    free(net);

    cudaFree(d_B1);
    cudaFree(d_W1);
    cudaFree(d_W2);
}


// Main function
int main() {
    printf("MNIST Neural Network\n\n");

    double** train_images = loadMNISTImages("data/train-images.idx3-ubyte", 60000);
    double** train_labels = loadMNISTLabels("data/train-labels.idx1-ubyte", 60000);
    double** test_images = loadMNISTImages("data/t10k-images.idx3-ubyte", 10000);
    double** test_labels = loadMNISTLabels("data/t10k-labels.idx1-ubyte", 10000);

    NeuralNetwork* net = createNetwork();

    // TODO (DONE): Make W1 and W2 as GPU variables.
    initialize_weights_gpu();
    train(net->b2, train_images, train_labels, 60000);

    // TODO (DONE): Before testing, cudaMemcpy(CPU weights, GPU weights).
    cudaMemcpy(net->W1, d_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(net->W2, d_W2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(net->b1, d_B1, HIDDEN_SIZE * sizeof(double), cudaMemcpyDeviceToHost);

    evaluate(net, test_images, test_labels, 10000);

    freeNetwork(net);
    return 0;
}