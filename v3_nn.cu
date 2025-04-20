/*
* This is the optimized version of neural network.
* It uses launch configurations, optimized memory and communication,
* occupancy, CUDA supported variables, atomic functions.
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
#define LEARNING_RATE 0.01f
#define EPOCHS 3

#define HIDDEN_BLOCK 512   
#define OUTPUT_BLOCK 512   
#define UPDATE_BLOCK_X 16
#define UPDATE_BLOCK_Y 16

double get_time(clock_t start) {
    return (double)(clock() - start) / CLOCKS_PER_SEC;
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

// Helper to flatten a 2D matrix into a contiguous array.
float* flattenMatrix(float** matrix, int rows, int cols) {
    float* flat = (float*)malloc(rows * cols * sizeof(float));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            flat[i * cols + j] = matrix[i][j];
        }
    }
    return flat;
}

typedef struct {
    // Device parameters only
    float* d_W1;
    float* d_W2;
    float* d_b1;
    float* d_b2;
    
    // Device buffers
    float* d_input;
    float* d_hidden;
    float* d_output;
    float* d_d_output;
    float* d_d_hidden;
} NeuralNetwork;

NeuralNetwork* createNetwork() {
    NeuralNetwork* net = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    
    // Temporary host buffers
    float* h_W1 = (float*)malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(float));
    float* h_W2 = (float*)malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    float* h_b1 = (float*)calloc(HIDDEN_SIZE, sizeof(float));
    float* h_b2 = (float*)calloc(OUTPUT_SIZE, sizeof(float));

    // Initialize weights
    srand(time(NULL));
    for (int i = 0; i < HIDDEN_SIZE*INPUT_SIZE; i++) 
        h_W1[i] = ((float)rand()/RAND_MAX) * 0.01f;
    for (int i = 0; i < OUTPUT_SIZE*HIDDEN_SIZE; i++) 
        h_W2[i] = ((float)rand()/RAND_MAX) * 0.01f;

    // Device allocations
    cudaMalloc(&net->d_W1, HIDDEN_SIZE*INPUT_SIZE*sizeof(float));
    cudaMalloc(&net->d_W2, OUTPUT_SIZE*HIDDEN_SIZE*sizeof(float));
    cudaMalloc(&net->d_b1, HIDDEN_SIZE*sizeof(float));
    cudaMalloc(&net->d_b2, OUTPUT_SIZE*sizeof(float));
    cudaMalloc(&net->d_input, INPUT_SIZE*sizeof(float));
    cudaMalloc(&net->d_hidden, HIDDEN_SIZE*sizeof(float));
    cudaMalloc(&net->d_output, OUTPUT_SIZE*sizeof(float));
    cudaMalloc(&net->d_d_output, OUTPUT_SIZE*sizeof(float));
    cudaMalloc(&net->d_d_hidden, HIDDEN_SIZE*sizeof(float));
    // Copy to device
    cudaMemcpy(net->d_W1, h_W1, HIDDEN_SIZE*INPUT_SIZE*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(net->d_W2, h_W2, OUTPUT_SIZE*HIDDEN_SIZE*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(net->d_b1, h_b1, HIDDEN_SIZE*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(net->d_b2, h_b2, OUTPUT_SIZE*sizeof(float), cudaMemcpyHostToDevice);

    free(h_W1);
    free(h_W2);
    free(h_b1);
    free(h_b2);

    return net;
}

__global__ void forward_hidden(const float* W1, const float* b1, const float* input, float* hidden) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= HIDDEN_SIZE) return;
    
    float sum = b1[idx];
    for (int j = 0; j < INPUT_SIZE; j++) {
        sum += W1[idx * INPUT_SIZE + j] * input[j];
    }

    hidden[idx] = fmaxf(0.0f, sum);
}

__global__ void forward_output(const float* W2, const float* b2, const float* hidden, float* output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= OUTPUT_SIZE) return;
    
    float sum = b2[idx];
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        sum += W2[idx * HIDDEN_SIZE + j] * hidden[j];
    }

    output[idx] = sum;
}

__global__ void compute_d_hidden(float* d_hidden, const float* W2, const float* d_output, const float* hidden) {
    int idx = threadIdx.x;
    if (idx >= HIDDEN_SIZE) return;

    float gradient = 0.0f;
    for (int j = 0; j < OUTPUT_SIZE; j++) {
        gradient += W2[j * HIDDEN_SIZE + idx] * d_output[j];
    }

    d_hidden[idx] = gradient * (hidden[idx] > 0.0f ? 1.0f : 0.0f);
}

void softmax(float* x, int size) {
    float max_val = x[0];
    for(int i=1; i<size; i++) 
        if(x[i] > max_val) max_val = x[i];
    
    float sum = 0.0f;
    for(int i=0; i<size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for(int i=0; i<size; i++) 
        x[i] /= sum;
}

__global__ void update_weights_W2(float* W2, const float* d_output, const float* hidden, float lr) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if(row < OUTPUT_SIZE && col < HIDDEN_SIZE) {
        atomicAdd(&W2[row*HIDDEN_SIZE + col], 
                  -lr * d_output[row] * hidden[col]);
    }
}

__global__ void updateBiases(float* biases, const float* deltas, int size, float lr) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size) 
        atomicAdd(&biases[idx], -lr * deltas[idx]);
}

__global__ void update_weights_W1(float* W1, const float* d_hidden, const float* input, float lr) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < HIDDEN_SIZE && col < INPUT_SIZE) {
        atomicAdd(&W1[row * INPUT_SIZE + col], -lr * d_hidden[row] * input[col]);
    }
}

void forward(NeuralNetwork* net, float* input, float* output) {
    cudaMemcpy(net->d_input, input, INPUT_SIZE*sizeof(float), cudaMemcpyHostToDevice);
    
    forward_hidden<<<(HIDDEN_SIZE+HIDDEN_BLOCK-1)/HIDDEN_BLOCK, HIDDEN_BLOCK>>>(
        net->d_W1, net->d_b1, net->d_input, net->d_hidden);
    
    forward_output<<<(OUTPUT_SIZE+OUTPUT_BLOCK-1)/OUTPUT_BLOCK, OUTPUT_BLOCK>>>(
        net->d_W2, net->d_b2, net->d_hidden, net->d_output);
    
    cudaMemcpy(output, net->d_output, OUTPUT_SIZE*sizeof(float), cudaMemcpyDeviceToHost);
    softmax(output, OUTPUT_SIZE);
}

void backward(NeuralNetwork* net, float* output, float* target) {
    float d_output[OUTPUT_SIZE];
    
    // Compute output layer gradients
    for (int i = 0; i < OUTPUT_SIZE; i++) 
        d_output[i] = output[i] - target[i];
    
    // Copy output gradients to device
    cudaMemcpy(net->d_d_output, d_output, OUTPUT_SIZE*sizeof(float), cudaMemcpyHostToDevice);

    // Compute hidden layer gradients BEFORE any updates
    compute_d_hidden<<<1, HIDDEN_SIZE>>>(net->d_d_hidden, net->d_W2, net->d_d_output, net->d_hidden);
    cudaDeviceSynchronize();

    // Update OUTPUT LAYER (W2 and b2)
    dim3 blockDim2(UPDATE_BLOCK_X, UPDATE_BLOCK_Y);
    dim3 gridDim2((OUTPUT_SIZE + blockDim2.x -1)/blockDim2.x, 
                  (HIDDEN_SIZE + blockDim2.y -1)/blockDim2.y);
    update_weights_W2<<<gridDim2, blockDim2>>>(net->d_W2, net->d_d_output, net->d_hidden, LEARNING_RATE);
    updateBiases<<<(OUTPUT_SIZE+255)/256, 256>>>(net->d_b2, net->d_d_output, OUTPUT_SIZE, LEARNING_RATE);

    // 5. Update HIDDEN LAYER (W1 and b1)
    dim3 blockDim1(UPDATE_BLOCK_X, UPDATE_BLOCK_Y);
    dim3 gridDim1((HIDDEN_SIZE + blockDim1.x -1)/blockDim1.x, 
                  (INPUT_SIZE + blockDim1.y -1)/blockDim1.y);
    update_weights_W1<<<gridDim1, blockDim1>>>(net->d_W1, net->d_d_hidden, net->d_input, LEARNING_RATE);
    updateBiases<<<(HIDDEN_SIZE+255)/256, 256>>>(net->d_b1, net->d_d_hidden, HIDDEN_SIZE, LEARNING_RATE);
}

// Batch training function
void train(NeuralNetwork* net, float** images, float** labels, int numImages) {
    clock_t total_start = clock();
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        clock_t epoch_start = clock();
        float loss = 0.0f;
        int correct = 0;
        for (int i = 0; i < numImages; i ++) {
            float output[OUTPUT_SIZE];
            forward(net, images[i], output);
            backward(net, output, labels[i]);

            // Compute cross-entropy loss and accuracy.
            for (int k = 0; k < OUTPUT_SIZE; k++) {
                loss -= labels[i][k] * logf(output[k] + 1e-10f); // Add epsilon to prevent log(0)
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

// Evaluate accuracy on test images.
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

// Read MNIST images.
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

// Read MNIST labels.
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

// Free network memory, including device memory.
void freeNetwork(NeuralNetwork* net) {
    cudaFree(net->d_W1);
    cudaFree(net->d_W2);
    cudaFree(net->d_b1);
    cudaFree(net->d_b2);
    cudaFree(net->d_input);
    cudaFree(net->d_hidden);
    cudaFree(net->d_output);
    cudaFree(net->d_d_output);
    cudaFree(net->d_d_hidden);

    free(net);
}

// Main function.
int main() {
    printf("MNIST Neural Network V3\n\n");
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