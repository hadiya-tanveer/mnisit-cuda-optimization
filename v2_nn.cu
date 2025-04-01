/*
* This implementation is the naive version of neural network classification.
* Here simple CUDA threads are launched which uses global memory for each access.
* No further optimization techniques are used in this version.
*/

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

    srand(time(NULL));
    for (int i = 0; i < HIDDEN_SIZE; i++)
        for (int j = 0; j < INPUT_SIZE; j++)
            net->W1[i][j] = ((double)rand() / RAND_MAX) * 0.01;

    for (int i = 0; i < OUTPUT_SIZE; i++)
        for (int j = 0; j < HIDDEN_SIZE; j++)
            net->W2[i][j] = ((double)rand() / RAND_MAX) * 0.01;

    return net;
}

// Forward pass
void forward(NeuralNetwork* net, double* input, double* hidden, double* output) {
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        hidden[i] = net->b1[i];
        for (int j = 0; j < INPUT_SIZE; j++)
            hidden[i] += net->W1[i][j] * input[j];
    }
    relu(hidden, HIDDEN_SIZE);

    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output[i] = net->b2[i];
        for (int j = 0; j < HIDDEN_SIZE; j++)
            output[i] += net->W2[i][j] * hidden[j];
    }
    softmax(output, OUTPUT_SIZE);
}

/* 
* Kernel to compute the gradients for the hidden layer (d_hidden) during backpropagation. 
* Each thread calculates the gradient for a specific hidden unit based on the weights (W2) and 
* the output layer's gradient (d_output).
*/
__global__ void compute_d_hidden(double* d_hidden, double* W2, double* d_output, double* hidden) {
    int i = threadIdx.x;
    if (i < HIDDEN_SIZE) {
        d_hidden[i] = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            d_hidden[i] += W2[j * HIDDEN_SIZE + i] * d_output[j];
        }
        d_hidden[i] *= (hidden[i] > 0);
    }
}

__global__ void updateW2(double* W2, double* d_output, double* hidden, double learning_rate, int hidden_size) {
    int row = blockIdx.x * blockDim.x + threadIdx.x; // Row index 
    int col = blockIdx.y * blockDim.y + threadIdx.y; // Column index

    if (row < OUTPUT_SIZE && col < HIDDEN_SIZE) {
        int idx = row * HIDDEN_SIZE + col; 
        W2[idx] -= learning_rate * d_output[row] * hidden[col];
    }
}

__global__ void updateW1(double* W1, double* d_hidden, double* input, double learning_rate, int input_size) {
    int row = blockIdx.x * blockDim.x + threadIdx.x; // Row index 
    int col = blockIdx.y * blockDim.y + threadIdx.y; // Column index 

    if (row < HIDDEN_SIZE && col < INPUT_SIZE) {
        int idx = row * INPUT_SIZE + col;
        W1[idx] -= learning_rate * d_hidden[row] * input[col];
    }
}

// Backpropagation
void backward(NeuralNetwork* net, double* input, double* hidden, double* output, double* target) {
    double d_output[OUTPUT_SIZE], d_hidden[HIDDEN_SIZE];

    // Compute output layer gradient
    for (int i = 0; i < OUTPUT_SIZE; i++)
        d_output[i] = output[i] - target[i];

    // Compute hidden layer gradient through CUDA threads.
    double *d_W1, *d_W2, *d_hidden_d, *d_output_d, *hidden_d, *input_d; 
    
    cudaMalloc(&input_d, IMAGE_SIZE_TRAIN * sizeof(double));  
    cudaMalloc(&d_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(double));
    cudaMalloc(&d_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double));
    cudaMalloc(&d_hidden_d, HIDDEN_SIZE * sizeof(double));
    cudaMalloc(&d_output_d, OUTPUT_SIZE * sizeof(double));
    cudaMalloc(&hidden_d, HIDDEN_SIZE * sizeof(double));
    
    cudaMemcpy(input_d, input, IMAGE_SIZE_TRAIN * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W1, net->W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W2, net->W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output_d, d_output, OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(hidden_d, hidden, HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice);

    compute_d_hidden<<<1, HIDDEN_SIZE>>>(d_hidden_d, d_W2, d_output_d, hidden_d);    
    // Update weights (gradient descent)
    // For W2
    dim3 blockDim(16, 16);
    dim3 gridDim((OUTPUT_SIZE + blockDim.x - 1) / blockDim.x, 
                (HIDDEN_SIZE + blockDim.y - 1) / blockDim.y);

    updateW2<<<gridDim, blockDim>>>(d_W2, d_output_d, hidden_d, LEARNING_RATE, HIDDEN_SIZE);
    cudaDeviceSynchronize();

    // For W1
    dim3 gridDim2((HIDDEN_SIZE + blockDim.x - 1) / blockDim.x, 
                (INPUT_SIZE + blockDim.y - 1) / blockDim.y);

    updateW1<<<gridDim2, blockDim>>>(d_W1, d_hidden_d, input_d, LEARNING_RATE, INPUT_SIZE);
    cudaDeviceSynchronize();

    for (int i = 0; i < OUTPUT_SIZE; i++)
        net->b2[i] -= LEARNING_RATE * d_output[i];

    for (int i = 0; i < HIDDEN_SIZE; i++)
        net->b1[i] -= LEARNING_RATE * d_hidden[i];

    cudaFree(d_W1);
    cudaFree(d_W2);
    cudaFree(d_hidden_d);
    cudaFree(d_output_d);
    cudaFree(hidden_d);
    cudaFree(input_d);
        
}

// Train network
void train(NeuralNetwork* net, double** images, double** labels, int numImages) {
    clock_t total_start = clock();
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        clock_t epoch_start = clock();
        double loss = 0.0;
        int correct = 0;

        for (int i = 0; i < numImages; i++) {
            // TODO: Convert these variables to GPU side variables and pass them to both 
            //       FORWARD() and BACKWARD() variables.
            double hidden[HIDDEN_SIZE], output[OUTPUT_SIZE];

            // TODO: Convert 'images' to GPU variable as well.
            
            forward(net, images[i], hidden, output);
            backward(net, images[i], hidden, output, labels[i]);

            // Compute loss & accuracy
            for (int k = 0; k < OUTPUT_SIZE; k++) loss -= labels[i][k] * log(output[k]);
            int pred = 0, actual = 0;
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                if (output[j] > output[pred]) pred = j;
                if (labels[i][j] > labels[i][actual]) actual = j;
            }
            if (pred == actual) correct++;
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
}


// Main function
int main() {
    printf("MNIST Neural Network\n\n");

    double** train_images = loadMNISTImages("data/train-images.idx3-ubyte", 60000);
    double** train_labels = loadMNISTLabels("data/train-labels.idx1-ubyte", 60000);
    double** test_images = loadMNISTImages("data/t10k-images.idx3-ubyte", 10000);
    double** test_labels = loadMNISTLabels("data/t10k-labels.idx1-ubyte", 10000);

    NeuralNetwork* net = createNetwork();
    train(net, train_images, train_labels, 60000);
    evaluate(net, test_images, test_labels, 10000);

    freeNetwork(net);
    return 0;
}
