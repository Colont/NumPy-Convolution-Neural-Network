import numpy as np
from mnist_data import main

def generate_kernel(kernel_size):
    kernel = np.random.randn(kernel_size, kernel_size) * 0.1
    return kernel

def convolution(matrix, kernel):
    '''
    Performs convolution and activation all in one function
    matrix: input_matrix
    kernel: the kernel matrix used in convolution
    '''
    batch_size, matrix_height, matrix_width = matrix.shape
    kernel_height, kernel_width = kernel.shape

    output_height = matrix_height - kernel_height + 1
    output_width = matrix_width - kernel_width + 1

    conv_layer1 = np.zeros((batch_size, output_height, output_width))
    for x in range(batch_size):
        for i in range(output_height):
            for j in range(output_width):
                submatrix = matrix[x, i:i+kernel_height, j:j+kernel_height]
                conv_layer1[x, i,j] = np.sum(submatrix * kernel)

    ReLU = np.maximum(0, conv_layer1)
    return conv_layer1, ReLU

def max_pooling(matrix, pool_size):
    '''
    Performs max pooling on specific convolution layers
    matrix: input matrix
    pool_size: the size in which we pool by
    '''
    batch_size, m, n = matrix.shape
    output_height = m // pool_size
    output_width = n // pool_size

    pooled_matrix = np.zeros((batch_size, output_height, output_width))
    for x in range(batch_size):
        for i in range(output_height):
            for j in range(output_width):
                i_start, i_end = i*pool_size, (i+1)*pool_size
                j_start, j_end = j*pool_size, (j+1)*pool_size
                pooled_matrix[x, i, j] = np.max(matrix[x, i_start:i_end, j_start:j_end])

    return pooled_matrix

def initialize_parameters(input_size, output_size):
    """Initialize weights and biases for dense layer"""
    scale = np.sqrt(2.0 / (input_size + output_size)) # Xavier initialization
    weights = np.random.randn(input_size, output_size) * scale
    biases = np.zeros((1, output_size))
    return weights, biases

def dense_layer(input_vector, weights, biases):
    return np.dot(input_vector, weights) + biases

def softmax(x):
    '''
    Implement softmax with numerical stability
    '''
    x = x - np.max(x, axis=1, keepdims=True)  # Prevent overflow
    exp_logits = np.exp(x)
    prob = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    return prob

def forward_pass(image, kernels, weights, biases):
    '''
    Full forward pass through the network
    Returns probabilities and intermediate values
    '''
    # Conv + ReLU
    conv1, ReLU1 = convolution(image, kernels[0])
    pooled_1 = max_pooling(conv1, 2) # 12x12

    conv2, ReLU2 = convolution(pooled_1, kernels[1])
    pooled_2 = max_pooling(conv2, pool_size=2)
    
    # Flatten
    flattened = pooled_2.reshape(-1, 16)
    
    # Dense layer
    logits = dense_layer(flattened, weights, biases)
    
    # Softmax
    prob = softmax(logits)
    
    vals = {
        "z1": conv1, "a1": ReLU1, "p1": pooled_1,
        "z2": conv2, "a2": ReLU2, "p2": pooled_2,
        'flattened': flattened
    }

    return prob, vals

def cross_entropy_loss(probabilities, labels):
    '''
    Calculate cross-entropy loss
    '''
    n = len(labels)
    probabilities_of_right = np.zeros_like(probabilities)
    probabilities_of_right[np.arange(n), labels] = 1
    return -np.sum(np.log(probabilities_of_right + 1e-10)) / n