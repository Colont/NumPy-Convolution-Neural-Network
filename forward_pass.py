import numpy as np


def generate_kernel(kernel_size):
    """Randomly initialize a convolution kernel"""
    return np.random.randn(kernel_size, kernel_size) * 0.1

def convolution(matrix, kernel):
    """
    Perform convolution with ReLU activation
    matrix: (batch, height, width)
    kernel: (k_height, k_width)
    Returns: pre-activation, post-ReLU
    """
    batch_size, matrix_height, matrix_width = matrix.shape
    k_height, k_width = kernel.shape

    out_h = matrix_height - k_height + 1
    out_w = matrix_width - k_width + 1

    conv_out = np.zeros((batch_size, out_h, out_w))
    for b in range(batch_size):
        for i in range(out_h):
            for j in range(out_w):
                conv_out[b, i, j] = np.sum(matrix[b, i:i+k_height, j:j+k_width] * kernel)

    return conv_out, np.maximum(0, conv_out)

def max_pooling(matrix, pool_size):
    """Max pooling (non-overlapping)"""
    batch_size, h, w = matrix.shape
    out_h, out_w = h // pool_size, w // pool_size
    pooled = np.zeros((batch_size, out_h, out_w))

    for b in range(batch_size):
        for i in range(out_h):
            for j in range(out_w):
                h_start, h_end = i*pool_size, (i+1)*pool_size
                w_start, w_end = j*pool_size, (j+1)*pool_size
                pooled[b, i, j] = np.max(matrix[b, h_start:h_end, w_start:w_end])
    return pooled

def initialize_parameters(input_size, output_size):
    """Dense layer weights and biases (Xavier)"""
    scale = np.sqrt(2.0 / (input_size + output_size))
    weights = np.random.randn(input_size, output_size) * scale
    biases = np.zeros((1, output_size))
    return weights, biases

def dense_layer(inputs, weights, biases):
    return np.dot(inputs, weights) + biases

def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)  # numerical stability
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def forward_pass(image, kernels, weights, biases):
    """
    Forward pass through:
    Conv1 → ReLU → MaxPool1 → Conv2 → ReLU → MaxPool2 → Flatten → Dense → Softmax
    """
    # Ensure batch dimension
    if image.ndim == 2:  # single image
        image = image[np.newaxis, :, :]
    batch_size = image.shape[0]

    # Conv1 + ReLU + MaxPool1
    conv1, ReLU1 = convolution(image, kernels[0])
    pooled1 = max_pooling(conv1, 2)

    # Conv2 + ReLU + MaxPool2
    conv2, ReLU2 = convolution(pooled1, kernels[1])
    pooled2 = max_pooling(conv2, 2)

    # Flatten
    flattened = pooled2.reshape(batch_size, -1)

    # Dense layer
    logits = dense_layer(flattened, weights, biases)

    # Softmax
    prob = softmax(logits)

    # Store intermediate values for backprop
    vals = {
        "z1": conv1, "a1": ReLU1, "p1": pooled1,
        "z2": conv2, "a2": ReLU2, "p2": pooled2,
        "flattened": flattened,
        "input_to_conv1": image, 
        "input_to_conv2": pooled1 
    }

    return prob, vals

def cross_entropy_loss(probabilities, labels):
    """Batch cross-entropy loss"""
    n = len(labels)
    return -np.sum(np.log(probabilities[np.arange(n), labels] + 1e-10)) / n